"""CV autoencoder-based neural compression model (paper Fig. 2) and RD loss.

Architecture (Fig. 2, all layers complex-valued, 5x5 kernels, stride 2,
zero padding 2, transposed convolutions with output padding 1)::

    encoder g_a : Conv(1->192) GDN  Conv(192->192) GDN  Conv(192->192) GDN  Conv(192->320)
    quantizer Q : y~ = y + U(-.5,.5) (training) / round (inference) on Re and Im
    entropy     : factorized prior + arithmetic coder (AE / AD)
    decoder g_s : TConv(320->192) IGDN TConv(192->192) IGDN TConv(192->192) IGDN TConv(192->1)

A 256 x 256 complex raw-data patch is mapped to a 320 x 16 x 16 complex latent.
The decoder has no activation on its output layer.

Rate-distortion loss (Eq. 1, 2, 7):  ``L = L_R + lambda * L_D`` with

* ``L_R`` = expected code length, reported in **bits per real sample**
  (bits / (2 * number of complex samples)), the same unit as the "bpp"
  axis of Figs. 4-7 where BAQ with N bits per I/Q sample sits at N bpp;
* ``L_D`` = mean squared error per real sample between the (normalised)
  input and output raw data (Eq. 7).

With unit-power normalisation (see codec.py), ``L_D = 1 / SQNR`` and the
optimal operating point satisfies approximately
``SQNR_dB ~ 10*log10(2*ln(2)*lambda)``; this guides the choice of lambda.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_layers import ComplexConv2d, ComplexConvTranspose2d, ComplexGDN
from .entropy_models import EntropyBottleneck

__all__ = ["CVSARCompressor", "RateDistortionLoss", "lambda_for_sqnr", "expected_sqnr_db",
           "polyphase_scale_for_lambda"]


def lambda_for_sqnr(sqnr_db: float) -> float:
    """Approximate lambda giving a target raw-data SQNR (6.02 dB/bit regime)."""
    return 10.0 ** (sqnr_db / 10.0) / (2.0 * math.log(2.0))


def expected_sqnr_db(lmbda: float) -> float:
    return 10.0 * math.log10(2.0 * math.log(2.0) * lmbda)


def polyphase_scale_for_lambda(lmbda: float) -> float:
    """Latent scale s with 12 s^2 = expected SQNR (uniform quantization, unit-power input)."""
    return math.sqrt(2.0 * math.log(2.0) * float(lmbda) / 12.0)


class CVSARCompressor(nn.Module):
    """Complex-valued autoencoder with factorized-prior entropy model."""

    def __init__(self, N: int = 192, M: int = 320, in_channels: int = 1,
                 gdn_mode: str = "magnitude", kernel_size: int = 5, entropy_init_scale: float = 10.0):
        super().__init__()
        self.N, self.M, self.in_channels = int(N), int(M), int(in_channels)
        self.gdn_mode = gdn_mode
        self.kernel_size = int(kernel_size)
        self.entropy_init_scale = float(entropy_init_scale)
        k, p = self.kernel_size, self.kernel_size // 2

        def conv(i, o):
            return ComplexConv2d(i, o, k, stride=2, padding=p)

        def tconv(i, o):
            return ComplexConvTranspose2d(i, o, k, stride=2, padding=p, output_padding=1)

        self.g_a = nn.Sequential(
            conv(in_channels, N), ComplexGDN(N, mode=gdn_mode),
            conv(N, N), ComplexGDN(N, mode=gdn_mode),
            conv(N, N), ComplexGDN(N, mode=gdn_mode),
            conv(N, M),
        )
        self.g_s = nn.Sequential(
            tconv(M, N), ComplexGDN(N, inverse=True, mode=gdn_mode),
            tconv(N, N), ComplexGDN(N, inverse=True, mode=gdn_mode),
            tconv(N, N), ComplexGDN(N, inverse=True, mode=gdn_mode),
            tconv(N, in_channels),
        )
        # real and imaginary part of each complex latent channel -> 2*M densities
        self.entropy_bottleneck = EntropyBottleneck(2 * M, init_scale=self.entropy_init_scale)

    # ------------------------------------------------------------------ helpers
    downsampling_factor = 16

    def config(self) -> Dict:
        return {"N": self.N, "M": self.M, "in_channels": self.in_channels, "gdn_mode": self.gdn_mode,
                "kernel_size": self.kernel_size, "entropy_init_scale": self.entropy_init_scale}

    @staticmethod
    def _to_real(y: torch.Tensor) -> torch.Tensor:
        return torch.cat([y.real, y.imag], dim=1)

    @staticmethod
    def _to_complex(y: torch.Tensor) -> torch.Tensor:
        r, i = y.chunk(2, dim=1)
        return torch.complex(r.contiguous(), i.contiguous())

    # 2x2 orthonormal butterflies acting on the polyphase components (p, q) -> index 2p+q
    _BASES = {
        "identity": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "haar": [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, -0.5, 0.5]],
    }

    @torch.no_grad()
    def init_polyphase_(self, latent_scale: float = 1.0, others_scale: float = 0.1,
                        basis: str = "identity") -> "CVSARCompressor":
        """Optional initialisation (NOT part of the paper): start from an exact transform.

        SAR raw data are almost white, so the network has to learn a near-identity,
        full-rank mapping through the 16x down-sampling, which is slow from a random
        start.  This sets the first 4 / 16 / 64 / 256 channels of the four encoder
        layers to a polyphase (space-to-depth) decomposition scaled by
        ``latent_scale`` and the decoder to its exact inverse (depth-to-space); all
        other channels keep a down-scaled random initialisation and the GDN / IGDN
        pairs start as (almost) identities.  With unit-power input, uniform
        quantization of the latents then gives ``SQNR ~ 12 * latent_scale^2``, i.e.
        training starts from a scalar-quantizer-like operating point
        (see :func:`polyphase_scale_for_lambda`).  Requires N >= 64, M >= 256 (true for
        the paper's N=192, M=320) and kernel size 5.

        ``basis="haar"`` replaces the pure polyphase split by the orthonormal 2x2 Haar
        butterfly at every layer, i.e. the 256 identity latents become a full
        Walsh-Hadamard wavelet-packet decomposition of each 16x16 block.  Its channels
        have unequal variances on band-limited (over-sampled) raw data, so the
        factorized entropy model exploits the coding gain from the first iteration.
        """
        if self.N < 64 or self.M < 256 or self.kernel_size != 5 or self.in_channels != 1:
            raise ValueError("polyphase init needs N >= 64, M >= 256, kernel 5, 1 input channel")
        H = self._BASES[basis]
        for m in list(self.g_a) + list(self.g_s):
            if isinstance(m, ComplexGDN):
                m.gamma.copy_(m.gamma_reparam.init(1e-5 * torch.eye(m.channels)))
        convs = [m for m in self.g_a if isinstance(m, ComplexConv2d)]
        tconvs = [m for m in self.g_s if isinstance(m, ComplexConvTranspose2d)]
        n_in = 1
        for li, conv in enumerate(convs):  # encoder: identity channels 0..4*n_in-1
            gain = float(latent_scale) if li == 0 else 1.0
            conv.weight_real.mul_(others_scale)
            conv.weight_imag.mul_(others_scale)
            for c in range(n_in):
                for k in range(4):
                    o = 4 * c + k
                    conv.weight_real[o].zero_()
                    conv.weight_imag[o].zero_()
                    conv.bias_real[o] = 0.0
                    conv.bias_imag[o] = 0.0
                    for p in range(2):
                        for q in range(2):
                            conv.weight_real[o, c, 2 + p, 2 + q] = gain * H[k][2 * p + q]
            n_in *= 4
        n_out = 1
        for li, tconv in enumerate(reversed(tconvs)):  # decoder, from the output layer backwards
            gain = 1.0 / float(latent_scale) if li == 0 else 1.0
            tconv.weight_real.mul_(others_scale)
            tconv.weight_imag.mul_(others_scale)
            for c in range(n_out):
                tconv.weight_real[:, c].zero_()
                tconv.weight_imag[:, c].zero_()
                tconv.bias_real[c] = 0.0
                tconv.bias_imag[c] = 0.0
                for k in range(4):  # orthonormal basis: synthesis = transpose of analysis
                    for p in range(2):
                        for q in range(2):
                            tconv.weight_real[4 * c + k, c, 2 + p, 2 + q] = gain * H[k][2 * p + q]
            n_out *= 4
        return self

    def main_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.endswith("quantiles")]

    def aux_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.endswith("quantiles")]

    def aux_loss(self) -> torch.Tensor:
        return self.entropy_bottleneck.loss()

    def update(self, force: bool = False) -> bool:
        return self.entropy_bottleneck.update(force=force)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not torch.is_complex(x):
            raise TypeError("input must be a complex tensor (B, 1, H, W)")
        y = self.g_a(x)
        y_hat_r, likelihoods = self.entropy_bottleneck(self._to_real(y))
        y_hat = self._to_complex(y_hat_r)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": likelihoods, "y": y, "y_hat": y_hat}

    # ------------------------------------------------------------------ coding
    def _check_size(self, x: torch.Tensor) -> None:
        h, w = x.shape[-2:]
        f = self.downsampling_factor
        if h % f or w % f:
            raise ValueError(f"spatial size must be a multiple of {f}, got {h}x{w}")

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Dict:
        self._check_size(x)
        y = self.g_a(x)
        strings = self.entropy_bottleneck.compress(self._to_real(y))
        return {"strings": strings, "shape": tuple(y.shape[-2:])}

    @torch.no_grad()
    def decompress(self, strings: Sequence[bytes], shape: Tuple[int, int]) -> Dict:
        y_hat = self._to_complex(self.entropy_bottleneck.decompress(strings, shape))
        return {"x_hat": self.g_s(y_hat), "y_hat": y_hat}

    # ------------------------------------------------------------------ checkpoints
    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state"])
        return model, ckpt


class RateDistortionLoss(nn.Module):
    """``L_RD = L_R + lambda * L_D`` (paper Eq. 1)."""

    def __init__(self, lmbda: float):
        super().__init__()
        self.lmbda = float(lmbda)

    def forward(self, output: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        num_real = 2.0 * target.numel()  # real samples (I and Q)
        bits = torch.sum(-torch.log2(output["likelihoods"]))
        bpp = bits / num_real
        err = output["x_hat"] - target
        mse = torch.sum(err.real ** 2 + err.imag ** 2) / num_real
        sig = torch.sum(target.real ** 2 + target.imag ** 2) / num_real
        loss = bpp + self.lmbda * mse
        sqnr = 10.0 * torch.log10(sig.detach() / mse.detach().clamp_min(1e-12))
        return {"loss": loss, "bpp": bpp, "mse": mse, "sqnr_db": sqnr}
