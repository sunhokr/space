"""Factorized-prior entropy model ("entropy bottleneck") of Balle et al.

Paper Sec. II-D-1: the rate term ``L_R = E[-log2 p(s)]`` (Eq. 2) is the
cross-entropy between the true symbol distribution and the learned model
``p``.  Following [13] (Balle et al., ICLR 2018, Appendix 6.1) every latent
channel gets its own flexible, non-parametric univariate density whose
cumulative is

    c(x) = sigmoid(f_K( ... f_1(x) ))
    f_k(x) = h_k(H_k x + b_k),   h_k(x) = x + tanh(a_k) * tanh(x)

and the probability mass of an integer bin is ``c(x + 1/2) - c(x - 1/2)``.

* training:  ``y~ = y + U(-1/2, 1/2)`` (additive uniform noise, paper Sec. II-D)
* inference: ``y^ = round(y - median) + median`` and arithmetic coding.

The complex latent of the CV autoencoder is fed to this module as
``2*M`` real channels (real parts followed by imaginary parts), i.e. the real
and imaginary component of every complex latent channel own separate densities.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import entropy_coder as ac
from .complex_layers import LowerBound

__all__ = ["EntropyBottleneck"]


class EntropyBottleneck(nn.Module):
    _TABLE_BUFFERS = ("_cdf", "_cdf_length", "_lo")

    def __init__(self, channels: int, tail_mass: float = 1e-9, init_scale: float = 10.0,
                 filters: Sequence[int] = (3, 3, 3), likelihood_bound: float = 1e-9):
        super().__init__()
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.likelihood_lower_bound = LowerBound(likelihood_bound)

        filters_ = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()
        for i in range(len(self.filters) + 1):
            init = math.log(math.expm1(1.0 / scale / filters_[i + 1]))
            self.matrices.append(nn.Parameter(torch.full((channels, filters_[i + 1], filters_[i]), init)))
            self.biases.append(nn.Parameter(torch.empty(channels, filters_[i + 1], 1).uniform_(-0.5, 0.5)))
            if i < len(self.filters):
                self.factors.append(nn.Parameter(torch.zeros(channels, filters_[i + 1], 1)))

        q = torch.tensor([-self.init_scale, 0.0, self.init_scale])
        self.quantiles = nn.Parameter(q.repeat(channels, 1, 1))  # (C, 1, 3)
        target = math.log(2.0 / self.tail_mass - 1.0)
        self.register_buffer("target", torch.tensor([-target, 0.0, target]))

        # integer CDF tables for the arithmetic coder (filled by update())
        self.register_buffer("_cdf", torch.zeros(0, 0, dtype=torch.int32))
        self.register_buffer("_cdf_length", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("_lo", torch.zeros(0, dtype=torch.int32))

    # ------------------------------------------------------------------ density
    def _logits_cumulative(self, inputs: torch.Tensor, stop_gradient: bool) -> torch.Tensor:
        logits = inputs  # (C, 1, N)
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            bias = self.biases[i]
            if stop_gradient:
                matrix = matrix.detach()
                bias = bias.detach()
            logits = torch.matmul(F.softplus(matrix), logits) + bias
            if i < len(self.factors):
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _likelihood(self, inputs: torch.Tensor, stop_gradient: bool = False) -> torch.Tensor:
        lower = self._logits_cumulative(inputs - 0.5, stop_gradient)
        upper = self._logits_cumulative(inputs + 0.5, stop_gradient)
        sign = -torch.sign(lower + upper).detach()  # evaluate in the numerically safe tail
        return torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

    def medians(self) -> torch.Tensor:
        return self.quantiles[:, 0, 1].detach()

    def loss(self) -> torch.Tensor:
        """Auxiliary loss that fits ``quantiles`` (tail_mass/2, median, 1-tail_mass/2)."""
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        return torch.abs(logits - self.target).sum()

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, training: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (noisy-or-quantized y, likelihoods); x has shape (B, C, H, W)."""
        if training is None:
            training = self.training
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.shape[1]}")
        v = x.permute(1, 0, 2, 3).contiguous()
        shape = v.shape
        v = v.reshape(self.channels, 1, -1)
        medians = self.quantiles[:, :, 1:2].detach()
        if training:
            out = v + torch.empty_like(v).uniform_(-0.5, 0.5)
        else:
            out = torch.round(v - medians) + medians
        lik = self.likelihood_lower_bound(self._likelihood(out))
        out = out.reshape(shape).permute(1, 0, 2, 3).contiguous()
        lik = lik.reshape(shape).permute(1, 0, 2, 3).contiguous()
        return out, lik

    # ------------------------------------------------------------------ coding tables
    @property
    def has_tables(self) -> bool:
        return self._cdf.numel() > 0

    @torch.no_grad()
    def update(self, force: bool = False) -> bool:
        """Compute the integer CDF tables used by the arithmetic coder."""
        if self.has_tables and not force:
            return False
        q = self.quantiles.detach()
        medians = q[:, 0, 1]
        # support of every channel; capped so that the CDF fits the 16-bit precision
        # (values outside the support are still coded losslessly via the escape symbol)
        cap = 8192
        minima = torch.ceil(medians - q[:, 0, 0]).to(torch.int64).clamp(min=0, max=cap)
        maxima = torch.ceil(q[:, 0, 2] - medians).to(torch.int64).clamp(min=0, max=cap)
        pmf_start = medians - minima.to(medians.dtype)
        pmf_length = (maxima + minima + 1)
        max_length = int(pmf_length.max().item())
        samples = torch.arange(max_length, device=q.device, dtype=q.dtype)
        samples = samples[None, None, :] + pmf_start[:, None, None]
        pmf = self._likelihood(samples, stop_gradient=True)[:, 0, :].double().cpu().numpy()

        C = self.channels
        lengths = pmf_length.cpu().numpy()
        cdf = np.zeros((C, max_length + 2), dtype=np.int64)
        cdf_length = np.zeros(C, dtype=np.int64)
        for c in range(C):
            L = int(lengths[c])
            p = pmf[c, :L]
            escape = max(1.0 - float(p.sum()), 1e-12)
            row = ac.pmf_to_quantized_cdf(np.concatenate([p, [escape]]))
            cdf[c, : L + 2] = row
            cdf[c, L + 2:] = ac.TOTAL  # padding, never used
            cdf_length[c] = L + 1  # symbols incl. escape
        dev = self.quantiles.device
        self._cdf = torch.from_numpy(cdf.astype(np.int32)).to(dev)
        self._cdf_length = torch.from_numpy(cdf_length.astype(np.int32)).to(dev)
        self._lo = (-minima).to(torch.int32).to(dev)
        return True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # coding tables have data-dependent shapes: resize before loading
        for name in self._TABLE_BUFFERS:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _tables_numpy(self):
        if not self.has_tables:
            raise RuntimeError("entropy coding tables missing: call model.update() first")
        return (self._cdf.cpu().numpy().astype(np.int64),
                self._cdf_length.cpu().numpy().astype(np.int64),
                self._lo.cpu().numpy().astype(np.int64))

    # ------------------------------------------------------------------ coding
    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> List[bytes]:
        """Quantize and arithmetic-code ``x`` (B, C, H, W) -> one byte string per item."""
        cdfs, lengths, lo = self._tables_numpy()
        B, C, H, W = x.shape
        med = self.quantiles[:, 0, 1].detach().reshape(1, C, 1, 1).to(x.dtype)
        symbols = torch.round(x - med).to(torch.int64).cpu().numpy()
        idx = np.repeat(np.arange(C, dtype=np.int64), H * W)
        return [ac.encode(symbols[b].reshape(-1), idx, cdfs, lengths, lo) for b in range(B)]

    @torch.no_grad()
    def decompress(self, strings: Sequence[bytes], size: Tuple[int, int]) -> torch.Tensor:
        cdfs, lengths, lo = self._tables_numpy()
        H, W = int(size[0]), int(size[1])
        C = self.channels
        idx = np.repeat(np.arange(C, dtype=np.int64), H * W)
        sym = np.stack([ac.decode(s, idx, cdfs, lengths, lo).reshape(C, H, W) for s in strings])
        dev = self.quantiles.device
        med = self.quantiles[:, 0, 1].detach().reshape(1, C, 1, 1)
        return torch.from_numpy(sym).to(device=dev, dtype=med.dtype) + med
