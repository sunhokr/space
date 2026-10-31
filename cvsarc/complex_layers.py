"""Complex-valued (CV) layers of the CV autoencoder (paper Sec. II-D-2 and Fig. 2).

All feature maps are native PyTorch complex tensors (``torch.complex64``).

* :class:`ComplexConv2d` / :class:`ComplexConvTranspose2d` implement
  ``(W_r + jW_i) * (x_r + jx_i) = (W_r*x_r - W_i*x_i) + j(W_r*x_i + W_i*x_r)``.
  For speed the four real convolutions are fused into one real convolution
  with a 2x2 block weight matrix.
* :class:`ComplexGDN` is the CV extension of Generalized Divisive Normalization
  (Balle et al.).  The default ``mode="magnitude"`` normalises every complex
  channel by a *real, positive* gain computed from the energies ``|x_j|^2``::

        GDN :  y_i = x_i / sqrt(beta_i + sum_j gamma_ij |x_j|^2)
        IGDN:  y_i = x_i * sqrt(beta_i + sum_j gamma_ij |x_j|^2)

  which leaves the phase of every sample untouched (phase preservation is the
  key requirement for SAR data).  ``mode="split"`` applies a real GDN to the
  real and imaginary parts separately (ablation option).

Back-propagation: PyTorch differentiates real-valued losses w.r.t. complex
tensors with Wirtinger calculus (paper Eq. (8)); no custom backward is needed.
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ComplexConv2d",
    "ComplexConvTranspose2d",
    "ComplexGDN",
    "LowerBound",
    "NonNegativeParametrizer",
    "complex_rayleigh_init_",
]

IntOrPair = Union[int, Tuple[int, int]]


def _pair(v: IntOrPair) -> Tuple[int, int]:
    return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))


def complex_rayleigh_init_(w_real: torch.Tensor, w_imag: torch.Tensor, fan_in: float) -> None:
    """Variance-preserving complex initialisation (Trabelsi et al., ICLR 2018).

    ``|w| ~ Rayleigh(sigma)`` and ``arg(w) ~ U(-pi, pi)`` with
    ``E|w|^2 = 2 sigma^2 = 1 / fan_in``.
    """
    sigma = 1.0 / math.sqrt(2.0 * max(float(fan_in), 1.0))
    with torch.no_grad():
        u = torch.rand_like(w_real).clamp_(1e-7, 1.0 - 1e-7)
        mag = sigma * torch.sqrt(-2.0 * torch.log(u))
        phase = (2.0 * torch.rand_like(w_real) - 1.0) * math.pi
        w_real.copy_(mag * torch.cos(phase))
        w_imag.copy_(mag * torch.sin(phase))


def _check_complex(x: torch.Tensor, name: str) -> None:
    if not torch.is_complex(x):
        raise TypeError(f"{name} expects a complex tensor, got {x.dtype}")


class ComplexConv2d(nn.Module):
    """Complex 2-D convolution (cross-correlation, like ``nn.Conv2d``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrPair = 5,
                 stride: IntOrPair = 1, padding: IntOrPair = 0, bias: bool = True):
        super().__init__()
        k = _pair(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = k
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.weight_real = nn.Parameter(torch.empty(out_channels, in_channels, *k))
        self.weight_imag = nn.Parameter(torch.empty(out_channels, in_channels, *k))
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        complex_rayleigh_init_(self.weight_real, self.weight_imag, fan_in=in_channels * k[0] * k[1])

    @property
    def weight(self) -> torch.Tensor:
        return torch.complex(self.weight_real, self.weight_imag)

    def _block_weight(self) -> torch.Tensor:
        wr, wi = self.weight_real, self.weight_imag
        # rows: [y_r; y_i], columns: [x_r, x_i]
        return torch.cat([torch.cat([wr, -wi], dim=1), torch.cat([wi, wr], dim=1)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_complex(x, "ComplexConv2d")
        b = None if self.bias_real is None else torch.cat([self.bias_real, self.bias_imag])
        y = F.conv2d(torch.cat([x.real, x.imag], dim=1), self._block_weight(), b,
                     self.stride, self.padding)
        yr, yi = y.chunk(2, dim=1)
        return torch.complex(yr, yi)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, bias={self.bias_real is not None}")


class ComplexConvTranspose2d(nn.Module):
    """Complex 2-D transposed convolution (like ``nn.ConvTranspose2d``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrPair = 5,
                 stride: IntOrPair = 1, padding: IntOrPair = 0, output_padding: IntOrPair = 0,
                 bias: bool = True):
        super().__init__()
        k = _pair(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = k
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.weight_real = nn.Parameter(torch.empty(in_channels, out_channels, *k))
        self.weight_imag = nn.Parameter(torch.empty(in_channels, out_channels, *k))
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        # every output sample receives on average in*k^2/stride^2 contributions
        fan_in = in_channels * k[0] * k[1] / float(self.stride[0] * self.stride[1])
        complex_rayleigh_init_(self.weight_real, self.weight_imag, fan_in=fan_in)

    @property
    def weight(self) -> torch.Tensor:
        return torch.complex(self.weight_real, self.weight_imag)

    def _block_weight(self) -> torch.Tensor:
        wr, wi = self.weight_real, self.weight_imag
        # conv_transpose weight layout is (in, out, kh, kw);
        # rows: inputs [x_r; x_i], columns: outputs [y_r, y_i]
        return torch.cat([torch.cat([wr, wi], dim=1), torch.cat([-wi, wr], dim=1)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_complex(x, "ComplexConvTranspose2d")
        b = None if self.bias_real is None else torch.cat([self.bias_real, self.bias_imag])
        y = F.conv_transpose2d(torch.cat([x.real, x.imag], dim=1), self._block_weight(), b,
                               self.stride, self.padding, self.output_padding)
        yr, yi = y.chunk(2, dim=1)
        return torch.complex(yr, yi)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, "
                f"output_padding={self.output_padding}, bias={self.bias_real is not None}")


class _LowerBoundFunction(torch.autograd.Function):
    """max(x, bound) whose gradient still flows when it pushes x upwards."""

    @staticmethod
    def forward(ctx, x, bound):  # noqa: D102
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):  # noqa: D102
        x, bound = ctx.saved_tensors
        pass_through = (x >= bound) | (grad_output < 0)
        return pass_through.to(grad_output.dtype) * grad_output, None


class LowerBound(nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.tensor([float(bound)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LowerBoundFunction.apply(x, self.bound.to(x.dtype))


class NonNegativeParametrizer(nn.Module):
    """Re-parametrisation ``value = max(p, sqrt(min + ped))^2 - ped`` used by GDN."""

    def __init__(self, minimum: float = 0.0, reparam_offset: float = 2 ** -18):
        super().__init__()
        self.minimum = float(minimum)
        self.register_buffer("pedestal", torch.tensor([float(reparam_offset) ** 2]))
        self.lower_bound = LowerBound((self.minimum + float(reparam_offset) ** 2) ** 0.5)

    def init(self, x: torch.Tensor) -> torch.Tensor:
        ped = self.pedestal.to(x.dtype)
        return torch.sqrt(torch.clamp(x + ped, min=float(ped)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lower_bound(x)
        return out * out - self.pedestal.to(x.dtype)


class ComplexGDN(nn.Module):
    """Complex Generalized Divisive Normalization (and its inverse, IGDN)."""

    def __init__(self, channels: int, inverse: bool = False, mode: str = "magnitude",
                 beta_min: float = 1e-6, gamma_init: float = 0.1):
        super().__init__()
        if mode not in ("magnitude", "split"):
            raise ValueError("mode must be 'magnitude' or 'split'")
        self.channels = int(channels)
        self.inverse = bool(inverse)
        self.mode = mode
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        self.beta = nn.Parameter(self.beta_reparam.init(torch.ones(channels)))
        self.gamma_reparam = NonNegativeParametrizer()
        self.gamma = nn.Parameter(self.gamma_reparam.init(gamma_init * torch.eye(channels)))

    def _normaliser(self, energy: torch.Tensor) -> torch.Tensor:
        C = self.channels
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma).reshape(C, C, 1, 1)
        return F.conv2d(energy, gamma, beta)

    def _gain(self, energy: torch.Tensor) -> torch.Tensor:
        norm = self._normaliser(energy)
        return torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_complex(x, "ComplexGDN")
        xr, xi = x.real, x.imag
        if self.mode == "magnitude":
            g = self._gain(xr * xr + xi * xi)
            return torch.complex(xr * g, xi * g)
        return torch.complex(xr * self._gain(xr * xr), xi * self._gain(xi * xi))

    def extra_repr(self) -> str:
        return f"{self.channels}, inverse={self.inverse}, mode={self.mode!r}"
