"""Block Adaptive Quantization (BAQ) baseline (paper Sec. II-B, refs. [4], [27], [28]).

The raw-data matrix is divided into non-overlapping blocks (default: 128
complex samples of one range line, the classic BAQ block).  For each block the
standard deviation sigma of the I/Q samples is estimated and transmitted
(``sigma_bits`` bits, log-uniform code book), and every I and Q sample is
quantized independently with the N-bit Lloyd-Max (MSE-optimal) quantizer of a
zero-mean Gaussian with that sigma.

Bit rate (bits per real sample)::

    N + sigma_bits / (2 * block_size) + (64 header bits) / (2 * n_samples)

With N = 2..5 this gives the BAQ curve of Figs. 4-7 (the paper reports it at the
nominal N bpp).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

__all__ = ["lloyd_max_gaussian", "BAQCodec", "lloyd_max_sqnr_db"]


def _phi(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    f = np.isfinite(x)
    out[f] = np.exp(-0.5 * x[f] ** 2) / math.sqrt(2.0 * math.pi)
    return out


def _Phi(x):
    x = np.asarray(x, dtype=np.float64)
    try:
        from scipy.special import ndtr

        return ndtr(x)
    except ImportError:  # pragma: no cover
        v = np.vectorize(lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))) if np.isfinite(t)
                         else (1.0 if t > 0 else 0.0))
        return v(x)


@lru_cache(maxsize=None)
def lloyd_max_gaussian(nbits: int, iters: int = 5000, tol: float = 1e-13) -> Tuple[np.ndarray, np.ndarray]:
    """(thresholds[2^N - 1], levels[2^N]) of the N-bit Lloyd-Max quantizer for N(0, 1)."""
    L = 2 ** int(nbits)
    levels = np.linspace(-1.0, 1.0, L) * min(3.0, 0.8 + 0.6 * nbits)
    for _ in range(iters):
        t = 0.5 * (levels[:-1] + levels[1:])
        e = np.concatenate([[-np.inf], t, [np.inf]])
        new = (_phi(e[:-1]) - _phi(e[1:])) / np.maximum(_Phi(e[1:]) - _Phi(e[:-1]), 1e-300)
        if np.max(np.abs(new - levels)) < tol:
            levels = new
            break
        levels = new
    t = 0.5 * (levels[:-1] + levels[1:])
    return t, levels


def lloyd_max_sqnr_db(nbits: int) -> float:
    """Theoretical SQNR of the Lloyd-Max quantizer for a Gaussian source."""
    t, lev = lloyd_max_gaussian(nbits)
    e = np.concatenate([[-np.inf], t, [np.inf]])
    # D = 1 - sum_k P_k * c_k^2 for centroid levels
    P = _Phi(e[1:]) - _Phi(e[:-1])
    D = 1.0 - float(np.sum(P * lev ** 2))
    return -10.0 * math.log10(D)


class BAQCodec:
    """N-bit block adaptive quantizer for complex raw data."""

    def __init__(self, nbits: int, block: Tuple[int, int] = (1, 128), sigma_bits: int = 8):
        if nbits < 1 or nbits > 8:
            raise ValueError("nbits must be in 1..8")
        self.nbits = int(nbits)
        self.block = (int(block[0]), int(block[1]))
        self.sigma_bits = int(sigma_bits)
        self.thr, self.lev = lloyd_max_gaussian(self.nbits)

    # ------------------------------------------------------------------ helpers
    def _block_sigma(self, x: np.ndarray) -> np.ndarray:
        ba, br = self.block
        h, w = x.shape
        nh, nw = -(-h // ba), -(-w // br)
        p = np.zeros((nh * ba, nw * br), dtype=np.float64)
        cnt = np.zeros_like(p)
        p[:h, :w] = x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2
        cnt[:h, :w] = 1.0
        ps = p.reshape(nh, ba, nw, br).sum(axis=(1, 3))
        cs = cnt.reshape(nh, ba, nw, br).sum(axis=(1, 3))
        return np.sqrt(ps / (2.0 * np.maximum(cs, 1.0)))

    def _sigma_codebook(self, smin: float, smax: float) -> np.ndarray:
        n = 2 ** self.sigma_bits
        cb = np.zeros(n)
        if smax <= 0:
            return cb
        smin = max(smin, smax * 1e-6)
        cb[1:] = np.exp(np.linspace(math.log(smin), math.log(smax), n - 1)) if n > 2 else smax
        return cb

    def _quantize_sigma(self, sig: np.ndarray, cb: np.ndarray) -> np.ndarray:
        idx = np.zeros(sig.shape, dtype=np.int64)
        pos = sig > 0
        if np.any(pos) and cb.size > 1:
            ls = np.log(sig[pos])
            lcb = np.log(cb[1:])
            j = np.clip(np.searchsorted(lcb, ls), 1, lcb.size - 1)
            pick = np.where(np.abs(lcb[j - 1] - ls) <= np.abs(lcb[j] - ls), j - 1, j)
            idx[pos] = pick + 1
        return idx

    def _expand(self, a: np.ndarray, shape) -> np.ndarray:
        ba, br = self.block
        h, w = shape
        return np.repeat(np.repeat(a, ba, axis=0), br, axis=1)[:h, :w]

    def _code(self, x: np.ndarray, cb: np.ndarray):
        """Quantize sigma per block and the I/Q samples of ``x`` with code book ``cb``."""
        sidx = self._quantize_sigma(self._block_sigma(x), cb)
        s = self._expand(cb[sidx], x.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ni = np.where(s > 0, x.real / s, 0.0)
            nq = np.where(s > 0, x.imag / s, 0.0)
        ci = np.searchsorted(self.thr, ni).astype(np.uint8)
        cq = np.searchsorted(self.thr, nq).astype(np.uint8)
        return sidx.astype(np.uint16), ci, cq

    @staticmethod
    def _range(sig: np.ndarray):
        pos = sig[sig > 0]
        return (float(pos.min()), float(pos.max())) if pos.size else (0.0, 0.0)

    # ------------------------------------------------------------------ API
    def encode(self, raw: np.ndarray) -> Dict:
        x = np.asarray(raw, dtype=np.complex64)
        smin, smax = self._range(self._block_sigma(x))
        sidx, ci, cq = self._code(x, self._sigma_codebook(smin, smax))
        return {"codes_i": ci, "codes_q": cq, "sigma_idx": sidx,
                "smin": smin, "smax": smax, "shape": x.shape}

    def decode(self, enc: Dict) -> np.ndarray:
        cb = self._sigma_codebook(enc["smin"], enc["smax"])
        s = self._expand(cb[enc["sigma_idx"].astype(np.int64)], enc["shape"])
        re = self.lev[enc["codes_i"]] * s
        im = self.lev[enc["codes_q"]] * s
        return (re + 1j * im).astype(np.complex64)

    def bits(self, shape) -> int:
        h, w = shape
        ba, br = self.block
        n_blocks = (-(-h // ba)) * (-(-w // br))
        return 2 * h * w * self.nbits + n_blocks * self.sigma_bits + 64

    def compress_decompress(self, raw: np.ndarray, chunk_lines: int = 2048):
        """Quantize + reconstruct a (possibly memory-mapped) matrix chunk by chunk."""
        h, w = raw.shape
        out = np.zeros((h, w), dtype=np.complex64)
        ba = self.block[0]
        step = max(ba, (chunk_lines // ba) * ba)
        # one sigma code book for the whole matrix (its range travels in the 64-bit header)
        sig = np.concatenate([self._block_sigma(np.asarray(raw[a:a + step], dtype=np.complex64)).ravel()
                              for a in range(0, h, step)])
        smin, smax = self._range(sig)
        cb = self._sigma_codebook(smin, smax)
        for a in range(0, h, step):
            blk = np.asarray(raw[a:a + step], dtype=np.complex64)
            sidx, ci, cq = self._code(blk, cb)
            out[a:a + step] = self.decode({"codes_i": ci, "codes_q": cq, "sigma_idx": sidx,
                                           "smin": smin, "smax": smax, "shape": blk.shape})
        bits = self.bits((h, w))
        info = {"bits": bits, "bpp": bits / (2.0 * h * w), "bpp_nominal": float(self.nbits)}
        return out, info
