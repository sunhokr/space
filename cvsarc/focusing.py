"""Range-Doppler algorithm (RDA): stripmap raw data -> Single Look Complex image.

In the paper (Fig. 3) the quality of every compression method is measured on
the SLC images obtained by focusing the original and the decompressed raw data
(the rate is measured on the raw data).  This module implements a standard RDA:

1. range compression with the transmitted chirp replica
   (``exp(j2pi(f0 t + Kr t^2 / 2))``, ``0 <= t < Tp``), generalized-Hamming
   weighting over the chirp band;
2. Doppler-centroid estimation (average cross-correlation coefficient, ACCC);
3. azimuth FFT; range cell migration correction (RCMC) by windowed-sinc
   interpolation with the exact hyperbolic migration ``R0 (1/D - 1)``,
   ``D(f) = sqrt(1 - (lambda f / (2 Vr))^2)``;
4. azimuth matched filter ``exp(-j * s * 4 pi R0 (D - 1) / lambda)`` (keeps the
   propagation phase as in standard SLC products) with weighting over the
   processed Doppler bandwidth; azimuth IFFT.

All reconstructions of an experiment must be focused with *identical*
parameters: call :func:`focus` once on the reference raw data and pass
``info["params"]`` (Doppler centroid and sign resolved) to the other calls.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

C_LIGHT = 299792458.0

try:
    import scipy.fft as _sfft

    def _fft(a, n=None, axis=-1):
        return _sfft.fft(a, n=n, axis=axis, workers=-1)

    def _ifft(a, n=None, axis=-1):
        return _sfft.ifft(a, n=n, axis=axis, workers=-1)

    def next_fast_len(n: int) -> int:
        return _sfft.next_fast_len(int(n))
except ImportError:  # pragma: no cover
    def _fft(a, n=None, axis=-1):
        return np.fft.fft(a, n=n, axis=axis)

    def _ifft(a, n=None, axis=-1):
        return np.fft.ifft(a, n=n, axis=axis)

    def next_fast_len(n: int) -> int:
        return 1 << int(math.ceil(math.log2(max(1, n))))

__all__ = ["FocusParams", "chirp_replica", "range_compress", "estimate_doppler_centroid",
           "azimuth_compress", "focus", "valid_region", "image_contrast", "quicklook"]


@dataclass
class FocusParams:
    wavelength: float
    prf: float
    fs: float
    chirp_rate: float
    pulse_length: float
    near_range: float
    velocity: float
    velocity_near: Optional[float] = None
    velocity_far: Optional[float] = None
    chirp_start_freq: Optional[float] = None  # None -> centred chirp (-B/2 .. +B/2)
    doppler_centroid: Optional[float] = None  # None -> ACCC estimate
    azimuth_sign: int = -1  # -1: s ~ exp(-j4piR/lambda) (standard), +1, 0: automatic
    range_window_alpha: float = 0.75  # generalized Hamming (1.0 = rectangular)
    azimuth_window_alpha: float = 0.75
    azimuth_bandwidth: Optional[float] = None  # processed Doppler bandwidth [Hz]
    azimuth_bandwidth_fraction: float = 0.8  # used if azimuth_bandwidth is None
    rcmc: bool = True
    rcmc_taps: int = 8
    keep_propagation_phase: bool = True
    block_lines: int = 1024
    block_cols: int = 1024
    row_block: int = 128

    @classmethod
    def from_metadata(cls, meta: Dict, defaults: Optional[Dict] = None, **overrides) -> "FocusParams":
        """Precedence: dataclass defaults < ``defaults`` (e.g. config file) <
        ``meta['focusing']`` (dataset specific) < explicit ``overrides`` (not None)."""
        need = ("wavelength", "prf", "fs", "chirp_rate", "pulse_length", "near_range")
        missing = [k for k in need if meta.get(k) is None]
        v = meta.get("effective_velocity", meta.get("velocity"))
        if v is None and overrides.get("velocity") is None:
            missing.append("effective_velocity")
        if missing:
            raise KeyError(f"metadata lacks focusing parameters: {missing}")
        kw = dict(wavelength=float(meta["wavelength"]), prf=float(meta["prf"]), fs=float(meta["fs"]),
                  chirp_rate=float(meta["chirp_rate"]), pulse_length=float(meta["pulse_length"]),
                  near_range=float(meta["near_range"]),
                  velocity=float(v) if v is not None else 0.0,
                  velocity_near=meta.get("effective_velocity_near"),
                  velocity_far=meta.get("effective_velocity_far"),
                  chirp_start_freq=meta.get("chirp_start_freq"),
                  doppler_centroid=meta.get("doppler_centroid"))
        kw.update({k: val for k, val in (defaults or {}).items() if val is not None})
        kw.update(meta.get("focusing", {}) or {})
        kw.update({k: val for k, val in overrides.items() if val is not None})
        return cls(**kw)

    def to_dict(self) -> Dict:
        return asdict(self)

    # --------------------------------------------------------------- derived
    @property
    def range_spacing(self) -> float:
        return C_LIGHT / (2.0 * self.fs)

    @property
    def chirp_bandwidth(self) -> float:
        return abs(self.chirp_rate) * self.pulse_length

    @property
    def start_freq(self) -> float:
        if self.chirp_start_freq is None:
            return -0.5 * self.chirp_rate * self.pulse_length
        return float(self.chirp_start_freq)

    @property
    def processed_bandwidth(self) -> float:
        b = self.azimuth_bandwidth if self.azimuth_bandwidth else self.azimuth_bandwidth_fraction * self.prf
        return min(float(b), self.prf)

    def velocities(self, n: int) -> np.ndarray:
        if self.velocity_near is not None and self.velocity_far is not None:
            return np.linspace(float(self.velocity_near), float(self.velocity_far), n)
        return np.full(n, float(self.velocity))


def _gen_hamming(u: np.ndarray, alpha: float) -> np.ndarray:
    w = alpha + (1.0 - alpha) * np.cos(2.0 * np.pi * u)
    return np.where(np.abs(u) <= 0.5, w, 0.0)


def chirp_replica(fs: float, chirp_rate: float, pulse_length: float,
                  start_freq: Optional[float] = None) -> np.ndarray:
    n = max(1, int(round(pulse_length * fs)))
    t = np.arange(n) / fs
    f0 = -0.5 * chirp_rate * pulse_length if start_freq is None else float(start_freq)
    return np.exp(2j * np.pi * (f0 * t + 0.5 * chirp_rate * t * t)).astype(np.complex64)


def _range_filter(p: FocusParams, nfft: int) -> Tuple[np.ndarray, int]:
    rep = chirp_replica(p.fs, p.chirp_rate, p.pulse_length, p.chirp_start_freq)
    H = np.conj(np.fft.fft(rep.astype(np.complex128), nfft))
    f = np.fft.fftfreq(nfft, 1.0 / p.fs)
    B = p.chirp_bandwidth
    fc = p.start_freq + 0.5 * p.chirp_rate * p.pulse_length
    d = np.mod(f - fc + 0.5 * p.fs, p.fs) - 0.5 * p.fs
    if B > 0:
        H = H * _gen_hamming(d / B, p.range_window_alpha)
    return H.astype(np.complex64), rep.size


def range_compress(raw: np.ndarray, p: FocusParams, out: Optional[np.ndarray] = None) -> np.ndarray:
    Na, Nr = raw.shape
    Np = max(1, int(round(p.pulse_length * p.fs)))
    nfft = next_fast_len(Nr + Np - 1)
    H, _ = _range_filter(p, nfft)
    if out is None:
        out = np.empty((Na, Nr), dtype=np.complex64)
    for a0 in range(0, Na, p.block_lines):
        a1 = min(Na, a0 + p.block_lines)
        X = _fft(np.asarray(raw[a0:a1], dtype=np.complex64), n=nfft, axis=1)
        X *= H[None, :]
        out[a0:a1] = _ifft(X, axis=1)[:, :Nr]
    return out


def estimate_doppler_centroid(data: np.ndarray, prf: float, max_cols: int = 4096,
                              chunk: int = 4096) -> float:
    """ACCC Doppler-centroid estimate (fractional PRF part).

    ``fdc = PRF / (2 pi) * arg( sum_eta s(eta + 1) s*(eta) )`` over all lines and
    (a strided subset of at most ``max_cols``) range bins.
    """
    Na, Nr = data.shape
    cs = max(1, Nr // max_cols)
    acc = 0.0 + 0.0j
    for a0 in range(0, Na - 1, chunk):
        a1 = min(Na, a0 + chunk + 1)  # one line overlap -> every adjacent pair counted once
        sub = np.asarray(data[a0:a1, ::cs], dtype=np.complex128)
        acc += np.sum(sub[1:] * np.conj(sub[:-1]))
    return float(prf * np.angle(acc) / (2.0 * np.pi))


def doppler_frequencies(n: int, prf: float, fdc: float) -> np.ndarray:
    f = np.fft.fftfreq(n, 1.0 / prf)
    return fdc + np.mod(f - fdc + 0.5 * prf, prf) - 0.5 * prf


def _sinc_interp_rows(rows: np.ndarray, pos: np.ndarray, taps: int) -> np.ndarray:
    n, N = rows.shape
    i0 = np.floor(pos).astype(np.int64)
    frac = (pos - i0).astype(np.float32)
    half = taps // 2
    out = np.zeros(rows.shape, dtype=np.complex64)
    wsum = np.zeros(rows.shape, dtype=np.float32)
    for m in range(-half + 1, half + 1):
        idx = i0 + m
        d = frac - np.float32(m)
        w = (np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / half))).astype(np.float32)
        valid = (idx >= 0) & (idx < N)
        w = np.where(valid, w, np.float32(0.0))
        vals = np.take_along_axis(rows, np.clip(idx, 0, N - 1), axis=1)
        out += vals * w
        wsum += w
    nz = np.abs(wsum) > 1e-3
    out[nz] /= wsum[nz]
    return out


def azimuth_compress(rc: np.ndarray, p: FocusParams, fdc: float, sign: int,
                     inplace: bool = False) -> np.ndarray:
    Na, Nr = rc.shape
    work = rc if inplace else np.array(rc, dtype=np.complex64, copy=True)
    for c0 in range(0, Nr, p.block_cols):
        c1 = min(Nr, c0 + p.block_cols)
        work[:, c0:c1] = _fft(work[:, c0:c1], axis=0)
    f = doppler_frequencies(Na, p.prf, fdc)
    dr = p.range_spacing
    R = p.near_range + np.arange(Nr) * dr
    vr = p.velocities(Nr)
    W = _gen_hamming((f - fdc) / p.processed_bandwidth, p.azimuth_window_alpha)
    kk = np.arange(Nr, dtype=np.float64)[None, :]
    for j0 in range(0, Na, p.row_block):
        j1 = min(Na, j0 + p.row_block)
        fj = f[j0:j1, None]
        D = np.sqrt(np.clip(1.0 - (p.wavelength * fj / (2.0 * vr[None, :])) ** 2, 1e-6, None))
        rows = work[j0:j1]
        if p.rcmc:
            shift = R[None, :] * (1.0 / D - 1.0) / dr
            rows = _sinc_interp_rows(rows, kk + shift, p.rcmc_taps)
        Dm = (D - 1.0) if p.keep_propagation_phase else D
        phase = -float(sign) * 4.0 * np.pi * R[None, :] * Dm / p.wavelength
        work[j0:j1] = rows * (np.exp(1j * phase) * W[j0:j1, None]).astype(np.complex64)
    for c0 in range(0, Nr, p.block_cols):
        c1 = min(Nr, c0 + p.block_cols)
        work[:, c0:c1] = _ifft(work[:, c0:c1], axis=0)
    return work


def image_contrast(img: np.ndarray) -> float:
    I = np.abs(img).astype(np.float64) ** 2
    m = I.mean()
    return float((I * I).mean() / max(m * m, 1e-300))


def _auto_sign(rc: np.ndarray, p: FocusParams, fdc: float, cols: int = 512) -> int:
    Na, Nr = rc.shape
    c0 = max(0, Nr // 2 - cols // 2)
    c1 = min(Nr, c0 + cols)
    sub = np.ascontiguousarray(rc[:, c0:c1])
    q = replace(p, near_range=p.near_range + c0 * p.range_spacing)
    best, best_c = -1, -1.0
    for s in (-1, 1):
        c = image_contrast(azimuth_compress(sub, q, fdc, s))
        if c > best_c:
            best, best_c = s, c
    return best


def valid_region(shape: Tuple[int, int], p: FocusParams) -> Tuple[int, int, int, int]:
    """(a0, a1, r0, r1) of fully focused pixels (no wrap-around / truncated echoes)."""
    Na, Nr = shape
    Np = int(round(p.pulse_length * p.fs))
    r_far = p.near_range + (Nr - 1) * p.range_spacing
    vmin = float(np.min(p.velocities(2)))
    ka = 2.0 * vmin ** 2 / (p.wavelength * r_far)
    nap = int(math.ceil(p.processed_bandwidth / ka * p.prf))
    a0 = min(nap // 2 + 1, Na // 2)
    a1 = max(Na - nap // 2 - 1, a0 + 1)
    r1 = max(Nr - Np, 1)
    return a0, a1, 0, r1


def focus(raw: np.ndarray, p: FocusParams, verbose: bool = False):
    """Focus raw data. Returns ``(slc, info)``; ``info['params']`` is the resolved parameter set."""
    rc = range_compress(raw, p)
    fdc = p.doppler_centroid
    if fdc is None:
        fdc = estimate_doppler_centroid(rc, p.prf)
    sign = int(p.azimuth_sign)
    if sign == 0:
        sign = _auto_sign(rc, p, fdc)
    if verbose:
        print(f"  focusing: fdc = {fdc:.2f} Hz, azimuth sign = {sign:+d}")
    slc = azimuth_compress(rc, p, fdc, sign, inplace=True)
    q = replace(p, doppler_centroid=float(fdc), azimuth_sign=sign)
    va = valid_region(raw.shape, q)
    if va[1] - va[0] < 16 and verbose:
        print("  WARNING: azimuth extent shorter than the synthetic aperture; metrics use few lines")
    return slc, {"doppler_centroid": float(fdc), "azimuth_sign": sign, "valid": va, "params": q}


def quicklook(img: np.ndarray, path, look: Tuple[int, int] = (1, 1), percentiles=(2.0, 99.7),
              title: Optional[str] = None) -> None:
    """Save a dB-scaled amplitude image (multi-looked by block averaging of power)."""
    P = np.abs(np.asarray(img)).astype(np.float64) ** 2
    la, lr = look
    if la > 1 or lr > 1:
        h = (P.shape[0] // la) * la
        w = (P.shape[1] // lr) * lr
        P = P[:h, :w].reshape(h // la, la, w // lr, lr).mean(axis=(1, 3))
    dB = 10.0 * np.log10(P + 1e-12)
    lo, hi = np.percentile(dB, percentiles)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8 * P.shape[0] / max(P.shape[1], 1) + 0.5))
        ax.imshow(dB, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        ax.set_xlabel("range sample")
        ax.set_ylabel("azimuth line")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except ImportError:  # pragma: no cover
        from PIL import Image

        u8 = (np.clip((dB - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(u8).save(path)
