"""Adaptation of decoded (FD)BAQ Sentinel-1 raw data (paper Sec. II-A, ref. [20]).

Sentinel-1 L0 echoes were compressed on board with FDBAQ, so the decoded
samples only take a few reconstruction values per block and do not have the
statistics of the uncompressed, uniformly quantized raw data on board.
Following ref. [20] (Asiyabi et al., IGARSS 2024) "a random uniform noise with
a suitable amplitude is added to the decoded SAR raw data to fill in the gaps
between the quantization levels and represent the data in quasi-uniformly
quantized samples".

Implementation (``mode="side_info"``, exact):
  for every I/Q sample the decision cell of its reconstruction level is
  rebuilt from the block's (BRC, THIDX) (cell borders = mid-points between
  neighbouring levels, the outermost cell is symmetric around its level) and the
  magnitude is redrawn uniformly inside that cell (sign kept).  The result is
  then rounded to the integer ADC grid (quasi-uniform quantization) and
  optionally clipped to the 10-bit range.

``mode="generic"`` does the same without side information: the levels of each
128-quad block are taken as the distinct magnitudes present in the block.
``mode="fixed"`` adds U(-step/2, step/2) to I and Q.

The paper notes that *without* this adaptation BAQ at 3 bit behaves
anomalously well (Figs. 4-6, orange x) because its levels coincide with the
FDBAQ lattice; the evaluation scripts can be run on adapted and non-adapted
data to reproduce this effect.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import s1_tables as T

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

__all__ = ["level_cells", "adapt_with_side_info", "adapt_generic", "adapt_fixed", "adapt"]


def level_cells(levels: np.ndarray):
    """Decision cells [lo, hi) of sorted unsigned reconstruction levels."""
    L = np.asarray(levels, dtype=np.float64)
    n = L.size
    lo = np.zeros(n)
    hi = np.zeros(n)
    mids = 0.5 * (L[:-1] + L[1:])
    lo[1:] = mids
    hi[:-1] = mids
    hi[-1] = L[-1] + (L[-1] - lo[-1]) if n > 1 else max(L[-1] * 2.0, 0.5)
    return lo, hi


def _cell_tables(nlev_list, level_fn, n_tables: int, n_thidx: int = 256, width: int = 16):
    """Tables MID (thresholds), LO, HI of shape (n_tables, 256, width)."""
    mid = np.full((n_tables, n_thidx, width), np.inf, dtype=np.float64)
    lo = np.zeros((n_tables, n_thidx, width), dtype=np.float64)
    hi = np.zeros((n_tables, n_thidx, width), dtype=np.float64)
    for b in range(n_tables):
        n = nlev_list[b]
        for t in range(n_thidx):
            lev = np.array([level_fn(b, t, m) for m in range(n)])
            order = np.argsort(lev, kind="stable")
            lev_s = lev[order]
            l_, h_ = level_cells(lev_s)
            mid[b, t, : n - 1] = 0.5 * (lev_s[:-1] + lev_s[1:])
            lo[b, t, :n] = l_
            hi[b, t, :n] = h_
    return mid, lo, hi


_FDBAQ_CELLS = None
_BAQ_CELLS: Dict[int, tuple] = {}


def _fdbaq_cells():
    global _FDBAQ_CELLS
    if _FDBAQ_CELLS is None:
        _FDBAQ_CELLS = _cell_tables(T.FDBAQ_NUM_LEVELS, T.fdbaq_level, 5)
    return _FDBAQ_CELLS


def _baq_cells(nbits: int):
    if nbits not in _BAQ_CELLS:
        n = 1 << (nbits - 1)
        _BAQ_CELLS[nbits] = _cell_tables([n], lambda b, t, m: T.baq_level(nbits, t, m), 1)
    return _BAQ_CELLS[nbits]


def _dither_component(v: np.ndarray, tb: np.ndarray, tt: np.ndarray, cells, rng) -> np.ndarray:
    """Redraw |v| uniformly inside its decision cell; tb/tt = table / thidx per sample."""
    mid, lo, hi = cells
    a = np.abs(v).astype(np.float64)
    m = np.zeros(a.shape, dtype=np.int64)
    for k in range(mid.shape[2]):
        thr = mid[tb, tt, k]
        if not np.any(np.isfinite(thr)):
            break
        m += (a > thr)
    lo_v = lo[tb, tt, m]
    hi_v = hi[tb, tt, m]
    mag = lo_v + (hi_v - lo_v) * rng.random(a.shape)
    sgn = np.where(np.signbit(v), -1.0, 1.0)
    return sgn * mag


def adapt_with_side_info(raw: np.ndarray, side: Dict[str, np.ndarray], out: Optional[np.ndarray] = None,
                         seed: int = 0, round_output: bool = True, clip: Optional[float] = None,
                         chunk_lines: int = 256, verbose: bool = False) -> np.ndarray:
    """Exact adaptation using the per-block BRC/THIDX stored by the decoder."""
    rng = np.random.default_rng(seed)
    n_lines, n_samples = raw.shape
    if out is None:
        out = np.zeros((n_lines, n_samples), dtype=np.complex64)
    brc_all, thidx_all = side["brc"], side["thidx"]
    baq_all = side["baq_mode"].astype(np.int64)
    valid_all = side.get("valid_line", np.ones(n_lines, dtype=bool))
    s0 = int(side.get("sample_start", 0))
    # complex sample j (absolute index s0 + j) belongs to quad (s0+j)//2 -> block quad//128
    blk = ((np.arange(n_samples) + s0) // 2) // T.BAQ_BLOCK_QUADS
    for l0 in range(0, n_lines, chunk_lines):
        l1 = min(n_lines, l0 + chunk_lines)
        x = np.asarray(raw[l0:l1], dtype=np.complex64)
        y = x.copy()
        modes = baq_all[l0:l1]
        for mode in np.unique(modes):
            rows = np.flatnonzero((modes == mode) & valid_all[l0:l1])
            if rows.size == 0 or mode == 0:  # bypass data are already uniformly quantized
                continue
            tt = thidx_all[l0:l1][rows][:, blk].astype(np.int64)
            if mode in T.FDBAQ_MODES:
                cells = _fdbaq_cells()
                tb = brc_all[l0:l1][rows][:, blk].astype(np.int64)
            elif mode in T.BAQ_MODES:
                cells = _baq_cells(int(mode))
                tb = np.zeros_like(tt)
            else:
                raise ValueError(f"unsupported BAQ mode {mode}")
            xr = x[rows]
            re = _dither_component(xr.real, tb, tt, cells, rng)
            im = _dither_component(xr.imag, tb, tt, cells, rng)
            y[rows] = (re + 1j * im).astype(np.complex64)
        y = _finish(y, round_output, clip)
        out[l0:l1] = y
        if verbose:
            print(f"  adapted lines {l1}/{n_lines}", flush=True)
    return out


@njit(cache=True)
def _generic_block(mags, u):
    """Redraw magnitudes inside cells formed by the distinct values of the block."""
    n = mags.shape[0]
    lev = np.unique(mags)
    k = lev.shape[0]
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = np.searchsorted(lev, mags[i])
        if k == 1:
            lo = 0.0
            hi = max(2.0 * lev[0], 0.5)
        else:
            lo = 0.0 if j == 0 else 0.5 * (lev[j - 1] + lev[j])
            if j == k - 1:
                hi = lev[j] + (lev[j] - lo)
            else:
                hi = 0.5 * (lev[j] + lev[j + 1])
        res[i] = lo + (hi - lo) * u[i]
    return res


def adapt_generic(raw: np.ndarray, out: Optional[np.ndarray] = None, seed: int = 0,
                  round_output: bool = True, clip: Optional[float] = None,
                  block_samples: int = 2 * T.BAQ_BLOCK_QUADS, chunk_lines: int = 256,
                  verbose: bool = False) -> np.ndarray:
    """Adaptation without side information (levels inferred per block)."""
    rng = np.random.default_rng(seed)
    n_lines, n_samples = raw.shape
    if out is None:
        out = np.zeros((n_lines, n_samples), dtype=np.complex64)
    for l0 in range(0, n_lines, chunk_lines):
        l1 = min(n_lines, l0 + chunk_lines)
        x = np.asarray(raw[l0:l1], dtype=np.complex64)
        y = np.empty_like(x)
        for r in range(x.shape[0]):
            for c0 in range(0, n_samples, block_samples):
                c1 = min(n_samples, c0 + block_samples)
                seg = x[r, c0:c1]
                comp = np.concatenate([seg.real, seg.imag]).astype(np.float64)
                mags = np.abs(comp)
                newm = _generic_block(mags, rng.random(mags.size))
                vals = np.where(np.signbit(comp), -newm, newm)
                h = c1 - c0
                y[r, c0:c1] = (vals[:h] + 1j * vals[h:]).astype(np.complex64)
        out[l0:l1] = _finish(y, round_output, clip)
        if verbose:
            print(f"  adapted lines {l1}/{n_lines}", flush=True)
    return out


def adapt_fixed(raw: np.ndarray, step: float = 1.0, out: Optional[np.ndarray] = None, seed: int = 0,
                round_output: bool = False, clip: Optional[float] = None,
                chunk_lines: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_lines, n_samples = raw.shape
    if out is None:
        out = np.zeros((n_lines, n_samples), dtype=np.complex64)
    for l0 in range(0, n_lines, chunk_lines):
        l1 = min(n_lines, l0 + chunk_lines)
        x = np.asarray(raw[l0:l1], dtype=np.complex64)
        d = (rng.random(x.shape) - 0.5) * step + 1j * (rng.random(x.shape) - 0.5) * step
        out[l0:l1] = _finish((x + d).astype(np.complex64), round_output, clip)
    return out


def _finish(y: np.ndarray, round_output: bool, clip: Optional[float]) -> np.ndarray:
    if round_output:
        y = (np.rint(y.real) + 1j * np.rint(y.imag)).astype(np.complex64)
    if clip is not None:
        y = (np.clip(y.real, -clip, clip) + 1j * np.clip(y.imag, -clip, clip)).astype(np.complex64)
    return y


def adapt(raw: np.ndarray, side: Optional[Dict[str, np.ndarray]] = None, mode: str = "auto",
          **kwargs) -> np.ndarray:
    """Dispatch: ``auto`` uses side information when available, else ``generic``."""
    if mode == "auto":
        mode = "side_info" if side is not None and "brc" in side else "generic"
    if mode == "side_info":
        if side is None:
            raise ValueError("side information required")
        return adapt_with_side_info(raw, side, **kwargs)
    if mode == "generic":
        return adapt_generic(raw, **kwargs)
    if mode == "fixed":
        return adapt_fixed(raw, **kwargs)
    raise ValueError(f"unknown adaptation mode {mode!r}")
