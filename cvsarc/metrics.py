"""Quality metrics of the paper (Sec. III-A) and RD-table helpers (Sec. III-B).

All metrics compare the SLC image focused from the original raw data (``ref``)
with the SLC focused from the decompressed raw data (``test``):

* SQNR [dB]            = 10 log10( sum|ref|^2 / sum|ref - test|^2 )
* phase error std [deg] = std( arg(ref * conj(test)) )      (Fig. 5 axis label)
* complex coherence     = |<ref test*>| / sqrt(<|ref|^2><|test|^2>)
  estimated in a sliding boxcar window and averaged over the image
  (``coherence_local``); the single-window estimate over the whole image is
  reported as ``coherence_global``.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["sqnr_db", "phase_error_deg", "coherence", "magnitude_difference", "evaluate_pair",
           "interp_at", "gain_tables", "bitrate_at_quality"]


def sqnr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref = np.asarray(ref)
    test = np.asarray(test)
    num = float(np.sum(np.abs(ref.astype(np.complex128)) ** 2))
    den = float(np.sum(np.abs(ref.astype(np.complex128) - test) ** 2))
    if den == 0.0:
        return float("inf")
    return 10.0 * math.log10(num / den)


def phase_error_deg(ref: np.ndarray, test: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    d = np.angle(np.asarray(ref, dtype=np.complex128) * np.conj(np.asarray(test, dtype=np.complex128)))
    d = np.degrees(d)
    if mask is not None:
        d = d[mask]
    d = d.ravel()
    return {"std": float(np.std(d)), "mean_abs": float(np.mean(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d ** 2)))}


def _boxcar(a: np.ndarray, w: int) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter

        return uniform_filter(a, size=w, mode="reflect")
    except ImportError:  # pragma: no cover
        pad = w // 2
        p = np.pad(a, pad, mode="reflect")
        c = np.cumsum(np.cumsum(p, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        s = c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]
        return s[: a.shape[0], : a.shape[1]] / (w * w)


def coherence(ref: np.ndarray, test: np.ndarray, window: Optional[int] = 5) -> Dict[str, float]:
    r = np.asarray(ref, dtype=np.complex128)
    t = np.asarray(test, dtype=np.complex128)
    cross = r * np.conj(t)
    pr = np.abs(r) ** 2
    pt = np.abs(t) ** 2
    g_global = float(np.abs(cross.sum()) / math.sqrt(max(pr.sum() * pt.sum(), 1e-300)))
    out = {"coherence_global": g_global}
    if window and window > 1:
        num_r = _boxcar(cross.real, window)
        num_i = _boxcar(cross.imag, window)
        den = np.sqrt(_boxcar(pr, window) * _boxcar(pt, window))
        ok = den > 0
        g = np.zeros_like(den)
        g[ok] = np.hypot(num_r[ok], num_i[ok]) / den[ok]
        out["coherence_local"] = float(np.mean(np.clip(g[ok], 0.0, 1.0))) if np.any(ok) else float("nan")
    return out


def magnitude_difference(ref: np.ndarray, test: np.ndarray, normalize: str = "max") -> np.ndarray:
    """Normalised magnitude difference used for the histograms of Fig. 9."""
    a = np.abs(ref)
    b = np.abs(test)
    if normalize == "max":
        n = float(a.max()) if a.size else 1.0
    elif normalize == "mean":
        n = float(a.mean()) if a.size else 1.0
    else:
        n = 1.0
    return (b - a) / max(n, 1e-30)


def evaluate_pair(ref: np.ndarray, test: np.ndarray, coherence_window: Optional[int] = 5) -> Dict[str, float]:
    res = {"sqnr_db": sqnr_db(ref, test)}
    pe = phase_error_deg(ref, test)
    res["phase_std_deg"] = pe["std"]
    res["phase_mean_abs_deg"] = pe["mean_abs"]
    res.update(coherence(ref, test, coherence_window))
    return res


def interp_at(bpps: Sequence[float], values: Sequence[float], targets: Iterable[float],
              extrapolate: bool = False) -> List[float]:
    """Linear interpolation of an RD curve at given bit rates (Tables I-III)."""
    b = np.asarray(bpps, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(b) & np.isfinite(v)
    b, v = b[ok], v[ok]
    if b.size == 0:
        return [float("nan") for _ in targets]
    o = np.argsort(b)
    b, v = b[o], v[o]
    out = []
    for t in targets:
        if b.size == 1:
            out.append(float(v[0]) if abs(t - b[0]) < 1e-9 else float("nan"))
            continue
        if (t < b[0] or t > b[-1]) and not extrapolate:
            out.append(float("nan"))
            continue
        if t < b[0]:
            out.append(float(v[0] + (t - b[0]) * (v[1] - v[0]) / (b[1] - b[0])))
        elif t > b[-1]:
            out.append(float(v[-1] + (t - b[-1]) * (v[-1] - v[-2]) / (b[-1] - b[-2])))
        else:
            out.append(float(np.interp(t, b, v)))
    return out


def gain_tables(curves: Dict[str, Dict[str, Sequence[float]]], metric: str,
                bpps=(2.0, 3.0, 4.0), reference: str = "CAE", higher_is_better: bool = True,
                extrapolate: bool = False) -> List[Dict[str, float]]:
    """Rows ``{bpp, <method>: value, '<ref>-<method> gain': ...}`` (Tables I-III)."""
    vals = {m: interp_at(c["bpp"], c[metric], bpps, extrapolate) for m, c in curves.items()}
    rows = []
    for i, b in enumerate(bpps):
        row = {"bpp": b}
        for m in curves:
            row[m] = vals[m][i]
        if reference in curves:
            for m in curves:
                if m == reference:
                    continue
                g = vals[reference][i] - vals[m][i]
                row[f"{reference}-{m} gain"] = g if higher_is_better else -g
        rows.append(row)
    return rows


def bitrate_at_quality(bpps: Sequence[float], quality: Sequence[float], target: float) -> float:
    """Bit rate needed to reach ``target`` quality (e.g. SQNR) by interpolation."""
    b = np.asarray(bpps, dtype=float)
    q = np.asarray(quality, dtype=float)
    ok = np.isfinite(b) & np.isfinite(q)
    b, q = b[ok], q[ok]
    if b.size < 2:
        return float("nan")
    o = np.argsort(q)
    q, b = q[o], b[o]
    if target < q[0] or target > q[-1]:
        return float("nan")
    return float(np.interp(target, q, b))
