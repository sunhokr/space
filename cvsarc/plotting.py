"""Figures equivalent to Figs. 4-9 of the paper and markdown tables (Tables I-III)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

STYLE = {
    "CAE": {"color": "#4472C4", "marker": "o", "label": "CV Autoencoder"},
    "BAQ": {"color": "#ED7D31", "marker": "o", "label": "BAQ"},
    "JPEG2000": {"color": "#FFC000", "marker": "o", "label": "JPEG2000"},
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_rd(curves: Dict[str, Dict[str, Sequence[float]]], metric: str, ylabel: str, path,
            title: Optional[str] = None, extra_points: Optional[List[Dict]] = None,
            yscale: float = 1.0) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for m, c in curves.items():
        b = np.asarray(c["bpp"], dtype=float)
        v = np.asarray(c[metric], dtype=float) * yscale
        o = np.argsort(b)
        st = STYLE.get(m, {"color": None, "marker": "o", "label": m})
        ax.plot(b[o], v[o], marker=st["marker"], color=st["color"], label=st["label"], lw=2)
    for p in extra_points or []:
        ax.plot([p["bpp"]], [p[metric] * yscale], marker="x", ms=12, mew=2,
                color=STYLE.get(p.get("method", "BAQ"), {}).get("color", "k"),
                ls="none", label=p.get("label", "no adaptation"))
    ax.set_xlabel("Bitrate (bpp, bits per real I/Q sample)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _db(img: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(img) + 1e-12)


def plot_slc_panels(images: Dict[str, np.ndarray], path, title: Optional[str] = None,
                    percentiles=(2.0, 99.7)) -> None:
    plt = _plt()
    names = list(images)
    ref = _db(images[names[0]])
    lo, hi = np.percentile(ref, percentiles)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.4))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        ax.imshow(_db(images[n]), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(n, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_histograms(diffs: Dict[str, np.ndarray], path, bins: int = 201, title: Optional[str] = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    allv = np.concatenate([np.ravel(v) for v in diffs.values()])
    lim = float(np.percentile(np.abs(allv), 99.9)) if allv.size else 1.0
    edges = np.linspace(-lim, lim, bins)
    for name, v in diffs.items():
        key = name.split(" ")[0]
        st = STYLE.get(key, {"color": None})
        ax.hist(np.ravel(v), bins=edges, histtype="step", lw=2, color=st["color"], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("normalized magnitude difference (|compressed| - |original|) / max|original|")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.4)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown_table(rows: List[Dict], columns: List[str], fmt: str = "{:.2f}") -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_text(path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
