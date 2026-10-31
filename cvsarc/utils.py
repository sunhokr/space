"""I/O helpers and small utilities.

A *raw dataset directory* is the on-disk exchange format used by every script:

    <dir>/raw.npy          complex64 matrix  (azimuth lines x range samples)
    <dir>/metadata.json    sensor / focusing parameters (see focusing.FocusParams)
    <dir>/side_info.npz    optional: per-block BRC / THIDX of the (FD)BAQ decoder

``raw.npy`` is always opened memory-mapped, so whole Sentinel-1 scenes
(tens of GB) can be handled without loading them into RAM.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

RAW_FILE = "raw.npy"
META_FILE = "metadata.json"
SIDE_FILE = "side_info.npz"


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars / arrays and paths."""

    def default(self, o):  # noqa: D102
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (complex, np.complexfloating)):
            return [float(np.real(o)), float(np.imag(o))]
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=NumpyJSONEncoder, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_raw_dataset(dirpath, shape: Tuple[int, int], metadata: Optional[dict] = None,
                       dtype=np.complex64) -> np.memmap:
    """Create ``<dir>/raw.npy`` as a writable memory map and (optionally) metadata."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(d / RAW_FILE, mode="w+", dtype=dtype, shape=tuple(int(s) for s in shape))
    if metadata is not None:
        save_json(metadata, d / META_FILE)
    return arr


def save_raw_dataset(dirpath, raw: np.ndarray, metadata: Optional[dict] = None,
                     side_info: Optional[dict] = None, chunk_lines: int = 4096) -> Path:
    """Write a raw dataset directory (raw is copied in chunks to limit RAM use)."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(d / RAW_FILE, mode="w+", dtype=np.complex64, shape=raw.shape)
    for a0 in range(0, raw.shape[0], chunk_lines):
        out[a0:a0 + chunk_lines] = np.asarray(raw[a0:a0 + chunk_lines], dtype=np.complex64)
    out.flush()
    del out
    meta = dict(metadata or {})
    meta["n_lines"] = int(raw.shape[0])
    meta["n_samples"] = int(raw.shape[1])
    save_json(meta, d / META_FILE)
    if side_info is not None:
        np.savez_compressed(d / SIDE_FILE, **side_info)
    return d


def load_raw_dataset(dirpath, mmap: bool = True):
    """Return ``(raw, metadata, side_info_or_None)`` of a raw dataset directory."""
    d = Path(dirpath)
    raw_path = d / RAW_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found (expected a raw dataset directory)")
    raw = np.load(raw_path, mmap_mode="r" if mmap else None)
    meta = load_json(d / META_FILE) if (d / META_FILE).exists() else {}
    side = None
    if (d / SIDE_FILE).exists():
        with np.load(d / SIDE_FILE) as z:
            side = {k: z[k] for k in z.files}
    return raw, meta, side


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def iter_tiles(shape: Tuple[int, int], tile: int) -> Iterator[Tuple[int, int, int, int]]:
    """Yield (r0, r1, c0, c1) of non-overlapping tiles covering ``shape``.

    Edge tiles are shorter than ``tile``; the caller pads them.
    """
    H, W = shape
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            yield r0, min(r0 + tile, H), c0, min(c0 + tile, W)


def pad_to_multiple(x: np.ndarray, multiple: int, mode: str = "constant") -> np.ndarray:
    """Pad the last two axes of ``x`` up to a multiple of ``multiple``."""
    H, W = x.shape[-2:]
    ph = (-H) % multiple
    pw = (-W) % multiple
    if ph == 0 and pw == 0:
        return x
    pad = [(0, 0)] * (x.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(x, pad, mode=mode)


def is_integer_valued(x: np.ndarray, max_check: int = 1_000_000) -> bool:
    """True if real and imaginary parts of (a sample of) ``x`` are integers."""
    flat = np.asarray(x).reshape(-1)
    if flat.size > max_check:
        idx = np.linspace(0, flat.size - 1, max_check).astype(np.int64)
        flat = flat[idx]
    return bool(np.all(np.round(flat.real) == flat.real) and np.all(np.round(flat.imag) == flat.imag))


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")
