"""PyTorch datasets of complex raw-data patches (paper: 256 x 256 input patches).

Raw dataset directories are opened lazily (memory-mapped) inside each
DataLoader worker, so huge Sentinel-1 scenes never get copied into RAM.
A fraction of the azimuth extent of every scene can be reserved for
validation (``azimuth_range``), e.g. train on [0, 0.9) and validate on [0.9, 1).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .codec import tile_scale
from .utils import RAW_FILE

__all__ = ["RawPatchDataset"]


class RawPatchDataset(Dataset):
    def __init__(self, dirs: Sequence, patch_size: int = 256, num_samples: int = 10000,
                 normalization: str = "patch_rms", global_scale: float = 1.0,
                 azimuth_range: Tuple[float, float] = (0.0, 1.0), deterministic: bool = False,
                 seed: int = 0, min_rms: float = 1e-3, max_tries: int = 20):
        self.files: List[Path] = [Path(d) / RAW_FILE for d in dirs]
        for f in self.files:
            if not f.exists():
                raise FileNotFoundError(f)
        self.P = int(patch_size)
        self.num_samples = int(num_samples)
        self.normalization = normalization
        self.global_scale = float(global_scale)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.min_rms = float(min_rms)
        self.max_tries = int(max_tries)
        self.shapes = []
        self.rows = []
        for f in self.files:
            arr = np.load(f, mmap_mode="r")
            H, W = arr.shape
            r0 = int(np.floor(azimuth_range[0] * H))
            r1 = int(np.floor(azimuth_range[1] * H))
            if r1 - r0 < self.P or W < self.P:
                raise ValueError(f"{f}: region {r1 - r0}x{W} smaller than patch {self.P} "
                                 f"(adjust train/val_azimuth_range or patch_size)")
            self.shapes.append((H, W))
            self.rows.append((r0, r1))
            del arr
        area = np.array([(r1 - r0) * W for (r0, r1), (_, W) in zip(self.rows, self.shapes)], dtype=float)
        self.weights = area / area.sum()
        self._arrays = None

    def __len__(self) -> int:
        return self.num_samples

    def _open(self):
        if self._arrays is None:
            self._arrays = [np.load(f, mmap_mode="r") for f in self.files]
        return self._arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None  # never pickle memory maps into workers
        return state

    def __getitem__(self, idx: int) -> torch.Tensor:
        arrays = self._open()
        rng = np.random.default_rng((self.seed, int(idx))) if self.deterministic else np.random.default_rng()
        P = self.P
        x = None
        for _ in range(self.max_tries):
            k = int(rng.choice(len(arrays), p=self.weights))
            r0, r1 = self.rows[k]
            W = self.shapes[k][1]
            a = int(rng.integers(r0, r1 - P + 1))
            c = int(rng.integers(0, W - P + 1))
            x = np.asarray(arrays[k][a:a + P, c:c + P], dtype=np.complex64)
            rms = float(np.sqrt(np.mean(np.abs(x) ** 2) / 2.0))
            if rms > self.min_rms:
                break
        s = tile_scale(x, self.normalization, self.global_scale)
        return torch.from_numpy((x / s).astype(np.complex64)[None])


def estimate_global_scale(dirs: Sequence, n_patches: int = 64, patch_size: int = 256, seed: int = 0) -> float:
    """Per-component RMS over random patches (for normalization='global')."""
    ds = RawPatchDataset(dirs, patch_size, n_patches, normalization="none", deterministic=True, seed=seed)
    p = [float(np.mean(np.abs(ds[i].numpy()) ** 2) / 2.0) for i in range(len(ds))]
    return float(np.sqrt(np.mean(p)))
