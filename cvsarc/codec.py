"""Tiled compression of complete raw-data matrices with a trained CV model.

The network is trained on 256 x 256 patches (paper Sec. II-D-2); a full raw
data matrix is therefore cut into tiles (default 256 x 256), every tile is
normalised, compressed into its own arithmetic-coded string and written to a
single container.  The reported bit rate is computed from the *complete
container size* (all headers and side information included).

Normalisation (not specified in the paper, documented assumption):

* ``patch_rms`` (default): every tile is divided by its per-component RMS
  ``s = sqrt(mean(|x|^2) / 2)``; ``s`` is stored as float32 side information
  (32 bits per 65536 complex samples = 0.00024 bit per real sample).  This
  makes the model independent of receiver gain / sensor scaling (e.g. 10-bit
  Sentinel-1 vs 5-bit ERS-1 data, paper Sec. III-C).
* ``global``: a fixed scale stored in the model checkpoint.
* ``none``: no scaling.

Container layout::

    b"CVSAR\\x01" | uint32 header_len | JSON header | per tile: float32 scale, uint32 n, n bytes
"""
from __future__ import annotations

import json
import struct
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .model import CVSARCompressor
from .utils import iter_tiles

MAGIC = b"CVSAR\x01"

__all__ = ["CAECodec", "tile_scale", "MAGIC"]


def tile_scale(x: np.ndarray, normalization: str, global_scale: float = 1.0) -> float:
    if normalization == "patch_rms":
        p = float(np.mean(x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2) / 2.0)
        s = float(np.sqrt(p))
        return s if s > 0 and np.isfinite(s) else 1.0
    if normalization == "global":
        return float(global_scale)
    if normalization == "none":
        return 1.0
    raise ValueError(f"unknown normalization {normalization!r}")


class CAECodec:
    """Compress / decompress complex raw data matrices with a ``CVSARCompressor``."""

    def __init__(self, model: CVSARCompressor, tile: int = 256, normalization: str = "patch_rms",
                 global_scale: float = 1.0, device: str = "cpu", batch_size: int = 8):
        f = model.downsampling_factor
        if tile % f:
            raise ValueError(f"tile must be a multiple of {f}")
        self.model = model.to(device).eval()
        self.model.update(force=False)
        self.tile = int(tile)
        self.normalization = normalization
        self.global_scale = float(global_scale)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.f = f

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu", tile: Optional[int] = None,
                        batch_size: int = 8) -> "CAECodec":
        model, ckpt = CVSARCompressor.from_checkpoint(path, map_location="cpu")
        norm = ckpt.get("normalization", {"mode": "patch_rms", "global_scale": 1.0})
        t = tile or int(ckpt.get("train_config", {}).get("patch_size", 256))
        codec = cls(model, tile=t, normalization=norm.get("mode", "patch_rms"),
                    global_scale=norm.get("global_scale", 1.0), device=device, batch_size=batch_size)
        codec.checkpoint_info = {"lmbda": ckpt.get("lmbda"), "path": str(path)}
        return codec

    # ------------------------------------------------------------------ helpers
    def _padded_shape(self, h: int, w: int) -> Tuple[int, int]:
        f = self.f
        return h + (-h) % f, w + (-w) % f

    def _groups(self, shape):
        groups = defaultdict(list)
        for t in iter_tiles(shape, self.tile):
            r0, r1, c0, c1 = t
            groups[self._padded_shape(r1 - r0, c1 - c0)].append(t)
        return groups

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def compress(self, raw: np.ndarray, estimate_rate: bool = False) -> Tuple[bytes, Dict]:
        H, W = raw.shape
        eb = self.model.entropy_bottleneck
        records: Dict[Tuple[int, int], Tuple[float, bytes]] = {}
        est_bits = 0.0
        for (ph, pw), tiles in self._groups((H, W)).items():
            for b0 in range(0, len(tiles), self.batch_size):
                batch = tiles[b0:b0 + self.batch_size]
                arr = np.zeros((len(batch), 1, ph, pw), dtype=np.complex64)
                scales = []
                for i, (r0, r1, c0, c1) in enumerate(batch):
                    t = np.asarray(raw[r0:r1, c0:c1], dtype=np.complex64)
                    s = tile_scale(t, self.normalization, self.global_scale)
                    scales.append(s)
                    arr[i, 0, : r1 - r0, : c1 - c0] = t / s
                x = torch.from_numpy(arr).to(self.device)
                y = self.model._to_real(self.model.g_a(x))
                if estimate_rate:
                    _, lik = eb(y, training=False)
                    est_bits += float(torch.sum(-torch.log2(lik)).item())
                strings = eb.compress(y)
                for (r0, r1, c0, c1), s, st in zip(batch, scales, strings):
                    records[(r0, c0)] = (s, st)
        header = {"H": int(H), "W": int(W), "tile": self.tile, "f": self.f,
                  "normalization": self.normalization, "global_scale": self.global_scale,
                  "M2": int(eb.channels)}
        hjson = json.dumps(header).encode("utf-8")
        parts = [MAGIC, struct.pack("<I", len(hjson)), hjson]
        for (r0, r1, c0, c1) in iter_tiles((H, W), self.tile):  # canonical order
            s, st = records[(r0, c0)]
            parts.append(struct.pack("<fI", s, len(st)))
            parts.append(st)
        blob = b"".join(parts)
        info = {"bits": 8 * len(blob), "bpp": 8.0 * len(blob) / (2.0 * H * W),
                "bpp_est": est_bits / (2.0 * H * W) if estimate_rate else None,
                "n_tiles": len(records)}
        return blob, info

    @torch.no_grad()
    def decompress(self, blob: bytes) -> np.ndarray:
        if not blob.startswith(MAGIC):
            raise ValueError("not a CVSAR container")
        pos = len(MAGIC)
        (hlen,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        header = json.loads(blob[pos:pos + hlen].decode("utf-8"))
        pos += hlen
        H, W, tile = header["H"], header["W"], header["tile"]
        if tile != self.tile:
            self.tile = tile
        records = {}
        for (r0, r1, c0, c1) in iter_tiles((H, W), tile):
            s, n = struct.unpack_from("<fI", blob, pos)
            pos += 8
            records[(r0, c0)] = (s, blob[pos:pos + n])
            pos += n
        out = np.zeros((H, W), dtype=np.complex64)
        f = self.f
        for (ph, pw), tiles in self._groups((H, W)).items():
            for b0 in range(0, len(tiles), self.batch_size):
                batch = tiles[b0:b0 + self.batch_size]
                strings = [records[(t[0], t[2])][1] for t in batch]
                x_hat = self.model.decompress(strings, (ph // f, pw // f))["x_hat"]
                x_hat = x_hat[:, 0].cpu().numpy()
                for i, (r0, r1, c0, c1) in enumerate(batch):
                    s = records[(r0, c0)][0]
                    out[r0:r1, c0:c1] = x_hat[i, : r1 - r0, : c1 - c0] * s
        return out

    def compress_decompress(self, raw: np.ndarray, estimate_rate: bool = True):
        blob, info = self.compress(raw, estimate_rate=estimate_rate)
        rec = self.decompress(blob)
        return rec, info
