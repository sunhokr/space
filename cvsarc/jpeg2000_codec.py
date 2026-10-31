"""JPEG2000 baseline (paper Sec. II-C, ref. [6]).

As in the paper, the JPEG2000 wavelet coder is applied *separately* to the
real and the imaginary component of the raw-data matrix.  Each component is
mapped to unsigned 16-bit integers and compressed with the irreversible 9/7
wavelet; the rate is controlled with OpenJPEG's compression ratio, refined by a
bisection search so that the measured code-stream size meets the target bit
rate (bits per real sample).  The reported rate is computed from the actual
code-stream sizes of both components.

Backends: Pillow (bundles OpenJPEG on Windows/Linux/macOS wheels; default) or
glymur (needs the OpenJPEG shared library).
"""
from __future__ import annotations

import io
import math
import os
import tempfile
from typing import Dict, Tuple

import numpy as np

from .utils import is_integer_valued

__all__ = ["JPEG2000Codec", "jpeg2000_available"]

_OFFSET = 32768
_PREC = 16


def jpeg2000_available(backend: str = "auto") -> bool:
    if backend in ("auto", "pillow"):
        try:
            from PIL import features

            if features.check("jpg_2000"):
                return True
        except Exception:  # noqa: BLE001
            pass
        if backend == "pillow":
            return False
    try:
        import glymur  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


class JPEG2000Codec:
    def __init__(self, target_bpp: float, backend: str = "auto", irreversible: bool = True,
                 num_resolutions: int = 6, rate_control: str = "bisection", max_iter: int = 12,
                 tol: float = 0.01):
        self.target_bpp = float(target_bpp)
        self.irreversible = bool(irreversible)
        self.num_resolutions = int(num_resolutions)
        self.rate_control = rate_control
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if backend == "auto":
            backend = "pillow" if jpeg2000_available("pillow") else "glymur"
        if not jpeg2000_available(backend):
            raise RuntimeError("no JPEG2000 encoder available (install Pillow with OpenJPEG or glymur)")
        self.backend = backend

    # ------------------------------------------------------------------ backends
    def _encode(self, u16: np.ndarray, ratio: float) -> bytes:
        nres = max(1, min(self.num_resolutions, int(math.log2(max(2, min(u16.shape)))) - 2))
        if self.backend == "pillow":
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(u16).save(buf, format="JPEG2000", quality_mode="rates",
                                      quality_layers=[float(ratio)], irreversible=self.irreversible,
                                      num_resolutions=nres, no_jp2=True)
            return buf.getvalue()
        import glymur

        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "c.j2k")
            glymur.Jp2k(fn, data=u16, cratios=[float(ratio)], irreversible=self.irreversible,
                        numres=nres)
            with open(fn, "rb") as f:
                return f.read()

    def _decode(self, data: bytes) -> np.ndarray:
        if self.backend == "pillow":
            from PIL import Image

            return np.array(Image.open(io.BytesIO(data)))
        import glymur

        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "c.j2k")
            with open(fn, "wb") as f:
                f.write(data)
            return np.asarray(glymur.Jp2k(fn)[:])

    # ------------------------------------------------------------------ rate control
    def _code_component(self, comp: np.ndarray) -> Tuple[bytes, np.ndarray]:
        integer = is_integer_valued(comp.astype(np.complex64))
        amax = float(np.max(np.abs(comp))) if comp.size else 0.0
        if integer and amax < _OFFSET:
            scale = 1.0
        else:
            scale = 2.0 ** math.floor(math.log2((2 ** 14) / max(amax, 1e-12)))
        q = np.clip(np.rint(comp * scale) + _OFFSET, 0, 65535).astype(np.uint16)
        npix = q.size
        target_bytes = self.target_bpp * npix / 8.0
        ratio0 = _PREC / self.target_bpp
        if self.rate_control == "direct":
            best = self._encode(q, ratio0)
        else:
            lo, hi = math.log(1.0), math.log(ratio0 * 64.0)  # log-ratio search interval
            best, best_err = None, float("inf")
            r = math.log(ratio0)
            for _ in range(self.max_iter):
                data = self._encode(q, math.exp(r))
                size = len(data)
                err = abs(size / target_bytes - 1.0)
                if size <= target_bytes * (1.0 + self.tol) and err < best_err:
                    best, best_err = data, err
                if err <= self.tol:
                    break
                if size > target_bytes:
                    lo = r
                else:
                    hi = r
                r = 0.5 * (lo + hi)
            if best is None:
                best = self._encode(q, math.exp(hi))
        dec = self._decode(best).astype(np.float64)
        return best, ((dec - _OFFSET) / scale).astype(np.float32)

    def compress_decompress(self, raw: np.ndarray):
        x = np.asarray(raw, dtype=np.complex64)
        d_re, re = self._code_component(x.real)
        d_im, im = self._code_component(x.imag)
        bits = 8 * (len(d_re) + len(d_im))
        info: Dict = {"bits": bits, "bpp": bits / (2.0 * x.size), "bpp_target": self.target_bpp}
        return (re + 1j * im).astype(np.complex64), info
