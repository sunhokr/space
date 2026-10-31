"""ERS-1/2 SAR raw data reader (CEOS Level-0 "RAW" format).

Used for the generalisation experiment of the paper (Sec. III-C, Fig. 7):
the network trained on Sentinel-1 (HH, SM) is applied without fine-tuning to an
ERS-1 VV scene (La Coruna oil spill, December 1992).  ERS raw data are not
FDBAQ-compressed (5-bit I/Q from the ADC), so no adaptation is needed.

CEOS raw data file layout (ESA ERS SAR.RAW / DAT_01.001):

* first record: file descriptor (its length is read from bytes 8..11),
* every following record: one echo line = ``prefix_bytes`` of line header
  (412 for ERS) + ``n_samples`` complex samples stored as interleaved unsigned
  bytes I, Q (5 significant bits, mean ~ 15.5).

Focusing parameters that are not stored in the data file (PRF, near range,
effective velocity, ...) must be supplied by the user (from the CEOS leader
file / product annotation); see ``configs/ers1_params_template.yaml``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .utils import save_raw_dataset

C_LIGHT = 299792458.0

ERS_DEFAULTS: Dict = {
    "sensor": "ERS-1",
    "carrier_frequency": 5.3e9,
    "wavelength": C_LIGHT / 5.3e9,
    "fs": 18.962468e6,  # range sampling rate [Hz]
    "chirp_rate": 4.18989015e11,  # [Hz/s] (15.55 MHz / 37.12 us)
    "pulse_length": 37.12e-6,  # [s]
    "chirp_start_freq": None,  # centred chirp
    "n_samples": 5616,
    "prefix_bytes": 412,
}

__all__ = ["read_ceos_raw", "ceos_to_dataset", "ERS_DEFAULTS"]


def _be_int32(mm: np.ndarray, pos: int) -> int:
    b = np.asarray(mm[pos:pos + 4]).tobytes()
    return int.from_bytes(b, "big")


def read_ceos_raw(path, n_samples: int = 5616, prefix_bytes: int = 412, start_line: int = 0,
                  n_lines: Optional[int] = None, record_length: Optional[int] = None,
                  first_record_length: Optional[int] = None, dc_offset="auto") -> np.ndarray:
    """Read (a block of) echo lines as complex64 (I - dc) + j (Q - dc)."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    L0 = first_record_length or _be_int32(mm, 8)
    rec = record_length or _be_int32(mm, L0 + 8)
    if rec < prefix_bytes + 2 * n_samples:
        raise ValueError(f"record length {rec} too short for {n_samples} samples + {prefix_bytes} B prefix")
    total = (mm.size - L0) // rec
    n = total - start_line if n_lines is None else min(n_lines, total - start_line)
    if n <= 0:
        raise ValueError("no lines to read")
    base = L0 + start_line * rec
    block = np.asarray(mm[base:base + n * rec]).reshape(n, rec)[:, prefix_bytes:prefix_bytes + 2 * n_samples]
    iq = block.astype(np.float32)
    i, q = iq[:, 0::2], iq[:, 1::2]
    if dc_offset == "auto":
        di, dq = float(i.mean()), float(q.mean())
    else:
        di = dq = float(dc_offset)
    return ((i - di) + 1j * (q - dq)).astype(np.complex64)


def ceos_to_dataset(dat_path, out_dir, params: Dict, start_line: int = 0, n_lines: Optional[int] = None,
                    sample_start: int = 0, max_samples: Optional[int] = None) -> Dict:
    """Convert a CEOS raw file into a raw dataset directory (raw.npy + metadata.json)."""
    p = dict(ERS_DEFAULTS)
    p.update({k: v for k, v in params.items() if v is not None})
    raw = read_ceos_raw(dat_path, n_samples=int(p["n_samples"]), prefix_bytes=int(p["prefix_bytes"]),
                        start_line=start_line, n_lines=n_lines, dc_offset=p.get("dc_offset", "auto"))
    s1 = raw.shape[1] if max_samples is None else min(raw.shape[1], sample_start + max_samples)
    raw = raw[:, sample_start:s1]
    meta = {k: p.get(k) for k in ("sensor", "carrier_frequency", "wavelength", "fs", "chirp_rate",
                                   "pulse_length", "chirp_start_freq", "prf", "near_range",
                                   "effective_velocity", "doppler_centroid", "polarisation")}
    if meta.get("near_range") is not None:
        meta["near_range"] = float(meta["near_range"]) + sample_start * C_LIGHT / (2.0 * float(p["fs"]))
    meta["source_file"] = str(Path(dat_path))
    meta["line_start"] = int(start_line)
    meta["sample_start"] = int(sample_start)
    meta["focusing"] = p.get("focusing", {}) or {}
    missing = [k for k in ("prf", "near_range", "effective_velocity") if meta.get(k) is None]
    if missing:
        print(f"WARNING: metadata lacks {missing}; edit metadata.json before focusing/evaluation")
    save_raw_dataset(out_dir, raw, meta)
    return meta
