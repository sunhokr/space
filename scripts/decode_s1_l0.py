#!/usr/bin/env python
"""Decode a Sentinel-1 Level-0 (RAW) product into a raw dataset directory.

Writes ``<out>/raw.npy`` (complex64, azimuth x range), ``<out>/metadata.json``
(focusing parameters, ephemeris) and ``<out>/side_info.npz`` (per-block BRC /
THIDX needed by the adaptation of ref. [20]).

Examples
--------
# list the acquisition chunks of the HH channel of a SAFE product
python scripts/decode_s1_l0.py --input S1A_S3_RAW__0SDH_...SAFE --pol hh --list

# decode 8192 lines of the largest echo chunk
python scripts/decode_s1_l0.py --input S1A_S3_RAW__0SDH_...SAFE --pol hh --out data/chicago_l0 \
       --line-start 2000 --max-lines 8192
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvsarc import s1_tables as T  # noqa: E402
from cvsarc.sentinel1_l0 import decode_l0_file, find_echo_chunks, index_packets  # noqa: E402


def find_measurement_file(inp: Path, pol: str) -> Path:
    if inp.is_file():
        return inp
    cands = [p for p in inp.rglob("*.dat")
             if "annot" not in p.name.lower() and "index" not in p.name.lower()
             and f"-{pol.lower()}-" in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f"no measurement .dat file for polarisation {pol} in {inp}")
    return sorted(cands)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help=".SAFE folder or measurement .dat file")
    ap.add_argument("--pol", default="hh", help="polarisation of the measurement file (hh, hv, vv, vh)")
    ap.add_argument("--out", help="output raw dataset directory")
    ap.add_argument("--list", action="store_true", help="only list the echo acquisition chunks")
    ap.add_argument("--chunk", type=int, help="chunk index (default: largest echo chunk)")
    ap.add_argument("--swath", type=int, help="restrict to a swath number")
    ap.add_argument("--line-start", type=int, default=0)
    ap.add_argument("--max-lines", type=int)
    ap.add_argument("--sample-start", type=int, default=0)
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--backend", choices=["builtin", "sentinel1decoder"], default="builtin")
    a = ap.parse_args()

    path = find_measurement_file(Path(a.input), a.pol)
    print(f"measurement file: {path}")
    if a.list:
        pk = index_packets(path)
        for i, c in enumerate(find_echo_chunks(pk)):
            print(f"chunk {i}: swath {c['swath']} ({T.ECC_NAMES.get(c['ecc'], c['ecc'])}), pol {c['polarisation']}, "
                  f"{c['n_lines']} lines (missing {c['missing_lines']}), {2 * c['num_quads']} samples, "
                  f"{T.BAQ_MODE_NAMES.get(c['baq_mode'], c['baq_mode'])}, "
                  f"t = {c['time_first']:.3f} .. {c['time_last']:.3f} s (GPS)")
        return
    if not a.out:
        ap.error("--out is required unless --list is given")
    meta = decode_l0_file(path, a.out, chunk_index=a.chunk, swath=a.swath, line_start=a.line_start,
                          max_lines=a.max_lines, sample_start=a.sample_start, max_samples=a.max_samples,
                          backend=a.backend)
    print(f"written {a.out}: {meta['n_lines']} lines x {meta['n_samples']} samples, "
          f"PRF {meta['prf']:.2f} Hz, fs {meta['fs'] / 1e6:.3f} MHz, "
          f"chirp {meta.get('chirp_bandwidth', 0) / 1e6:.2f} MHz / {meta['pulse_length'] * 1e6:.2f} us, "
          f"near range {meta['near_range'] / 1e3:.3f} km, Vr {meta.get('effective_velocity')}")


if __name__ == "__main__":
    main()
