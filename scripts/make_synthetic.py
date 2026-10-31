#!/usr/bin/env python
"""Generate synthetic Sentinel-1-like stripmap raw data (no download needed).

For every scene three raw dataset directories are written:
``<name>_uncompressed`` (10-bit ADC ground truth), ``<name>_l0decoded``
(FDBAQ-like compressed + decoded, with BRC/THIDX side information) and
``<name>_adapted`` (after the adaptation of ref. [20]).  ``--write-l0`` also
writes ``<name>.dat`` as genuine Sentinel-1 ISP packets.

Example
-------
python scripts/make_synthetic.py --out data/synthetic --scenes train1:urban train2:mixed train3:rural test:coastal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvsarc.simulation import SimParams, generate_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="+", default=["train1:urban", "train2:mixed", "train3:rural", "test:coastal"],
                    help="name:kind with kind in {mixed, urban, rural, coastal}")
    ap.add_argument("--n-az", type=int, default=2048)
    ap.add_argument("--n-rg", type=int, default=1536)
    ap.add_argument("--pulse-length", type=float, default=6e-6)
    ap.add_argument("--chirp-bandwidth", type=float, default=42e6)
    ap.add_argument("--snr-db", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-l0", action="store_true")
    a = ap.parse_args()
    for i, s in enumerate(a.scenes):
        name, _, kind = s.partition(":")
        sp = SimParams(n_az=a.n_az, n_rg=a.n_rg, pulse_length=a.pulse_length, chirp_bandwidth=a.chirp_bandwidth,
                       snr_db=a.snr_db, scene=kind or "mixed", seed=a.seed + 101 * i)
        paths = generate_dataset(a.out, name, sp, write_l0=a.write_l0, adapt_seed=a.seed + i)
        for k, v in paths.items():
            print(f"  {k:13s} {v}")


if __name__ == "__main__":
    main()
