#!/usr/bin/env python
"""Adapt FDBAQ-decoded Sentinel-1 raw data (paper Sec. II-A, ref. [20]).

Example
-------
python scripts/adapt_raw.py --input data/chicago_l0 --out data/chicago_adapted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cvsarc.adaptation import adapt  # noqa: E402
from cvsarc.utils import create_raw_dataset, load_raw_dataset, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="decoded raw dataset directory")
    ap.add_argument("--out", required=True, help="output raw dataset directory")
    ap.add_argument("--mode", choices=["auto", "side_info", "generic", "fixed"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-round", action="store_true", help="do not round to the integer ADC grid")
    ap.add_argument("--clip", type=float, default=None, help="clip I/Q to +-clip (e.g. 511 for 10 bit)")
    ap.add_argument("--step", type=float, default=1.0, help="noise width for --mode fixed")
    a = ap.parse_args()

    raw, meta, side = load_raw_dataset(a.input)
    mode = a.mode
    if mode == "auto":
        mode = "side_info" if side is not None and "brc" in side else "generic"
    print(f"adapting {raw.shape[0]} x {raw.shape[1]} samples, mode = {mode}")
    out = create_raw_dataset(a.out, raw.shape)
    kw = {"seed": a.seed, "clip": a.clip, "out": out}
    if mode == "fixed":
        kw.update(step=a.step, round_output=not a.no_round)
        adapt(raw, None, mode="fixed", **kw)
    else:
        kw.update(round_output=not a.no_round, verbose=True)
        adapt(raw, side, mode=mode, **kw)
    out.flush()
    meta = dict(meta)
    meta["adaptation"] = {"mode": mode, "seed": a.seed, "round": not a.no_round, "clip": a.clip,
                          "source": str(a.input), "reference": "Asiyabi et al., IGARSS 2024 (paper ref. [20])"}
    save_json(meta, Path(a.out) / "metadata.json")
    x = np.asarray(raw[: min(raw.shape[0], 512)])
    y = np.asarray(out[: min(raw.shape[0], 512)])
    print(f"distinct |I| values in the first lines: decoded {np.unique(np.abs(x.real)).size}, "
          f"adapted {np.unique(np.abs(y.real)).size}")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
