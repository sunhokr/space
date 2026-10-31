#!/usr/bin/env python
"""Focus a raw dataset with the Range-Doppler algorithm and save a quick-look / SLC.

Example
-------
python scripts/focus.py --input data/capeverde_adapted --out results/slc_capeverde --look 4 1 --save-slc
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cvsarc.focusing import C_LIGHT, FocusParams, focus, quicklook  # noqa: E402
from cvsarc.utils import load_raw_dataset, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("AZ0", "AZ1", "RG0", "RG1"))
    ap.add_argument("--look", type=int, nargs=2, default=[1, 1], help="multi-look factors (az, rg) for the PNG")
    ap.add_argument("--save-slc", action="store_true")
    ap.add_argument("--azimuth-sign", type=int, choices=[-1, 0, 1])
    ap.add_argument("--doppler-centroid", type=float)
    ap.add_argument("--velocity", type=float, help="override effective velocity [m/s]")
    ap.add_argument("--flip-chirp", action="store_true", help="invert the chirp-rate sign")
    a = ap.parse_args()
    raw, meta, _ = load_raw_dataset(a.input)
    meta = dict(meta)
    if a.crop:
        a0, a1, r0, r1 = a.crop
        raw = raw[a0:a1, r0:r1]
        meta["near_range"] = float(meta["near_range"]) + r0 * C_LIGHT / (2.0 * float(meta["fs"]))
    ov = {"azimuth_sign": a.azimuth_sign, "doppler_centroid": a.doppler_centroid, "velocity": a.velocity}
    if a.velocity is not None:
        ov.update(velocity_near=None, velocity_far=None)
        meta.pop("effective_velocity_near", None)
        meta.pop("effective_velocity_far", None)
    p = FocusParams.from_metadata(meta, **ov)
    if a.flip_chirp:
        p.chirp_rate = -p.chirp_rate
        if p.chirp_start_freq is not None:
            p.chirp_start_freq = -p.chirp_start_freq
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    slc, info = focus(np.asarray(raw, dtype=np.complex64), p, verbose=True)
    print(f"focused {raw.shape} in {time.time() - t0:.1f} s; valid region {info['valid']}")
    a0, a1, r0, r1 = info["valid"]
    quicklook(slc[a0:a1, r0:r1], out / "slc_quicklook.png", look=tuple(a.look), title=Path(a.input).name)
    save_json({"focus": info["params"].to_dict(), "valid": info["valid"]}, out / "focus_params.json")
    if a.save_slc:
        np.save(out / "slc.npy", slc)
    print(f"written {out}")


if __name__ == "__main__":
    main()
