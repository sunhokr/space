#!/usr/bin/env python
"""Run the experiment of the paper (Fig. 3) on a test scene.

CAE (trained checkpoints), BAQ (2-5 bit) and JPEG2000 (Re/Im separately) are
applied to the adapted raw data; original and decompressed raw data are
focused with the same Range-Doppler parameters; SQNR, phase-error std and
complex coherence are measured on the SLC images.  Produces Figs. 4-6, 8, 9
and Tables I-III equivalents in --out.

Example
-------
python scripts/evaluate.py --test data/capeverde_adapted --cae "runs/lambda_*/best.pt" \
       --nonadapted data/capeverde_l0 --out results/capeverde --device cuda
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvsarc.config import load_config, resolve_device  # noqa: E402
from cvsarc.evaluation import run_evaluation  # noqa: E402


def expand(patterns):
    out = []
    for p in patterns or []:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else [p])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--test", required=True, help="raw dataset directory of the test scene (adapted)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cae", nargs="*", default=[], help="CAE checkpoints (glob patterns allowed)")
    ap.add_argument("--baq-bits", type=int, nargs="*")
    ap.add_argument("--jp2-bpp", type=float, nargs="*")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("AZ0", "AZ1", "RG0", "RG1"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int, help="CAE tile size (default: training patch size)")
    ap.add_argument("--coherence-window", type=int)
    ap.add_argument("--nonadapted", help="non-adapted (FDBAQ-decoded) version of the test scene: adds the "
                                         "'BAQ without adaptation' points (orange x in Figs. 4-6)")
    ap.add_argument("--nonadapted-baq-bits", type=int, nargs="*", default=[3])
    ap.add_argument("--azimuth-sign", type=int, choices=[-1, 0, 1], help="0 = automatic")
    ap.add_argument("--doppler-centroid", type=float)
    ap.add_argument("--azimuth-bandwidth-fraction", type=float)
    ap.add_argument("--jp2-backend", default="auto", choices=["auto", "pillow", "glymur"])
    a = ap.parse_args()

    cfg = load_config(a.config)
    ev = cfg["eval"]
    focus_over = {}  # explicit command-line values win over metadata.json and the config file
    if a.azimuth_sign is not None:
        focus_over["azimuth_sign"] = a.azimuth_sign
    if a.doppler_centroid is not None:
        focus_over["doppler_centroid"] = a.doppler_centroid
    if a.azimuth_bandwidth_fraction is not None:
        focus_over["azimuth_bandwidth_fraction"] = a.azimuth_bandwidth_fraction
    ckpts = expand(a.cae)
    run_evaluation(
        a.test, a.out, cae_checkpoints=ckpts,
        baq_bits=a.baq_bits if a.baq_bits is not None else ev["baq_bits"],
        jp2_bpps=a.jp2_bpp if a.jp2_bpp is not None else ev["jp2_bpp"],
        device=resolve_device(a.device), crop=a.crop, focus_overrides=focus_over,
        focus_defaults=cfg.get("focusing", {}),
        coherence_window=a.coherence_window or ev["coherence_window"], baq_block=tuple(ev["baq_block"]),
        jp2_backend=a.jp2_backend, tile=a.tile, nonadapted_dir=a.nonadapted,
        nonadapted_baq_bits=a.nonadapted_baq_bits, table_bpps=ev["table_bpps"], target_sqnrs=ev["target_sqnrs"])


if __name__ == "__main__":
    main()
