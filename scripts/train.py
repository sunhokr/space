#!/usr/bin/env python
"""Train the CV autoencoder (paper Fig. 2) with the rate-distortion loss (Eq. 1).

One model is trained per lambda (paper Sec. III-A: "The network shown in Fig. 2
is trained with different lambda values to obtain different compression rates").

Example
-------
python scripts/train.py --train data/chicago_adapted data/houston_adapted data/saopaulo_adapted \
       --lmbda 16 32 64 128 256 512 --out runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvsarc.config import load_config  # noqa: E402
from cvsarc.training import train_many  # noqa: E402
from cvsarc.utils import save_json  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--train", nargs="+", required=True, help="raw dataset directories (training scenes)")
    ap.add_argument("--val", nargs="*", default=None,
                    help="validation dataset directories (default: held-out azimuth part of the training scenes)")
    ap.add_argument("--lmbda", type=float, nargs="+", default=None, help="lambda value(s) (default: config)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--samples-per-epoch", type=int)
    ap.add_argument("--val-samples", type=int)
    ap.add_argument("--patch-size", type=int)
    ap.add_argument("--N", type=int)
    ap.add_argument("--M", type=int)
    ap.add_argument("--gdn-mode", choices=["magnitude", "split"])
    ap.add_argument("--normalization", choices=["patch_rms", "global", "none"])
    ap.add_argument("--device")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-iters", type=int, help="stop each lambda after this many iterations")
    ap.add_argument("--aux-lr", type=float)
    ap.add_argument("--init", choices=["random", "polyphase", "haar"],
                    help="weight initialisation: random (standard) | polyphase / haar (optional exact-transform "
                         "starts, not in the paper)")
    ap.add_argument("--init-from", help="initialise (fine-tune) from an existing checkpoint, e.g. for a new sensor")
    ap.add_argument("--entropy-init-scale", type=float, help="initial width of the factorized densities")
    ap.add_argument("--entropy-lr-mult", type=float, help="learning-rate multiplier of the entropy-model densities")
    ap.add_argument("--warmup-iters", type=int, help="linear learning-rate warm-up iterations")
    ap.add_argument("--warm-start", action="store_true",
                    help="fine-tune every lambda model from the previously trained one")
    ap.add_argument("--train-range", type=float, nargs=2, metavar=("F0", "F1"),
                    help="azimuth fraction of each training scene used for training (default 0 0.9)")
    ap.add_argument("--val-range", type=float, nargs=2, metavar=("F0", "F1"),
                    help="azimuth fraction used for validation when --val is not given (default 0.9 1)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/lambda_X/last.pt")
    return ap.parse_args()


def apply_overrides(cfg, a):
    m, d, t = cfg["model"], cfg["data"], cfg["train"]
    for key, sec in (("N", m), ("M", m), ("gdn_mode", m), ("init", m), ("entropy_init_scale", m), ("init_from", t),
                     ("patch_size", d), ("normalization", d),
                     ("samples_per_epoch", d), ("val_samples", d), ("epochs", t), ("batch_size", t),
                     ("lr", t), ("aux_lr", t), ("entropy_lr_mult", t), ("warmup_iters", t), ("device", t),
                     ("num_workers", t), ("seed", t),
                     ("max_iters", t)):
        v = getattr(a, key)
        if v is not None:
            sec[key] = v
    if a.train_range:
        d["train_azimuth_range"] = list(a.train_range)
    if a.val_range:
        d["val_azimuth_range"] = list(a.val_range)
    if a.lmbda:
        t["lambdas"] = a.lmbda
    return cfg


def main():
    a = parse_args()
    cfg = apply_overrides(load_config(a.config), a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out / "config_used.json")
    summary = train_many(cfg, cfg["train"]["lambdas"], a.train, a.val, out, a.resume, warm_start=a.warm_start)
    save_json(summary, out / "summary.json")
    for s in summary:
        v = s.get("best_val") or {}
        print(f"lambda {s['lmbda']:g}: val bpp {v.get('bpp', float('nan')):.3f}, "
              f"val SQNR {v.get('sqnr_db', float('nan')):.2f} dB -> {s['checkpoint']}")


if __name__ == "__main__":
    main()
