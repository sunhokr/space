#!/usr/bin/env python
"""End-to-end demonstration of the complete pipeline on synthetic data.

    synthetic stripmap scenes -> FDBAQ-like on-board compression -> Sentinel-1 ISP packets (.dat)
    -> L0 decoding -> adaptation [20] -> training of the CV autoencoder (several lambdas)
    -> CAE / BAQ / JPEG2000 compression -> RDA focusing -> SQNR / phase error / coherence
    -> Figs. 4-6, 8, 9 and Tables I-III equivalents

The ``quick`` preset uses a reduced network (N=64, M=320, 64x64 patches, Haar start)
and a few hundred iterations so that it finishes on a CPU in minutes; it verifies the
pipeline, it does not reproduce the paper's numbers (that needs the real
Sentinel-1 data and the full N=192 / M=320 model trained to convergence).

Example
-------
python scripts/run_demo.py --out demo_output
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cvsarc.adaptation import adapt_with_side_info  # noqa: E402
from cvsarc.config import load_config, resolve_device  # noqa: E402
from cvsarc.evaluation import run_evaluation  # noqa: E402
from cvsarc.metrics import sqnr_db  # noqa: E402
from cvsarc.sentinel1_l0 import decode_l0_file  # noqa: E402
from cvsarc.simulation import SimParams, generate_dataset  # noqa: E402
from cvsarc.training import train_one  # noqa: E402
from cvsarc.utils import create_raw_dataset, load_raw_dataset, save_json  # noqa: E402

# quick: reduced width N=64 (M=320 as in the paper: the latent must not be smaller than the
# almost-white input, 2*320 real latents per 16x16 block of 512 real samples), 64x64 patches and
# the optional Haar wavelet-packet start (exact transform, not in the paper) so that a CPU run
# converges in minutes.
PRESETS = {
    "quick": {"train_az": 1024, "train_rg": 1024, "test_az": 2048, "test_rg": 1024, "N": 64, "M": 320,
              "patch": 64, "batch": 8, "iters": 600, "lambdas": [16, 64, 256], "samples": 800,
              "init": "haar", "eb_mult": 50.0, "lr": 1e-4},
    "standard": {"train_az": 2048, "train_rg": 1536, "test_az": 3072, "test_rg": 1536, "N": 192, "M": 320,
                 "patch": 256, "batch": 8, "iters": 20000, "lambdas": [16, 32, 64, 128, 256, 512],
                 "samples": 4000, "init": "random", "eb_mult": 1.0, "lr": 1e-4},
}


def histogram_tvd(a: np.ndarray, b: np.ndarray, lim: int = 511) -> float:
    """Total-variation distance between the integer histograms of the I/Q samples."""
    def hist(x):
        v = np.concatenate([x.real.ravel(), x.imag.ravel()])
        h = np.bincount(np.clip(np.rint(v).astype(np.int64) + lim, 0, 2 * lim), minlength=2 * lim + 1)
        return h / h.sum()

    return float(0.5 * np.abs(hist(a) - hist(b)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--lambdas", type=float, nargs="+")
    ap.add_argument("--iters", type=int, help="training iterations per lambda")
    ap.add_argument("--init", choices=["random", "polyphase", "haar"], help="override the preset initialisation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-data", action="store_true", help="reuse existing synthetic data")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing checkpoints")
    a = ap.parse_args()
    P = dict(PRESETS[a.preset])
    if a.lambdas:
        P["lambdas"] = a.lambdas
    if a.iters:
        P["iters"] = a.iters
    if a.init:
        P["init"] = a.init
    out = Path(a.out)
    data = out / "data"
    device = resolve_device(a.device)
    t_start = time.time()

    # ------------------------------------------------------------------ 1. data
    kinds = ["urban", "mixed", "rural"]
    train_dirs = [str(data / f"train{i + 1}_{k}_adapted") for i, k in enumerate(kinds)]
    if not a.skip_data:
        print("== 1. synthetic data (training scenes: FDBAQ-like L0 emulation + adaptation)")
        for i, k in enumerate(kinds):
            sp = SimParams(n_az=P["train_az"], n_rg=P["train_rg"], scene=k, seed=11 + i)
            generate_dataset(data, f"train{i + 1}_{k}", sp)
        print("== 1b. test scene written as genuine Sentinel-1 ISP packets, then decoded + adapted")
        sp = SimParams(n_az=P["test_az"], n_rg=P["test_rg"], scene="coastal", seed=99)
        paths = generate_dataset(data, "test_coastal", sp, write_l0=True)
        decode_l0_file(paths["l0_file"], data / "test_l0decoded_from_dat", verbose=True)
        raw, meta, side = load_raw_dataset(data / "test_l0decoded_from_dat")
        direct, _, _ = load_raw_dataset(paths["l0decoded"])
        print(f"   L0 decoder output identical to the emulated FDBAQ reconstruction: "
              f"{np.array_equal(np.asarray(raw), np.asarray(direct))}")
        ada = create_raw_dataset(data / "test_adapted", raw.shape)
        adapt_with_side_info(raw, side, out=ada, seed=0)
        ada.flush()
        meta = dict(meta)
        meta["adaptation"] = {"mode": "side_info", "seed": 0, "round": True}
        save_json(meta, data / "test_adapted" / "metadata.json")
        truth, _, _ = load_raw_dataset(paths["uncompressed"])
        print(f"   raw SQNR vs. uncompressed 10-bit ground truth: FDBAQ-decoded "
              f"{sqnr_db(np.asarray(truth), np.asarray(raw)):.2f} dB, adapted "
              f"{sqnr_db(np.asarray(truth), np.asarray(ada)):.2f} dB")
        tv_dec = histogram_tvd(np.asarray(truth), np.asarray(raw))
        tv_ada = histogram_tvd(np.asarray(truth), np.asarray(ada))
        print(f"   I/Q histogram distance to the uncompressed data (total variation, 0 = identical): "
              f"FDBAQ-decoded {tv_dec:.3f}, adapted {tv_ada:.3f}")

    # ------------------------------------------------------------------ 2. training
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["model"].update({"N": P["N"], "M": P["M"], "init": P["init"]})
    cfg["data"].update({"patch_size": P["patch"], "samples_per_epoch": P["samples"], "val_samples": 64,
                        "train_azimuth_range": [0.0, 0.8], "val_azimuth_range": [0.8, 1.0]})
    cfg["train"].update({"batch_size": P["batch"], "max_iters": P["iters"], "epochs": 10 ** 6,
                         "num_workers": 0, "device": device, "lr": P["lr"], "aux_lr": 1e-2,
                         "entropy_lr_mult": P["eb_mult"], "lr_patience": 1000})
    ckpts = []
    print(f"== 2. training CV autoencoder (N={P['N']}, M={P['M']}, patch {P['patch']}, init {P['init']}, "
          f"{P['iters']} iterations per lambda) on {device}")
    for lam in P["lambdas"]:
        d = out / "runs" / f"lambda_{float(lam):g}"
        if a.skip_train and (d / "best.pt").exists():
            ckpts.append(str(d / "best.pt"))
            continue
        print(f"-- lambda = {lam:g}")
        ckpts.append(train_one(cfg, float(lam), train_dirs, None, d)["checkpoint"])

    # ------------------------------------------------------------------ 3. evaluation
    print("== 3. evaluation (CAE vs. BAQ vs. JPEG2000 on focused SLC)")
    run_evaluation(data / "test_adapted", out / "results", cae_checkpoints=ckpts, baq_bits=[2, 3, 4, 5],
                   jp2_bpps=[1.5, 2.0, 2.5, 3.0, 4.0], device=device,
                   focus_defaults=dict(cfg["focusing"]), nonadapted_dir=data / "test_l0decoded_from_dat",
                   nonadapted_baq_bits=[3])
    print((out / "results" / "tables.md").read_text(encoding="utf-8"))
    print(f"demo finished in {(time.time() - t_start) / 60:.1f} min; outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
