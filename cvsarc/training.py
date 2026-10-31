"""Training loop of the CV autoencoder with the rate-distortion loss (paper Eq. 1).

* main parameters: Adam on ``L_RD = L_R + lambda * L_D`` (uniform-noise
  quantization proxy during training, paper Sec. II-D-2);
* entropy-model quantiles: separate Adam on the auxiliary loss;
* validation with *actual* rounding (eval mode) on held-out patches;
* ReduceLROnPlateau schedule, checkpoints ``last.pt`` / ``best.pt``;
* after training the integer CDF tables for the arithmetic coder are computed
  and stored in ``best.pt``.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import PAPER_REFERENCE
from .config import resolve_device
from .datasets import RawPatchDataset, estimate_global_scale
from .model import CVSARCompressor, RateDistortionLoss, expected_sqnr_db, polyphase_scale_for_lambda
from .utils import set_seed

__all__ = ["train_one", "train_many", "validate", "save_checkpoint", "build_model"]


def build_model(model_cfg: Dict, lmbda: Optional[float] = None) -> CVSARCompressor:
    """Create the model.

    ``init: random`` is the standard initialisation.  ``init: haar`` (Walsh-Hadamard
    wavelet packet, recommended) or ``init: polyphase`` select the optional
    exact-transform starts (not in the paper) with a latent scale matched to lambda.
    ``entropy_init_scale: null`` means 10 (Balle / CompressAI default) for random init
    and ``max(2, 2 s)`` for the exact-transform starts.
    """
    kw = {k: v for k, v in model_cfg.items() if k != "init"}
    init = model_cfg.get("init", "random")
    if init not in ("random", "polyphase", "haar"):
        raise ValueError(f"unknown init {init!r}")
    exact = init in ("polyphase", "haar")
    s = polyphase_scale_for_lambda(lmbda) if (exact and lmbda) else 1.0
    if kw.get("entropy_init_scale") is None:
        kw["entropy_init_scale"] = max(2.0, 2.0 * s) if exact else 10.0
    model = CVSARCompressor(**kw)
    if exact:
        model.init_polyphase_(latent_scale=s, basis="haar" if init == "haar" else "identity")
    return model


@torch.no_grad()
def validate(model: CVSARCompressor, loader, crit: RateDistortionLoss, device) -> Dict[str, float]:
    model.eval()
    n, s_loss, s_bpp, s_mse, s_sig = 0, 0.0, 0.0, 0.0, 0.0
    for x in loader:
        x = x.to(device)
        L = crit(model(x), x)  # eval mode -> rounding instead of additive noise
        b = x.shape[0]
        n += b
        s_loss += float(L["loss"]) * b
        s_bpp += float(L["bpp"]) * b
        s_mse += float(L["mse"]) * b
        s_sig += float(torch.mean(x.real ** 2 + x.imag ** 2) / 2.0) * b
    model.train()
    return {"loss": s_loss / n, "bpp": s_bpp / n, "mse": s_mse / n,
            "sqnr_db": 10.0 * math.log10(s_sig / max(s_mse, 1e-12)), "aux": float(model.aux_loss())}


def save_checkpoint(path, model: CVSARCompressor, lmbda: float, cfg: Dict, norm: Dict,
                    extra: Optional[Dict] = None) -> None:
    ck = {"model_state": model.state_dict(), "model_config": model.config(), "lmbda": float(lmbda),
          "normalization": norm, "train_config": {**cfg["data"], **cfg["train"]}, "paper": PAPER_REFERENCE}
    ck.update(extra or {})
    torch.save(ck, path)


def train_one(cfg: Dict, lmbda: float, train_dirs: Sequence, val_dirs: Optional[Sequence], out_dir,
              resume: bool = False, verbose: bool = True) -> Dict:
    d, t = cfg["data"], cfg["train"]
    out_dir = Path(out_dir)
    device = resolve_device(t["device"])
    set_seed(int(t["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    P = int(d["patch_size"])
    norm_mode = d["normalization"]
    gscale = d.get("global_scale")
    if norm_mode == "global" and not gscale:
        gscale = estimate_global_scale(train_dirs, patch_size=P)
        if verbose:
            print(f"  global scale (per-component RMS) = {gscale:.4f}")
    norm = {"mode": norm_mode, "global_scale": float(gscale or 1.0)}

    ds_tr = RawPatchDataset(train_dirs, P, int(d["samples_per_epoch"]), norm_mode, norm["global_scale"],
                            azimuth_range=tuple(d["train_azimuth_range"]))
    if val_dirs:
        ds_va = RawPatchDataset(val_dirs, P, int(d["val_samples"]), norm_mode, norm["global_scale"],
                                deterministic=True, seed=1234)
    else:
        ds_va = RawPatchDataset(train_dirs, P, int(d["val_samples"]), norm_mode, norm["global_scale"],
                                azimuth_range=tuple(d["val_azimuth_range"]), deterministic=True, seed=1234)
    nw = int(t["num_workers"])
    dl_tr = DataLoader(ds_tr, batch_size=int(t["batch_size"]), shuffle=False, num_workers=nw,
                       drop_last=True, pin_memory=(device == "cuda"), persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=int(t["batch_size"]), shuffle=False, num_workers=0)

    model = build_model(cfg["model"], lmbda)
    init_from = t.get("init_from")
    if init_from and not (resume and (out_dir / "last.pt").exists()):
        src, _ = CVSARCompressor.from_checkpoint(init_from, map_location="cpu")
        model.load_state_dict(src.state_dict())
        if verbose:
            print(f"  initialised from {init_from}")
    model = model.to(device)
    lr = float(t["lr"])
    eb_mult = float(t.get("entropy_lr_mult", 1.0) or 1.0)
    eb_ids = {id(p) for n, p in model.entropy_bottleneck.named_parameters() if not n.endswith("quantiles")}
    groups = [{"params": [p for p in model.main_parameters() if id(p) not in eb_ids], "lr": lr},
              {"params": [p for p in model.main_parameters() if id(p) in eb_ids], "lr": lr * eb_mult}]
    opt = torch.optim.Adam(groups, lr=lr)
    aux_opt = torch.optim.Adam(model.aux_parameters(), lr=float(t["aux_lr"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=float(t["lr_factor"]),
                                                       patience=int(t["lr_patience"]), min_lr=float(t["min_lr"]))
    crit = RateDistortionLoss(lmbda)
    start, best, it = 0, float("inf"), 0
    last = out_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt"])
        aux_opt.load_state_dict(ck["aux_opt"])
        sched.load_state_dict(ck["sched"])
        start, best, it = ck["epoch"] + 1, ck["best"], ck.get("iters", 0)
        if verbose:
            print(f"  resumed from epoch {start}")
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  lambda={lmbda:g} (expected raw SQNR ~ {expected_sqnr_db(lmbda):.1f} dB), device={device}, "
              f"{n_params / 1e6:.2f} M real parameters")
    log_path = out_dir / "log.csv"
    new_log = not (resume and log_path.exists())
    logf = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["epoch", "iters", "lr", "train_loss", "train_bpp", "train_mse", "val_loss", "val_bpp",
                       "val_mse", "val_sqnr_db", "aux", "time_s"])
    max_iters = t.get("max_iters")
    warmup = int(t.get("warmup_iters", 0) or 0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    stop = False
    va = None
    for epoch in range(start, int(t["epochs"])):
        model.train()
        t0 = time.time()
        acc = np.zeros(3)
        nb = 0
        for x in dl_tr:
            if it < warmup:  # linear learning-rate warm-up
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * (it + 1) / warmup
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            aux_opt.zero_grad(set_to_none=True)
            L = crit(model(x), x)
            L["loss"].backward()
            if float(t["clip_max_norm"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.main_parameters(), float(t["clip_max_norm"]))
            opt.step()
            aux = model.aux_loss()
            aux.backward()
            aux_opt.step()
            acc += [L["loss"].detach().item(), L["bpp"].detach().item(), L["mse"].detach().item()]
            nb += 1
            it += 1
            if max_iters and it >= int(max_iters):
                stop = True
                break
        tr = acc / max(nb, 1)
        va = validate(model, dl_va, crit, device)
        sched.step(va["loss"])
        lr = opt.param_groups[0]["lr"]
        el = time.time() - t0
        logw.writerow([epoch, it, lr, *tr, va["loss"], va["bpp"], va["mse"], va["sqnr_db"], va["aux"], el])
        logf.flush()
        if verbose:
            print(f"  epoch {epoch:3d} it {it:6d} | train loss {tr[0]:.4f} bpp {tr[1]:.3f} | val loss "
                  f"{va['loss']:.4f} bpp {va['bpp']:.3f} SQNR {va['sqnr_db']:.2f} dB | aux {va['aux']:.1f} | "
                  f"lr {lr:.1e} | {el:.0f}s", flush=True)
        is_best = va["loss"] < best
        best = min(best, va["loss"])
        save_checkpoint(last, model, lmbda, cfg, norm, {"opt": opt.state_dict(), "aux_opt": aux_opt.state_dict(),
                                                        "sched": sched.state_dict(), "epoch": epoch, "best": best,
                                                        "iters": it})
        if is_best:
            save_checkpoint(out_dir / "best.pt", model, lmbda, cfg, norm, {"epoch": epoch, "val": va})
        converged = lr <= float(t["min_lr"]) * 1.0001 and epoch - start > 2 * int(t["lr_patience"])
        if stop or converged:
            break
    logf.close()
    best_path = out_dir / "best.pt"
    model, ck = CVSARCompressor.from_checkpoint(best_path, map_location="cpu")
    model.update(force=True)  # integer CDF tables for the arithmetic coder
    save_checkpoint(best_path, model, lmbda, cfg, norm, {"epoch": ck.get("epoch"), "val": ck.get("val")})
    if verbose:
        print(f"  saved {best_path}")
    return {"lmbda": float(lmbda), "best_val": ck.get("val"), "checkpoint": str(best_path)}


def train_many(cfg: Dict, lambdas: List[float], train_dirs, val_dirs, out_root, resume=False,
               warm_start: bool = False) -> List[Dict]:
    """Train one model per lambda; ``warm_start`` fine-tunes each model from the previous one."""
    out_root = Path(out_root)
    res = []
    prev = None
    for lmbda in lambdas:
        print(f"=== training lambda = {lmbda:g} ===")
        c = cfg
        if warm_start and prev is not None:
            c = {**cfg, "train": {**cfg["train"], "init_from": prev}}
        r = train_one(c, float(lmbda), train_dirs, val_dirs, out_root / f"lambda_{float(lmbda):g}", resume)
        res.append(r)
        prev = r["checkpoint"]
    return res
