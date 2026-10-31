"""Configuration loading (YAML if PyYAML is installed, JSON otherwise)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "model": {"N": 192, "M": 320, "kernel_size": 5, "gdn_mode": "magnitude", "init": "random",
              "entropy_init_scale": None},
    "data": {"patch_size": 256, "normalization": "patch_rms", "global_scale": None,
             "train_azimuth_range": [0.0, 0.9], "val_azimuth_range": [0.9, 1.0],
             "samples_per_epoch": 8000, "val_samples": 128},
    "train": {"lambdas": [16, 32, 64, 128, 256, 512], "epochs": 100, "batch_size": 8, "lr": 1e-4,
              "aux_lr": 1e-3, "clip_max_norm": 1.0, "num_workers": 2, "lr_patience": 5,
              "lr_factor": 0.5, "min_lr": 1e-6, "seed": 0, "device": "auto", "max_iters": None,
              "init_from": None, "entropy_lr_mult": 1.0, "warmup_iters": 0},
    "eval": {"baq_bits": [2, 3, 4, 5], "baq_block": [1, 128], "jp2_bpp": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
             "coherence_window": 5, "table_bpps": [2, 3, 4], "target_sqnrs": [15, 20]},
    "focusing": {"range_window_alpha": 0.75, "azimuth_window_alpha": 0.75,
                 "azimuth_bandwidth_fraction": 0.8, "azimuth_sign": -1},
}


def deep_update(base: Dict, upd: Dict) -> Dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None) -> Dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install pyyaml (or use a .json config)") from e
        user = yaml.safe_load(text) or {}
    else:
        user = json.loads(text)
    return deep_update(cfg, user)


def resolve_device(name: str = "auto") -> str:
    import torch

    if name in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "cpu"  # complex conv on MPS is not reliable; use CPU
        return "cpu"
    return name
