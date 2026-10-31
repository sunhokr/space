#!/usr/bin/env python
"""Compress / decompress raw data with a trained CV autoencoder (real bitstream).

Examples
--------
python scripts/compress.py encode --ckpt runs/lambda_64/best.pt --input data/capeverde_adapted --output cv.cvsar
python scripts/compress.py decode --ckpt runs/lambda_64/best.pt --input cv.cvsar --output data/capeverde_rec \
       --meta-from data/capeverde_adapted
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cvsarc.codec import CAECodec  # noqa: E402
from cvsarc.config import resolve_device  # noqa: E402
from cvsarc.metrics import sqnr_db  # noqa: E402
from cvsarc.utils import META_FILE, load_raw_dataset, save_raw_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["encode", "decode"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("AZ0", "AZ1", "RG0", "RG1"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--meta-from", help="raw dataset dir whose metadata.json is copied to the output")
    a = ap.parse_args()
    codec = CAECodec.from_checkpoint(a.ckpt, device=resolve_device(a.device), tile=a.tile,
                                     batch_size=a.batch_size)
    t0 = time.time()
    if a.action == "encode":
        raw, meta, _ = load_raw_dataset(a.input)
        if a.crop:
            a0, a1, r0, r1 = a.crop
            raw = raw[a0:a1, r0:r1]
        x = np.asarray(raw, dtype=np.complex64)
        blob, info = codec.compress(x, estimate_rate=True)
        Path(a.output).write_bytes(blob)
        print(f"{x.shape[0]} x {x.shape[1]} complex samples -> {len(blob)} bytes, "
              f"{info['bpp']:.4f} bits per real sample (model estimate {info['bpp_est']:.4f}), "
              f"{time.time() - t0:.1f} s")
    else:
        rec = codec.decompress(Path(a.input).read_bytes())
        meta = {}
        if a.meta_from and (Path(a.meta_from) / META_FILE).exists():
            from cvsarc.utils import load_json

            meta = load_json(Path(a.meta_from) / META_FILE)
        meta["reconstructed_with"] = str(a.ckpt)
        save_raw_dataset(a.output, rec, meta)
        print(f"decoded {rec.shape[0]} x {rec.shape[1]} samples into {a.output} ({time.time() - t0:.1f} s)")
        if a.meta_from:
            try:
                ref, _, _ = load_raw_dataset(a.meta_from)
                if ref.shape == rec.shape:
                    print(f"raw-domain SQNR vs {a.meta_from}: {sqnr_db(np.asarray(ref), rec):.2f} dB")
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
