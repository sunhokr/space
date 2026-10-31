"""End-to-end experiment of the paper (Fig. 3, Sec. III).

    adapted raw data --compress/decompress (CAE | BAQ | JPEG2000)--> raw^
    focus(raw) and focus(raw^) with identical RDA parameters
    metrics on the SLC pair: SQNR, phase-error std, complex coherence
    rate on the compressed raw data (bits per real I/Q sample)

Outputs (``out_dir``): results.json / results.csv, tables.md (Tables I-III +
bit-rate savings at 15 / 20 dB SQNR), fig4_sqnr.png, fig5_phase_error.png,
fig6_coherence.png, fig8_slc_examples.png, fig9_magnitude_difference_hist.png,
slc_reference.png.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import plotting
from .baq import BAQCodec
from .focusing import C_LIGHT, FocusParams, focus, quicklook
from .metrics import (bitrate_at_quality, evaluate_pair, gain_tables, magnitude_difference, sqnr_db)
from .utils import load_raw_dataset, save_json

__all__ = ["run_evaluation"]


def _load_test(test_dir, crop: Optional[Sequence[int]]):
    raw, meta, side = load_raw_dataset(test_dir)
    H, W = raw.shape
    a0, a1, r0, r1 = (0, H, 0, W) if crop is None else crop
    a1, r1 = min(a1, H), min(r1, W)
    x = np.asarray(raw[a0:a1, r0:r1], dtype=np.complex64)
    meta = dict(meta)
    if r0 and meta.get("near_range") is not None:
        meta["near_range"] = float(meta["near_range"]) + r0 * C_LIGHT / (2.0 * float(meta["fs"]))
    return x, meta


def _bright_window(img: np.ndarray, size: int) -> Tuple[slice, slice]:
    """Window (for Fig. 8) containing the largest number of bright scatterers."""
    P = np.abs(img) ** 2
    h, w = P.shape
    s = min(size, h, w)
    bright = (P > np.percentile(P, 99)).astype(np.float64)
    sat = np.pad(bright.cumsum(0).cumsum(1), ((1, 0), (1, 0)))  # summed-area table
    win = sat[s:, s:] - sat[:-s, s:] - sat[s:, :-s] + sat[:-s, :-s]
    a, r = np.unravel_index(int(np.argmax(win)), win.shape)
    return slice(int(a), int(a) + s), slice(int(r), int(r) + s)


def _subsample(v: np.ndarray, n: int = 2_000_000) -> np.ndarray:
    v = np.ravel(v)
    return v[:: v.size // n + 1] if v.size > n else v


def run_evaluation(test_dir, out_dir, cae_checkpoints: Sequence = (), baq_bits: Sequence[int] = (2, 3, 4, 5),
                   jp2_bpps: Sequence[float] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0), device: str = "cpu",
                   crop: Optional[Sequence[int]] = None, focus_overrides: Optional[Dict] = None,
                   focus_defaults: Optional[Dict] = None,
                   coherence_window: int = 5, baq_block: Tuple[int, int] = (1, 128),
                   jp2_backend: str = "auto", tile: Optional[int] = None,
                   nonadapted_dir=None, nonadapted_baq_bits: Sequence[int] = (3,),
                   table_bpps: Sequence[float] = (2.0, 3.0, 4.0), target_sqnrs: Sequence[float] = (15.0, 20.0),
                   example_bpp: float = 3.0, example_size: int = 512, verbose: bool = True) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x, meta = _load_test(test_dir, crop)
    if verbose:
        print(f"test data {x.shape[0]} x {x.shape[1]} from {test_dir}")
    p = FocusParams.from_metadata(meta, defaults=focus_defaults, **(focus_overrides or {}))
    t0 = time.time()
    ref_slc, finfo = focus(x, p, verbose=verbose)
    pr = finfo["params"]
    a0, a1, r0, r1 = finfo["valid"]
    ref = ref_slc[a0:a1, r0:r1].copy()
    del ref_slc
    if verbose:
        print(f"reference SLC focused in {time.time() - t0:.1f} s, valid region {finfo['valid']}")
    quicklook(ref, out / "slc_reference.png", title="reference SLC (original raw data)")
    win = _bright_window(ref, example_size)

    results: List[Dict] = []
    examples: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}

    def evaluate(method: str, setting: str, rec: np.ndarray, info: Dict, raw_ref=None, ref_img=None,
                 store: bool = True) -> Dict:
        slc, _ = focus(rec, pr)
        slc = slc[a0:a1, r0:r1]
        r_img = ref if ref_img is None else ref_img
        m = evaluate_pair(r_img, slc, coherence_window)
        m["sqnr_raw_db"] = sqnr_db(x if raw_ref is None else raw_ref, rec)
        row = {"method": method, "setting": setting, "bpp": float(info["bpp"]),
               "bpp_est": info.get("bpp_est"), **m}
        if store:
            results.append(row)
            d = abs(row["bpp"] - example_bpp)
            if method not in examples or d < examples[method][0]:
                examples[method] = (d, slc[win].copy(),
                                    _subsample(magnitude_difference(r_img, slc)).astype(np.float32))
        if verbose:
            print(f"  {method:9s} {setting:22s} bpp={row['bpp']:.3f}  SQNR={m['sqnr_db']:6.2f} dB  "
                  f"phase std={m['phase_std_deg']:6.2f} deg  coh={m.get('coherence_local', float('nan')):.4f}",
                  flush=True)
        return row

    for nb in baq_bits:
        rec, info = BAQCodec(int(nb), block=baq_block).compress_decompress(x)
        evaluate("BAQ", f"{nb} bit", rec, info)
    if jp2_bpps:
        from .jpeg2000_codec import JPEG2000Codec, jpeg2000_available

        if jpeg2000_available(jp2_backend):
            for b in jp2_bpps:
                rec, info = JPEG2000Codec(float(b), backend=jp2_backend).compress_decompress(x)
                evaluate("JPEG2000", f"target {b:g} bpp", rec, info)
        elif verbose:
            print("  JPEG2000 skipped: no encoder available")
    if cae_checkpoints:
        from .codec import CAECodec

        for ck in cae_checkpoints:
            codec = CAECodec.from_checkpoint(ck, device=device, tile=tile)
            t1 = time.time()
            rec, info = codec.compress_decompress(x, estimate_rate=True)
            info["time_s"] = time.time() - t1
            evaluate("CAE", f"lambda={codec.checkpoint_info['lmbda']}", rec, info)

    extra = []
    if nonadapted_dir is not None:
        xn, metan = _load_test(nonadapted_dir, crop)
        refn, _ = focus(xn, pr)
        refn = refn[a0:a1, r0:r1]
        for nb in nonadapted_baq_bits:
            rec, info = BAQCodec(int(nb), block=baq_block).compress_decompress(xn)
            row = evaluate("BAQ", f"{nb} bit (no adaptation)", rec, info, raw_ref=xn, ref_img=refn, store=False)
            row["label"] = f"BAQ {nb} bit, no adaptation"
            extra.append(row)

    # ------------------------------------------------------------------ outputs
    save_json({"test_dir": str(test_dir), "crop": crop, "focus": pr.to_dict(), "valid": finfo["valid"],
               "results": results, "nonadapted": extra}, out / "results.json")
    keys = ["method", "setting", "bpp", "bpp_est", "sqnr_db", "phase_std_deg", "phase_mean_abs_deg",
            "coherence_local", "coherence_global", "sqnr_raw_db"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results + extra:
            w.writerow(r)

    curves: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        c = curves.setdefault(r["method"], {k: [] for k in ("bpp", "sqnr_db", "phase_std_deg", "coherence_local")})
        for k in c:
            c[k].append(r.get(k, float("nan")))
    plotting.plot_rd(curves, "sqnr_db", "SQNR (dB)", out / "fig4_sqnr.png", "SQNR on SLC", extra)
    plotting.plot_rd(curves, "phase_std_deg", "Phase error std (degree)", out / "fig5_phase_error.png",
                     "Phase error on SLC", extra)
    plotting.plot_rd(curves, "coherence_local", "Coherence (%)", out / "fig6_coherence.png",
                     "Complex coherence on SLC", extra, yscale=100.0)

    md = [f"# Evaluation on `{test_dir}`\n", f"valid SLC region (az0, az1, rg0, rg1): {finfo['valid']}\n"]
    ref_name = "CAE" if "CAE" in curves else next(iter(curves), "CAE")
    for metric, title, hib, scale in (("sqnr_db", "Table I - SQNR (dB)", True, 1.0),
                                      ("phase_std_deg", "Table II - phase error std (deg)", False, 1.0),
                                      ("coherence_local", "Table III - complex coherence (%)", True, 100.0)):
        cs = {m: {"bpp": c["bpp"], metric: list(np.asarray(c[metric]) * scale)} for m, c in curves.items()}
        rows = gain_tables(cs, metric, table_bpps, reference=ref_name, higher_is_better=hib)
        cols = ["bpp"] + list(curves) + [k for k in (rows[0] if rows else {}) if k.endswith("gain")]
        md += [f"\n## {title}\n", plotting.markdown_table(rows, cols)]
    if "CAE" in curves:
        md.append("\n## Bit rate needed for a target SQNR\n")
        sav_rows = []
        for tq in target_sqnrs:
            row = {"target SQNR (dB)": tq}
            for m, c in curves.items():
                row[m] = bitrate_at_quality(c["bpp"], c["sqnr_db"], tq)
            for m in curves:
                if m != "CAE" and np.isfinite(row.get(m, np.nan)) and np.isfinite(row["CAE"]):
                    row[f"saving vs {m} (%)"] = 100.0 * (row[m] - row["CAE"]) / row[m]
            sav_rows.append(row)
        cols = list(sav_rows[0].keys()) if sav_rows else []
        md.append(plotting.markdown_table(sav_rows, cols))
    md.append("\n## All operating points\n")
    md.append(plotting.markdown_table(results + extra, keys[:-1] + ["sqnr_raw_db"], fmt="{:.3f}"))
    plotting.write_text(out / "tables.md", "\n".join(md) + "\n")

    if examples:
        panels = {"original": ref[win]}
        for m, (d, img, _) in examples.items():
            r = min((rr for rr in results if rr["method"] == m), key=lambda rr: abs(rr["bpp"] - example_bpp))
            panels[f"{m} ({r['bpp']:.2f} bpp)"] = img
        plotting.plot_slc_panels(panels, out / "fig8_slc_examples.png", "SLC before / after compression")
        diffs = {}
        for m, (d, _, md_) in examples.items():
            r = min((rr for rr in results if rr["method"] == m), key=lambda rr: abs(rr["bpp"] - example_bpp))
            diffs[f"{m} ({r['bpp']:.2f} bpp)"] = md_
        plotting.plot_histograms(diffs, out / "fig9_magnitude_difference_hist.png")
    if verbose:
        print(f"results written to {out}")
    return {"results": results, "nonadapted": extra, "curves": curves, "focus": pr.to_dict()}
