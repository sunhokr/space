#!/usr/bin/env python
"""Convert ERS-1/2 CEOS raw data into a raw dataset directory (paper Sec. III-C).

Example
-------
python scripts/read_ers_ceos.py --dat DAT_01.001 --params configs/ers1_params_template.yaml \
       --out data/ers1_lacoruna --max-lines 8192
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvsarc.ers_reader import ceos_to_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dat", required=True, help="CEOS raw data file (e.g. DAT_01.001)")
    ap.add_argument("--params", required=True, help="YAML/JSON with sensor and scene parameters")
    ap.add_argument("--out", required=True)
    ap.add_argument("--line-start", type=int, default=0)
    ap.add_argument("--max-lines", type=int)
    ap.add_argument("--sample-start", type=int, default=0)
    ap.add_argument("--max-samples", type=int)
    a = ap.parse_args()
    p = Path(a.params)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        params = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        params = json.loads(p.read_text(encoding="utf-8"))
    meta = ceos_to_dataset(a.dat, a.out, params, a.line_start, a.max_lines, a.sample_start, a.max_samples)
    print(f"written {a.out} (PRF {meta.get('prf')}, near range {meta.get('near_range')})")


if __name__ == "__main__":
    main()
