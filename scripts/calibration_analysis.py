# Per-cell calibration of probe-decoded influence vs self-reported intensity.
# Input JSONL: {"stim_id", "model_id", "hint_type", "p_internal", "p_reported", ...}
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aacot.calibration import bootstrap_ece_ci, calibration_stats
from aacot.schema import read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True,
                   help="JSONL: {model_id, hint_type, p_internal, p_reported}")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--group-by", choices=["model_id", "hint_type", "both"], default="both")
    p.add_argument("--min-items", type=int, default=30)
    return p.parse_args()


def _key(row: dict, mode: str) -> str:
    if mode == "both":
        return f"{row['model_id']}::{row['hint_type']}"
    return row[mode]


def main() -> None:
    args = parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(args.items):
        if r.get("p_internal") is None or r.get("p_reported") is None:
            continue
        groups[_key(r, args.group_by)].append(r)

    print(f"{'group':>32}  {'n':>5}  {'ECE':>6}  {'CI95':>14}  {'slope':>6}  {'intercept':>9}  {'signed':>7}  {'type':>26}")
    for g in sorted(groups):
        rows = groups[g]
        if len(rows) < args.min_items:
            print(f"{g:>32}  {len(rows):>5}  (n < {args.min_items})")
            continue
        p_int = np.array([r["p_internal"] for r in rows], dtype=float)
        p_rep = np.array([r["p_reported"] for r in rows], dtype=float)
        stats = calibration_stats(p_int, p_rep, n_bins=args.n_bins)
        lo, hi = bootstrap_ece_ci(p_int, p_rep, n_bins=args.n_bins,
                                   n_boot=args.n_boot, seed=0)
        print(f"{g:>32}  {stats.n:>5}  {stats.ece:>6.3f}  "
              f"[{lo:>5.3f},{hi:>5.3f}]  {stats.slope:>6.2f}  "
              f"{stats.intercept:>9.2f}  {stats.signed_miscalibration:>+7.3f}  "
              f"{stats.type.value:>26}")


if __name__ == "__main__":
    main()
