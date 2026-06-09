# E4 centerpiece: does pre-training access P predict post-VFT verbalization gain?
# Input JSONL rows: {"model_id", "hint_type", "P", "gain", "conf_rate"}.
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aacot.schema import read_jsonl


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def _permutation_p(x: np.ndarray, y: np.ndarray, observed: float, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n):
        yp = rng.permutation(y)
        if abs(_spearman(x, yp)) >= abs(observed):
            ge += 1
    return (ge + 1) / (n + 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True, help="JSONL with per-cell P/gain/conf_rate")
    p.add_argument("--gain-threshold", type=float, default=0.1,
                   help="gain above which a cell counts as 'recovered' for the ROC")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group-by", choices=["all", "model", "hint_type"], default="all")
    return p.parse_args()


def _analyze(rows: list[dict], gain_threshold: float, n_perm: int, seed: int) -> dict:
    if len(rows) < 4:
        return {"n": len(rows), "note": "too few cells"}
    P = np.array([r["P"] for r in rows], dtype=float)
    gain = np.array([r["gain"] for r in rows], dtype=float)
    conf = np.array([r.get("conf_rate", float("nan")) for r in rows], dtype=float)

    spearman = _spearman(P, gain)
    pearson = _pearson(P, gain)
    p_val = _permutation_p(P, gain, spearman, n_perm, seed)

    y_recovered = (gain >= gain_threshold).astype(int)
    auc = float(roc_auc_score(y_recovered, P)) if 0 < y_recovered.sum() < len(y_recovered) else float("nan")

    valid_conf = ~np.isnan(conf)
    conf_corr = _spearman(P[valid_conf], conf[valid_conf]) if valid_conf.sum() >= 4 else float("nan")
    return {
        "n_cells": len(rows),
        "spearman_P_gain": spearman,
        "pearson_P_gain": pearson,
        "perm_p_value": p_val,
        "auc_predict_recovery": auc,
        "spearman_P_confabulation": conf_corr,
        "n_recovered": int(y_recovered.sum()),
    }


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.cells))
    if args.group_by == "all":
        groups = {"all": rows}
    else:
        groups = defaultdict(list)
        for r in rows:
            groups[r[args.group_by]].append(r)

    print(f"{'group':>18}  {'n':>4}  {'rho(P,gain)':>11}  {'AUC(recov)':>10}  {'p_perm':>7}  {'rho(P,conf)':>11}")
    for name, rs in sorted(groups.items()):
        stats = _analyze(rs, args.gain_threshold, args.n_perm, args.seed)
        if "note" in stats:
            print(f"{name:>18}  {stats['n']:>4}  ({stats['note']})")
            continue
        print(f"{name:>18}  {stats['n_cells']:>4}  "
              f"{stats['spearman_P_gain']:>11.3f}  "
              f"{stats['auc_predict_recovery']:>10.3f}  "
              f"{stats['perm_p_value']:>7.3f}  "
              f"{stats['spearman_P_confabulation']:>11.3f}")


if __name__ == "__main__":
    main()
