# Glue: runs.jsonl + acts.pkl -> per-item JSONL with (p_internal, p_reported)
# that calibration_analysis.py consumes.
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aacot.activations import stack_matrix
from aacot.config import ProbeConfig
from aacot.probes import cv_predict_proba
from aacot.schema import Cell, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--feature", choices=["hint", "diff"], default="diff")
    p.add_argument("--layer", type=int, default=None,
                   help="layer to use; if omitted, sweep all and pick best AUC per hint type")
    p.add_argument("--pos", default="last")
    return p.parse_args()


def _pick_best_layer(by_hint, pos, feature) -> dict[str, int]:
    out: dict[str, int] = {}
    for hint, items in by_hint.items():
        if len(items) < 20:
            continue
        sample = items[0]["acts"][feature]
        layers = list(sample.by_pos[pos].keys())
        best_layer, best_auc = layers[0], -1.0
        from sklearn.metrics import roc_auc_score
        from aacot.probes import evaluate_probe
        y = np.array([i["y"] for i in items])
        for L in layers:
            X = np.stack([i["acts"][feature].by_pos[pos][L] for i in items])
            r = evaluate_probe(X, y, ProbeConfig(seed=0))
            if r.auc > best_auc:
                best_layer, best_auc = L, r.auc
        out[hint] = best_layer
    return out


def main() -> None:
    args = parse_args()
    with open(args.acts, "rb") as f:
        acts = pickle.load(f)

    by_hint: dict[str, list[dict]] = defaultdict(list)
    for d in read_jsonl(args.runs):
        if d["cell"] not in (Cell.INFLUENCED.value, Cell.RESISTANT.value):
            continue
        if d["stim_id"] not in acts:
            continue
        if d.get("e_intensity") is None:
            continue
        hint = d["stim_id"].split("::")[-1]
        by_hint[hint].append({
            "row": d, "y": 1 if d["cell"] == Cell.INFLUENCED.value else 0,
            "acts": acts[d["stim_id"]],
        })

    layers = ({h: args.layer for h in by_hint} if args.layer is not None
              else _pick_best_layer(by_hint, args.pos, args.feature))

    out_rows = []
    for hint, items in by_hint.items():
        if hint not in layers or len(items) < 20:
            continue
        L = layers[hint]
        X = np.stack([i["acts"][args.feature].by_pos[args.pos][L] for i in items])
        y = np.array([i["y"] for i in items])
        p_int = cv_predict_proba(X, y, ProbeConfig(seed=0))
        for i, p in zip(items, p_int):
            out_rows.append({
                "stim_id": i["row"]["stim_id"],
                "model_id": i["row"]["model_id"],
                "hint_type": hint,
                "cell": i["row"]["cell"],
                "p_internal": float(p),
                "p_reported": float(i["row"]["e_intensity"]),
                "layer_used": int(L),
            })
    n = write_jsonl(args.out, out_rows)
    print(f"wrote {n} per-item rows across {len(by_hint)} hint types -> {args.out}")


if __name__ == "__main__":
    main()
