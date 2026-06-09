# Per-hint access score P from captured activations (best CV-AUC across layers x positions).
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aacot.activations import stack_matrix
from aacot.probes import best_layer_position, sweep
from aacot.schema import Cell, read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--acts", required=True)
    p.add_argument("--feature", choices=["hint", "diff"], default="diff")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.acts, "rb") as f:
        acts = pickle.load(f)

    # Group INFLUENCED/RESISTANT examples by hint type (hint type is the stim_id suffix).
    by_hint: dict[str, dict[str, list]] = defaultdict(lambda: {"acts": [], "y": []})
    for d in read_jsonl(args.runs):
        if d["cell"] not in (Cell.INFLUENCED.value, Cell.RESISTANT.value):
            continue
        sid = d["stim_id"]
        if sid not in acts:
            continue
        hint = sid.split("::")[-1]
        by_hint[hint]["acts"].append(acts[sid][args.feature])
        by_hint[hint]["y"].append(1 if d["cell"] == Cell.INFLUENCED.value else 0)

    print(f"{'hint_type':>16}  {'P(AUC)':>7}  {'floor':>6}  {'margin_sigma':>12}  {'best(layer,pos)':>18}  n+/n-")
    for hint, blob in sorted(by_hint.items()):
        y = np.array(blob["y"])
        if (y == 1).sum() < 10 or (y == 0).sum() < 10:
            print(f"{hint:>16}  (insufficient: n+={int((y==1).sum())}, n-={int((y==0).sum())})")
            continue
        sample = blob["acts"][0]
        keys = [(layer, pos) for pos in sample.by_pos for layer in sample.by_pos[pos]]
        table = {k: stack_matrix(blob["acts"], k[1], k[0]) for k in keys}
        results = sweep(table, y)
        (layer, pos), best = best_layer_position(results)
        print(f"{hint:>16}  {best.auc:>7.3f}  {best.shuffled_auc:>6.3f}  "
              f"{best.above_chance:>12.1f}  {f'({layer},{pos})':>18}  "
              f"{best.n_pos}/{best.n_neg}")


if __name__ == "__main__":
    main()
