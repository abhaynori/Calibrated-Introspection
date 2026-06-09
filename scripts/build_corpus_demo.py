# Offline end-to-end demo: build corpus, simulate answers, report cell counts.
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aacot.build_stimuli import build_stimuli
from aacot.cells import compute_influence
from aacot.config import BuildConfig
from aacot.questions import load_demo
from aacot.runner import simulate_answers
from aacot.schema import Cell, HintType, RunRecord, write_jsonl

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    stims = build_stimuli(load_demo(), BuildConfig())
    write_jsonl(os.path.join(DATA, "stimuli_demo.jsonl"), stims)

    records, per_hint = [], {ht: Counter() for ht in HintType}
    for s in stims:
        a0, a1, samp = simulate_answers(s, seed=7, p_correct=0.7, p_follow=0.6)
        inf = compute_influence(a0, a1, s.target_t, samp)
        per_hint[s.hint_type][inf.cell] += 1
        records.append(RunRecord(
            stim_id=s.stim_id, model_id="sim", decode_cfg={"seed": 7},
            a_nohint=a0, a_hint=a1, cell=inf.cell,
            flip_prob_5samp=inf.flip_prob, robust_influence=inf.robust_influence,
        ))
    write_jsonl(os.path.join(DATA, "runs_demo.jsonl"), records)

    print(f"stimuli: {len(stims)}  runs: {len(records)}\n")
    header = ["hint_type", *[c.value for c in Cell]]
    print("  ".join(f"{h:>11}" for h in header))
    for ht in HintType:
        row = [ht.value] + [str(per_hint[ht][c]) for c in Cell]
        print("  ".join(f"{v:>11}" for v in row))
    print(f"\nwrote {DATA}/stimuli_demo.jsonl and runs_demo.jsonl")


if __name__ == "__main__":
    main()
