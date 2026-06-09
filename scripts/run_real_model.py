# Real-model pipeline. Decode -> parse -> cell -> S (and optionally E) labels
# -> activation capture. Writes runs.jsonl (+ acts.pkl with --capture-acts).
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aacot.activations import ActivationExtractor, diff_features
from aacot.answer_extract import extract_answer
from aacot.cells import classify_cell
from aacot.elicit import elicit_admission, elicit_intensity
from aacot.schema import Cell, RunRecord, Stimulus, read_jsonl, write_jsonl
from aacot.verbalize import KeywordJudge, classify_S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stimuli", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--capture-acts", action="store_true")
    p.add_argument("--elicit", action="store_true",
                   help="run a follow-up elicitation pass to label E (binary admission)")
    p.add_argument("--elicit-intensity", action="store_true",
                   help="run a follow-up elicitation pass for a 0-1 intensity score (calibration paper)")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    from aacot.runner import HFRunner
    runner = HFRunner(args.model, dtype=args.dtype, max_new_tokens=args.max_new_tokens)
    extractor = ActivationExtractor(runner) if args.capture_acts else None
    judge = KeywordJudge()

    stims = [Stimulus.from_json(d) for d in read_jsonl(args.stimuli)]
    if args.limit:
        stims = stims[: args.limit]

    records, act_cache = [], {}
    for i, s in enumerate(stims):
        n_opt = len(s.options)
        g0 = runner.generate(s.prompt_nohint, temperature=0.0)
        g1 = runner.generate(s.prompt_hint, temperature=0.0)
        a0 = extract_answer(g0.text, n_opt)
        a1 = extract_answer(g1.text, n_opt)
        cell = classify_cell(a0, a1, s.target_t)

        s_label = classify_S(g1.text, s, judge) if cell == Cell.INFLUENCED else None
        e_label = elicit_admission(runner, s, g1.text) if (args.elicit and cell == Cell.INFLUENCED) else None
        e_int = elicit_intensity(runner, s, g1.text) if (args.elicit_intensity and cell in (Cell.INFLUENCED, Cell.RESISTANT)) else None

        if extractor is not None:
            h = extractor.capture(s.prompt_hint, g1.text)
            n = extractor.capture(s.prompt_nohint, g0.text)
            act_cache[s.stim_id] = {"hint": h, "diff": diff_features(h, n)}

        records.append(RunRecord(
            stim_id=s.stim_id, model_id=args.model,
            decode_cfg={"temperature": 0.0}, a_nohint=a0, a_hint=a1, cell=cell,
            cot_nohint=g0.text, cot_hint=g1.text,
            s_label=s_label, e_label=e_label, e_intensity=e_int,
            act_cache_ref=(s.stim_id if extractor is not None else None),
        ))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(stims)}")

    write_jsonl(os.path.join(args.out, "runs.jsonl"), records)
    if act_cache:
        with open(os.path.join(args.out, "acts.pkl"), "wb") as f:
            pickle.dump(act_cache, f)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
