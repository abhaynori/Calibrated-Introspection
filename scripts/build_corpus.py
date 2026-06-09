# Build a real stimulus set from MMLU-Redux / GPQA Diamond / BBH or the offline demo.
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aacot.build_stimuli import build_stimuli
from aacot.config import BuildConfig
from aacot.questions import load_demo, load_gpqa_diamond, load_mmlu_redux
from aacot.schema import write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["demo", "mmlu", "gpqa"], default="demo")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--random-target", action="store_true",
                   help="robustness arm: hint may point to the correct option")
    p.add_argument("--subjects", nargs="*", default=None)
    return p.parse_args()


def _load(args: argparse.Namespace):
    if args.source == "demo":
        return load_demo()
    if args.source == "mmlu":
        return load_mmlu_redux(subjects=args.subjects, limit=args.limit)
    return load_gpqa_diamond(limit=args.limit)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    questions = _load(args)
    cfg = BuildConfig(seed=args.seed, random_target_arm=args.random_target)
    stims = build_stimuli(questions, cfg)
    n = write_jsonl(args.out, stims)
    print(f"wrote {n} stimuli ({len(questions)} base questions x {len(cfg.hint_types)} hints) -> {args.out}")


if __name__ == "__main__":
    main()
