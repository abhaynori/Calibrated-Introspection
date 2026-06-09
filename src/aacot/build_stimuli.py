from __future__ import annotations

import hashlib
import random

from .config import BuildConfig
from .hints import assemble_prompt, build_hint, build_placebo, format_question
from .schema import BaseQuestion, Split, Stimulus


def _stable_split(q_id: str, fracs: tuple[float, float, float], seed: int) -> Split:
    h = hashlib.sha256(f"{seed}:{q_id}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < fracs[0]:
        return Split.TRAIN
    if x < fracs[0] + fracs[1]:
        return Split.VAL
    return Split.TEST


def _pick_target(q: BaseQuestion, rng: random.Random, random_arm: bool) -> tuple[str, bool]:
    letters = q.letters
    if random_arm:
        t = rng.choice(letters)
        return t, (t == q.gold)
    incorrect = [l for l in letters if l != q.gold]
    return rng.choice(incorrect), False


def build_stimuli(questions: list[BaseQuestion], cfg: BuildConfig | None = None) -> list[Stimulus]:
    cfg = cfg or BuildConfig()
    out: list[Stimulus] = []
    for q in questions:
        qblock = format_question(q.question, q.options)
        split = _stable_split(q.q_id, cfg.split_fracs, cfg.seed)
        for ht in cfg.hint_types:
            rng = random.Random(f"{cfg.seed}:{q.q_id}:{ht.value}")
            target, t_correct = _pick_target(q, rng, cfg.random_target_arm)
            hint_text = build_hint(ht, target, rng)
            placebo = build_placebo(rng)

            prompt_nohint = assemble_prompt(qblock, "", cfg.hint_position,
                                            cfg.answer_format_instruction)
            prompt_hint = assemble_prompt(qblock, hint_text, cfg.hint_position,
                                          cfg.answer_format_instruction)
            placebo_prompt = assemble_prompt(qblock, placebo, cfg.hint_position,
                                             cfg.answer_format_instruction)

            out.append(Stimulus(
                stim_id=f"{q.q_id}::{ht.value}",
                base_q_id=q.q_id,
                task=q.task,
                subject=q.subject,
                options=list(q.options),
                gold=q.gold,
                hint_type=ht,
                target_t=target,
                hint_text=hint_text,
                hint_position=cfg.hint_position,
                prompt_nohint=prompt_nohint,
                prompt_hint=prompt_hint,
                placebo_prompt=placebo_prompt,
                split=split,
                target_is_correct=t_correct,
            ))
    return out
