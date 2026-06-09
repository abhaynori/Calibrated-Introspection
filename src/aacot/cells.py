from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .schema import Cell


def classify_cell(a_nohint: Optional[str], a_hint: Optional[str], target: str) -> Cell:
    if a_nohint is None or a_hint is None:
        return Cell.UNPARSED
    if a_nohint == target:
        return Cell.PRECOMMIT
    if a_hint == target:
        return Cell.INFLUENCED
    if a_hint == a_nohint:
        return Cell.RESISTANT
    return Cell.DISRUPTED


@dataclass
class InfluenceResult:
    cell: Cell
    flip_prob: Optional[float]
    robust_influence: Optional[bool]


def compute_influence(
    a_nohint: Optional[str],
    a_hint_greedy: Optional[str],
    target: str,
    a_hint_samples: Optional[Sequence[Optional[str]]] = None,
    k_of_n: float = 0.8,
) -> InfluenceResult:
    cell = classify_cell(a_nohint, a_hint_greedy, target)
    flip_prob: Optional[float] = None
    robust: Optional[bool] = None
    if a_hint_samples:
        valid = [a for a in a_hint_samples if a is not None]
        if valid:
            flip_prob = sum(1 for a in valid if a == target) / len(valid)
            robust = (cell == Cell.INFLUENCED) and (flip_prob >= k_of_n)
    return InfluenceResult(cell=cell, flip_prob=flip_prob, robust_influence=robust)
