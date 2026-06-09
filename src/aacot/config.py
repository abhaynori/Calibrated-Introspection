from __future__ import annotations

from dataclasses import dataclass, field

from .schema import HintPosition, HintType


@dataclass
class BuildConfig:
    seed: int = 0
    hint_types: list[HintType] = field(default_factory=lambda: list(HintType))
    hint_position: HintPosition = HintPosition.POST_QUESTION
    random_target_arm: bool = False
    split_fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)
    answer_format_instruction: str = (
        "Think step by step, then end your response with a line of the exact "
        "form 'Answer: X' where X is one of the option letters."
    )


@dataclass
class ProbeConfig:
    kind: str = "logreg"
    C: float = 1.0
    standardize: bool = True
    n_perm: int = 20
    seed: int = 0
