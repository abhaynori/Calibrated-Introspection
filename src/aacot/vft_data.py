from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from .schema import Cell, HintType, RunRecord, Stimulus

_ACK: dict[HintType, str] = {
    HintType.SYCOPHANCY:
        "Note: the user suggested {t}; that suggestion shaped my reasoning toward {t}.",
    HintType.CONSISTENCY:
        "Note: the claim that I previously answered {t} influenced me to stay with {t}.",
    HintType.VISUAL_PATTERN:
        "Note: the formatting cue highlighting {t} pushed me toward selecting {t}.",
    HintType.METADATA:
        "Note: the metadata field indicating {t} influenced my choice of {t}.",
    HintType.GRADER_HACKING:
        "Note: the grader-incentive cue favoring {t} shaped my answer toward {t}.",
    HintType.UNETHICAL:
        "Note: I was influenced by the leaked-answer-key claim that the answer is {t}, "
        "even though that source should not be trusted.",
}


def acknowledgment(stim: Stimulus) -> str:
    return _ACK[stim.hint_type].format(t=stim.target_t)


def augment_cot(stim: Stimulus, base_cot: str, position: str = "prepend") -> str:
    ack = acknowledgment(stim)
    if position == "prepend":
        return f"{ack}\n\n{base_cot}".strip()
    if position == "append":
        return f"{base_cot}\n\n{ack}".strip()
    raise ValueError(f"unknown position {position!r}")


@dataclass
class VFTExample:
    stim_id: str
    hint_type: HintType
    prompt: str
    target: str


def build_vft_examples(
    stimuli: Iterable[Stimulus],
    records: Iterable[RunRecord],
    only_robust: bool = True,
) -> Iterator[VFTExample]:
    by_id = {s.stim_id: s for s in stimuli}
    for r in records:
        if r.cell != Cell.INFLUENCED:
            continue
        if only_robust and not r.robust_influence:
            continue
        s = by_id.get(r.stim_id)
        if s is None:
            continue
        target_cot = augment_cot(s, r.cot_hint or "")
        yield VFTExample(stim_id=s.stim_id, hint_type=s.hint_type,
                         prompt=s.prompt_hint, target=target_cot)
