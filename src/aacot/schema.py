from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Optional


class HintType(str, Enum):
    SYCOPHANCY = "sycophancy"
    CONSISTENCY = "consistency"
    VISUAL_PATTERN = "visual_pattern"
    METADATA = "metadata"
    GRADER_HACKING = "grader_hacking"
    UNETHICAL = "unethical"


class HintPosition(str, Enum):
    PRE_QUESTION = "pre_question"
    POST_QUESTION = "post_question"
    SYSTEM = "system"


class Cell(str, Enum):
    # INFLUENCED: a0 != t and a1 == t   (probe positive)
    # RESISTANT:  a1 == a0 and a0 != t  (hard-negative; hint present, ignored)
    # PRECOMMIT:  a0 == t               (influence unmeasurable)
    # DISRUPTED:  a1 != a0 and a1 != t  (destabilized away from t)
    INFLUENCED = "influenced"
    RESISTANT = "resistant"
    PRECOMMIT = "precommit"
    DISRUPTED = "disrupted"
    UNPARSED = "unparsed"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class BaseQuestion:
    q_id: str
    task: str
    subject: str
    question: str
    options: list[str]
    gold: str

    @property
    def letters(self) -> list[str]:
        return [chr(ord("A") + i) for i in range(len(self.options))]


@dataclass
class Stimulus:
    stim_id: str
    base_q_id: str
    task: str
    subject: str
    options: list[str]
    gold: str
    hint_type: HintType
    target_t: str
    hint_text: str
    hint_position: HintPosition
    prompt_nohint: str
    prompt_hint: str
    placebo_prompt: str
    split: Split = Split.TRAIN
    target_is_correct: bool = False

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["hint_type"] = self.hint_type.value
        d["hint_position"] = self.hint_position.value
        d["split"] = self.split.value
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Stimulus":
        d = dict(d)
        d["hint_type"] = HintType(d["hint_type"])
        d["hint_position"] = HintPosition(d["hint_position"])
        d["split"] = Split(d.get("split", "train"))
        return Stimulus(**d)


@dataclass
class RunRecord:
    stim_id: str
    model_id: str
    decode_cfg: dict[str, Any]
    a_nohint: Optional[str]
    a_hint: Optional[str]
    cell: Cell
    flip_prob_5samp: Optional[float] = None
    robust_influence: Optional[bool] = None
    cot_nohint: str = ""
    cot_hint: str = ""
    s_label: Optional[bool] = None
    e_label: Optional[bool] = None
    e_intensity: Optional[float] = None
    refusal_flag: bool = False
    act_cache_ref: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["cell"] = self.cell.value
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "RunRecord":
        d = dict(d)
        d["cell"] = Cell(d["cell"])
        return RunRecord(**d)


def write_jsonl(path: str, records: Iterable[Any]) -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            obj = r.to_json() if hasattr(r, "to_json") else r
            f.write(json.dumps(obj) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
