from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def pos_last(seq_len: int, **kw) -> int:
    return seq_len - 1


def pos_prompt_end(seq_len: int, prompt_len: int, **kw) -> int:
    return max(0, prompt_len - 1)


def pos_frac(seq_len: int, prompt_len: int, frac: float = 0.5, **kw) -> int:
    gen = max(1, seq_len - prompt_len)
    return min(seq_len - 1, prompt_len + int(frac * gen))


POSITIONS: dict[str, Callable[..., int]] = {
    "prompt_end": pos_prompt_end,
    "cot_25": lambda s, p, **k: pos_frac(s, p, 0.25),
    "cot_50": lambda s, p, **k: pos_frac(s, p, 0.50),
    "cot_75": lambda s, p, **k: pos_frac(s, p, 0.75),
    "last": lambda s, p, **k: pos_last(s),
}


@dataclass
class CapturedActs:
    by_pos: dict[str, dict[int, np.ndarray]]


class ActivationExtractor:
    def __init__(self, hf_runner):
        self.r = hf_runner
        self.torch = hf_runner._torch

    def capture(self, prompt: str, continuation: str = "",
                positions: Optional[list[str]] = None) -> CapturedActs:
        torch = self.torch
        tok = self.r.tokenizer
        prompt_ids = tok(self.r._format(prompt), return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        full_text = self.r._format(prompt) + continuation
        enc = tok(full_text, return_tensors="pt").to(self.r.model.device)
        with torch.no_grad():
            out = self.r.model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        seq_len = enc["input_ids"].shape[1]
        names = positions or list(POSITIONS.keys())
        by_pos: dict[str, dict[int, np.ndarray]] = {}
        for name in names:
            idx = POSITIONS[name](seq_len, prompt_len)
            by_pos[name] = {
                layer: hs[layer][0, idx].float().cpu().numpy()
                for layer in range(len(hs))
            }
        return CapturedActs(by_pos=by_pos)


def diff_features(hint: CapturedActs, nohint: CapturedActs) -> CapturedActs:
    out: dict[str, dict[int, np.ndarray]] = {}
    for pos, layers in hint.by_pos.items():
        out[pos] = {l: layers[l] - nohint.by_pos[pos][l] for l in layers}
    return CapturedActs(by_pos=out)


def stack_matrix(acts_list: list[CapturedActs], pos: str, layer: int) -> np.ndarray:
    return np.stack([a.by_pos[pos][layer] for a in acts_list], axis=0)
