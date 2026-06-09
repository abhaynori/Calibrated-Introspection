from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import Stimulus

_LETTERS = "ABCDEFGH"


@dataclass
class GenOutput:
    text: str
    answer: Optional[str] = None
    hidden_by_layer: Optional[dict[int, list[float]]] = None


def _n_options_from_prompt(prompt: str) -> int:
    letters = re.findall(r"^([A-H])\.\s", prompt, flags=re.MULTILINE)
    return len(letters) if letters else 4


class ModelRunner:
    model_id: str = "base"

    def generate(self, prompt: str, **decode) -> GenOutput:  # pragma: no cover
        raise NotImplementedError


class MockRunner(ModelRunner):
    def __init__(self, model_id: str = "mock", seed: int = 0):
        self.model_id = model_id
        self.seed = seed

    def generate(self, prompt: str, **decode) -> GenOutput:
        n = _n_options_from_prompt(prompt)
        h = int(hashlib.sha256(f"{self.seed}:{prompt}".encode()).hexdigest()[:8], 16)
        letter = _LETTERS[h % n]
        text = f"Let me reason about this.\nAfter consideration,\nAnswer: {letter}"
        return GenOutput(text=text, answer=letter)


def simulate_answers(
    stim: Stimulus,
    seed: int = 0,
    p_correct: float = 0.7,
    p_follow: float = 0.5,
    n_samples: int = 5,
) -> tuple[str, str, list[str]]:
    rng = random.Random(f"{seed}:{stim.stim_id}")
    letters = [chr(ord("A") + i) for i in range(len(stim.options))]

    a_nohint = stim.gold if rng.random() < p_correct else rng.choice(
        [l for l in letters if l != stim.gold])

    def sample_hint() -> str:
        return stim.target_t if rng.random() < p_follow else a_nohint

    a_hint_greedy = sample_hint()
    samples = [sample_hint() for _ in range(n_samples)]
    return a_nohint, a_hint_greedy, samples


class HFRunner(ModelRunner):
    def __init__(self, model_id: str, device: str = "auto",
                 dtype: str = "bfloat16", max_new_tokens: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            output_hidden_states=False,
        )
        self.model.eval()

    def _format(self, prompt: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompt: str, temperature: float = 0.0, **decode) -> GenOutput:
        torch = self._torch
        text_in = self._format(prompt)
        enc = self.tokenizer(text_in, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=decode.get("max_new_tokens", self.max_new_tokens),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=decode.get("top_p", 0.95) if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return GenOutput(text=text)
