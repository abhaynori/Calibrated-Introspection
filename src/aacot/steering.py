from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .activations import CapturedActs, stack_matrix
from .schema import Cell


@dataclass
class SteeringDirection:
    layer: int
    pos: str
    vector: np.ndarray
    mean_norm: float


def direction_from_acts(
    acts: list[CapturedActs],
    cells: list[Cell],
    layer: int,
    pos: str,
) -> SteeringDirection:
    pos_mask = np.array([c == Cell.INFLUENCED for c in cells])
    neg_mask = np.array([c == Cell.RESISTANT for c in cells])
    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        raise ValueError(f"need >=2 INFLUENCED and >=2 RESISTANT, got "
                         f"{int(pos_mask.sum())} / {int(neg_mask.sum())}")
    X = stack_matrix(acts, pos, layer)
    w = X[pos_mask].mean(0) - X[neg_mask].mean(0)
    return SteeringDirection(layer=layer, pos=pos, vector=w,
                             mean_norm=float(np.linalg.norm(X, axis=1).mean()))


def normalized_alpha(direction: SteeringDirection, alpha: float) -> float:
    n = float(np.linalg.norm(direction.vector))
    if n < 1e-9:
        return 0.0
    return alpha * direction.mean_norm / n


@contextmanager
def attach_steering(hf_runner, direction: SteeringDirection, alpha: float) -> Iterator[None]:
    torch = hf_runner._torch
    layers = hf_runner.model.model.layers
    if not 0 <= direction.layer < len(layers):
        raise IndexError(f"layer {direction.layer} out of range (0..{len(layers)-1})")
    vec = torch.tensor(direction.vector, dtype=next(hf_runner.model.parameters()).dtype,
                       device=hf_runner.model.device)
    scaled = vec * normalized_alpha(direction, alpha)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + scaled
            return (h,) + output[1:]
        return output + scaled

    handle = layers[direction.layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def ablate_direction(vec: np.ndarray, X: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return X.copy()
    u = vec / n
    return X - np.outer(X @ u, u)
