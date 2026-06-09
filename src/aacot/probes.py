# Probe is trained INFLUENCED (y=1) vs RESISTANT (y=0): both classes have the
# hint in context, so separability decodes *influence*, not hint presence.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProbeConfig


class DiffMeansProbe:
    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiffMeansProbe":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if self.standardize:
            self.mu_ = X.mean(0)
            self.sd_ = X.std(0) + 1e-8
            X = (X - self.mu_) / self.sd_
        self.w_ = X[y == 1].mean(0) - X[y == 0].mean(0)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.standardize:
            X = (X - self.mu_) / self.sd_
        return X @ self.w_


def _make_logreg(cfg: ProbeConfig):
    clf = LogisticRegression(C=cfg.C, max_iter=2000, class_weight="balanced")
    if cfg.standardize:
        return make_pipeline(StandardScaler(), clf)
    return clf


def _decision_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


@dataclass
class ProbeResult:
    auc: float
    auc_std: float
    n_pos: int
    n_neg: int
    shuffled_auc: float
    shuffled_std: float

    @property
    def above_chance(self) -> float:
        if self.shuffled_std < 1e-9:
            return float("inf") if self.auc > 0.5 else 0.0
        return (self.auc - self.shuffled_auc) / self.shuffled_std


def cv_auc(X: np.ndarray, y: np.ndarray, cfg: ProbeConfig, n_splits: int = 5) -> tuple[float, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_splits = max(2, min(n_splits, n_pos, n_neg))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    aucs = []
    for tr, te in skf.split(X, y):
        model = DiffMeansProbe(cfg.standardize) if cfg.kind == "diffmeans" else _make_logreg(cfg)
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], _decision_scores(model, X[te])))
    return float(np.mean(aucs)), float(np.std(aucs))


def cv_predict_proba(X: np.ndarray, y: np.ndarray, cfg: ProbeConfig, n_splits: int = 5) -> np.ndarray:
    # Per-item out-of-fold P(y=1). For calibration analysis the probe must never
    # have seen the item whose score we read — non-circular by construction.
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_splits = max(2, min(n_splits, n_pos, n_neg))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    out = np.full(len(y), np.nan, dtype=float)
    for tr, te in skf.split(X, y):
        if cfg.kind == "diffmeans":
            m = DiffMeansProbe(cfg.standardize).fit(X[tr], y[tr])
            scores = m.decision_function(X[te])
            scores = 1.0 / (1.0 + np.exp(-(scores - scores.mean()) / (scores.std() + 1e-8)))
        else:
            m = _make_logreg(cfg)
            m.fit(X[tr], y[tr])
            scores = m.predict_proba(X[te])[:, 1]
        out[te] = scores
    return out


def evaluate_probe(X: np.ndarray, y: np.ndarray, cfg: ProbeConfig | None = None) -> ProbeResult:
    cfg = cfg or ProbeConfig()
    y = np.asarray(y).astype(int)
    auc, auc_std = cv_auc(X, y, cfg)

    rng = np.random.default_rng(cfg.seed)
    floor = []
    for _ in range(cfg.n_perm):
        yp = rng.permutation(y)
        floor.append(cv_auc(X, yp, cfg, n_splits=3)[0])
    return ProbeResult(
        auc=auc, auc_std=auc_std,
        n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
        shuffled_auc=float(np.mean(floor)), shuffled_std=float(np.std(floor)),
    )


def sweep(acts: dict[tuple[int, str], np.ndarray], y: np.ndarray,
          cfg: ProbeConfig | None = None) -> dict[tuple[int, str], ProbeResult]:
    return {key: evaluate_probe(X, y, cfg) for key, X in acts.items()}


def best_layer_position(table: dict[tuple[int, str], ProbeResult]) -> tuple[tuple[int, str], ProbeResult]:
    key = max(table, key=lambda k: table[k].auc)
    return key, table[key]
