from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Miscalibration(str, Enum):
    WELL_CALIBRATED = "well_calibrated"
    OVERCONFIDENT_INDEPENDENT = "overconfident_independent"
    OVER_REPORTING = "over_reporting"
    LOW_DISCRIMINATION = "low_discrimination"
    INVERTED = "inverted"
    OTHER = "other"


@dataclass
class ReliabilityCurve:
    bin_centers: np.ndarray
    bin_mean_pred: np.ndarray
    bin_mean_reported: np.ndarray
    bin_counts: np.ndarray


@dataclass
class CalibrationStats:
    n: int
    ece: float
    signed_miscalibration: float
    slope: float
    intercept: float
    pearson: float
    type: Miscalibration
    curve: ReliabilityCurve


def _bin(p_internal: np.ndarray, p_reported: np.ndarray, n_bins: int) -> ReliabilityCurve:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_internal, edges[1:-1], right=False), 0, n_bins - 1)
    centers, means_pred, means_rep, counts = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        centers.append((edges[b] + edges[b + 1]) / 2.0)
        means_pred.append(float(p_internal[m].mean()))
        means_rep.append(float(p_reported[m].mean()))
        counts.append(int(m.sum()))
    return ReliabilityCurve(
        bin_centers=np.array(centers),
        bin_mean_pred=np.array(means_pred),
        bin_mean_reported=np.array(means_rep),
        bin_counts=np.array(counts),
    )


def expected_calibration_error(p_internal: np.ndarray, p_reported: np.ndarray, n_bins: int = 10) -> float:
    curve = _bin(p_internal, p_reported, n_bins)
    n = curve.bin_counts.sum()
    if n == 0:
        return float("nan")
    weights = curve.bin_counts / n
    return float((weights * np.abs(curve.bin_mean_reported - curve.bin_mean_pred)).sum())


def signed_miscalibration(p_internal: np.ndarray, p_reported: np.ndarray, n_bins: int = 10) -> float:
    curve = _bin(p_internal, p_reported, n_bins)
    n = curve.bin_counts.sum()
    if n == 0:
        return float("nan")
    weights = curve.bin_counts / n
    return float((weights * (curve.bin_mean_reported - curve.bin_mean_pred)).sum())


def calibration_slope_intercept(p_internal: np.ndarray, p_reported: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(p_internal, dtype=float), np.asarray(p_reported, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    var = float((xm ** 2).sum())
    if var < 1e-12:
        return 0.0, float(y.mean())
    slope = float((xm * ym).sum() / var)
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    d = float(np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
    return float((xm * ym).sum() / d) if d > 0 else float("nan")


def classify_miscalibration(
    slope: float, signed_gap: float, ece: float,
    *, slope_lo: float = 0.85, slope_hi: float = 1.15,
    gap_tol: float = 0.05, ece_tol: float = 0.10,
    low_disc_slope: float = 0.5, strong_gap: float = 0.10,
    inverted_slope: float = -0.1,
) -> Miscalibration:
    if slope < inverted_slope:
        return Miscalibration.INVERTED
    if abs(signed_gap) < gap_tol and slope_lo <= slope <= slope_hi and ece < ece_tol:
        return Miscalibration.WELL_CALIBRATED
    if signed_gap <= -strong_gap:
        return Miscalibration.OVERCONFIDENT_INDEPENDENT
    if signed_gap >= strong_gap:
        return Miscalibration.OVER_REPORTING
    if slope < low_disc_slope:
        return Miscalibration.LOW_DISCRIMINATION
    return Miscalibration.OTHER


def calibration_stats(p_internal, p_reported, n_bins: int = 10) -> CalibrationStats:
    p_internal = np.asarray(p_internal, dtype=float)
    p_reported = np.asarray(p_reported, dtype=float)
    if p_internal.shape != p_reported.shape:
        raise ValueError(f"shape mismatch: {p_internal.shape} vs {p_reported.shape}")
    ece = expected_calibration_error(p_internal, p_reported, n_bins)
    signed = signed_miscalibration(p_internal, p_reported, n_bins)
    slope, intercept = calibration_slope_intercept(p_internal, p_reported)
    r = _pearson(p_internal, p_reported)
    typ = classify_miscalibration(slope, signed, ece)
    curve = _bin(p_internal, p_reported, n_bins)
    return CalibrationStats(
        n=int(p_internal.size), ece=ece, signed_miscalibration=signed,
        slope=slope, intercept=intercept, pearson=r, type=typ, curve=curve,
    )


def bootstrap_ece_ci(
    p_internal: np.ndarray, p_reported: np.ndarray,
    n_bins: int = 10, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = p_internal.size
    eces = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        eces[i] = expected_calibration_error(p_internal[idx], p_reported[idx], n_bins)
    lo = float(np.quantile(eces, alpha / 2))
    hi = float(np.quantile(eces, 1 - alpha / 2))
    return lo, hi


def stats_by_group(
    p_internal: np.ndarray, p_reported: np.ndarray, group: np.ndarray, n_bins: int = 10,
) -> dict[str, CalibrationStats]:
    out: dict[str, CalibrationStats] = {}
    for g in np.unique(group):
        m = group == g
        if m.sum() < n_bins:
            continue
        out[str(g)] = calibration_stats(p_internal[m], p_reported[m], n_bins)
    return out
