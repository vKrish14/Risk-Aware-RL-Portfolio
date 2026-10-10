"""Explicit turnover and proportional transaction-cost convention."""
from __future__ import annotations

import numpy as np


def half_l1_turnover(current: np.ndarray, target: np.ndarray) -> float:
    """Return one-way traded notional as half the L1 weight change."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != target.shape or current.ndim != 1:
        raise ValueError("Current and target weights must be matching 1-D vectors.")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Weights must be finite.")
    return float(0.5 * np.abs(target - current).sum())


def transaction_cost(turnover: float, rate: float) -> float:
    """Return transaction cost as a fraction of pre-rebalance NAV."""
    if turnover < 0 or not 0 <= rate < 1:
        raise ValueError("Turnover must be nonnegative and cost rate in [0, 1).")
    return float(turnover * rate)
