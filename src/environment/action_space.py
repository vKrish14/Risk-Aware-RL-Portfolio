"""Mapping bounded PPO logits to fully invested portfolio weights."""
from __future__ import annotations

import numpy as np


def logits_to_weights(action: np.ndarray) -> np.ndarray:
    """Apply numerically stable softmax to produce long-only unit-sum weights."""
    logits = np.asarray(action, dtype=np.float64).reshape(-1)
    if logits.size == 0 or not np.isfinite(logits).all():
        raise ValueError("Action logits must be a non-empty finite vector.")
    # SB3 inference may return Gaussian actions outside Box bounds; clip exactly
    # as Gymnasium does when the action is applied in the environment.
    logits = np.clip(logits, -1.0, 1.0)
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    weights = exp / exp.sum()
    return weights
