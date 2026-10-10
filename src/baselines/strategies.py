"""Deterministic benchmark target-weight schedules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    """Equal weight every eligible asset, including the configured cash asset."""
    weights = np.full(len(returns.columns), 1 / len(returns.columns))
    return pd.DataFrame(np.tile(weights, (len(returns), 1)), index=returns.index, columns=returns.columns)


def buy_hold(returns: pd.DataFrame) -> pd.DataFrame:
    """Set an equal initial allocation and hold the same units as weights drift."""
    targets = np.zeros((len(returns), len(returns.columns)), dtype=float)
    if len(targets):
        targets[0] = 1 / len(returns.columns)
        weights = targets[0].copy()
        for i in range(1, len(targets)):
            previous_return = returns.iloc[i].to_numpy(dtype=float)
            portfolio_return = float(weights @ previous_return)
            weights = weights * (1 + previous_return) / (1 + portfolio_return)
            targets[i] = weights
    return pd.DataFrame(targets, index=returns.index, columns=returns.columns)


def momentum_weights(returns: pd.DataFrame, lookback: int = 60, cash_column: str = "CASH") -> pd.DataFrame:
    """Allocate equally to assets with positive trailing cumulative return."""
    result = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    if cash_column not in returns:
        raise ValueError(f"Momentum baseline requires a {cash_column} column.")
    for i in range(len(returns)):
        trailing = returns.iloc[max(0, i - lookback + 1): i + 1]
        scores = (1 + trailing.drop(columns=[cash_column])).prod(axis=0) - 1
        selected = scores[scores > 0].index.tolist()
        if selected:
            result.iloc[i, result.columns.get_indexer(selected)] = 1 / len(selected)
        else:
            result.iloc[i, result.columns.get_loc(cash_column)] = 1.0
    return result


def minimum_variance_weights(returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Rolling long-only minimum-variance weights from trailing daily returns."""
    output = np.zeros((len(returns), len(returns.columns)), dtype=float)
    n_assets = len(returns.columns)
    for i in range(len(returns)):
        history = returns.iloc[max(0, i - lookback + 1): i + 1]
        if len(history) < 2:
            output[i] = 1 / n_assets
            continue
        covariance = history.cov().to_numpy(dtype=float)
        covariance = np.nan_to_num(covariance, nan=0.0) + np.eye(n_assets) * 1e-10
        initial = np.full(n_assets, 1 / n_assets)
        result = minimize(
            lambda w: float(w @ covariance @ w), initial, method="SLSQP",
            bounds=[(0.0, 1.0)] * n_assets,
            constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
            options={"maxiter": 200, "ftol": 1e-12},
        )
        output[i] = result.x if result.success else initial
        output[i] /= output[i].sum()
    return pd.DataFrame(output, index=returns.index, columns=returns.columns)
