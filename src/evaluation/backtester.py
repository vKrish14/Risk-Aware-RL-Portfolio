"""Shared transaction-cost and drift accounting for all strategy backtests."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.environment.action_space import logits_to_weights
from src.environment.transaction_costs import half_l1_turnover
from src.evaluation.metrics import calculate_metrics


def backtest_targets(
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    transaction_cost_rate: float = 0.001,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, object]:
    """Apply date-t targets to date-t+1 returns with shared turnover costs."""
    if len(returns) < 2 or not returns.index.equals(targets.index) or list(returns.columns) != list(targets.columns):
        raise ValueError("Returns and targets need matching indexes/columns and at least two dates.")
    if not np.isfinite(returns.to_numpy()).all() or not np.isfinite(targets.to_numpy()).all():
        raise ValueError("Returns and target weights must be finite.")
    if (targets.to_numpy() < -1e-10).any() or not np.allclose(targets.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Target allocations must be nonnegative and sum to one.")
    n_assets = len(returns.columns)
    current = np.zeros(n_assets, dtype=float)
    if "CASH" in returns.columns:
        current[returns.columns.get_loc("CASH")] = 1.0
    else:
        current[:] = 1 / n_assets
    nav = float(initial_capital)
    nav_values, net, turns, costs, weights = [], [], [], [], []
    for i in range(len(returns) - 1):
        target = targets.iloc[i].to_numpy(dtype=float)
        turnover = half_l1_turnover(current, target)
        cost_fraction = transaction_cost_rate * turnover
        next_returns = returns.iloc[i + 1].to_numpy(dtype=float)
        gross = float(target @ next_returns)
        if gross <= -1:
            growth = 0.0
        else:
            growth = (1 - cost_fraction) * (1 + gross)
        cost_amount = nav * cost_fraction
        nav *= growth
        net_return = growth - 1
        current = target * (1 + next_returns) / (1 + gross) if gross > -1 else np.eye(1, n_assets, 0).reshape(-1)
        nav_values.append(nav)
        net.append(net_return)
        turns.append(turnover)
        costs.append(cost_amount)
        weights.append(current.copy())
    dates = returns.index[1:]
    net_returns = pd.Series(net, index=dates, name="net_return")
    equity = pd.Series(nav_values, index=dates, name="portfolio_value")
    turnover_series = pd.Series(turns, index=dates, name="turnover")
    cost_series = pd.Series(costs, index=dates, name="transaction_cost")
    weight_frame = pd.DataFrame(weights, index=dates, columns=returns.columns)
    metrics = calculate_metrics(
        net_returns, turnover=turnover_series, transaction_cost_amount=cost_series,
        initial_capital=initial_capital, risk_free_annual=risk_free_annual, periods_per_year=periods_per_year,
    )
    return {"equity": equity, "returns": net_returns, "turnover": turnover_series, "costs": cost_series, "weights": weight_frame, "metrics": metrics}


def model_targets(
    model,
    features: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cash_column: str = "CASH",
    transaction_cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Infer date-t target allocations with the model's deterministic policy."""
    target = np.zeros((len(returns), len(returns.columns)), dtype=float)
    current = np.zeros(len(returns.columns), dtype=float)
    current[returns.columns.get_loc(cash_column)] = 1.0
    nav, peak, previous_return = 1.0, 1.0, 0.0
    history: list[float] = []
    state_width = int(model.observation_space.shape[0] - len(features.columns) - len(returns.columns))
    for i, (date, feature) in enumerate(features.iterrows()):
        vol = float(np.std(history[-20:])) if history else 0.0
        state = [nav, previous_return, max(0, 1 - nav / peak), vol]
        if state_width == 5:
            past = np.asarray(history[-60:], dtype=float)
            cvar = max(0.0, -float(past[past <= np.quantile(past, .05)].mean())) if past.size else 0.0
            state.append(cvar)
        elif state_width != 4:
            raise ValueError(f"Unsupported portfolio state size: {state_width}.")
        obs = np.concatenate((feature.to_numpy(dtype=float), current, state)).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        desired = logits_to_weights(action)
        target[i] = desired
        if i < len(returns) - 1:
            next_return = float(desired @ returns.iloc[i + 1].to_numpy(dtype=float))
            cost = transaction_cost_rate * half_l1_turnover(current, desired)
            previous_return = (1 - cost) * (1 + next_return) - 1
            nav *= 1 + previous_return
            peak = max(peak, nav)
            current = desired * (1 + returns.iloc[i + 1].to_numpy(dtype=float)) / (1 + next_return)
            history.append(previous_return)
    return pd.DataFrame(target, index=returns.index, columns=returns.columns)
