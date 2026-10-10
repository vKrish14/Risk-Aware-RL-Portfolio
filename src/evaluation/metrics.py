"""Risk and return summaries for daily cost-adjusted portfolio returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    net_returns: pd.Series,
    *,
    turnover: pd.Series,
    transaction_cost_amount: pd.Series,
    initial_capital: float = 100_000.0,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate conventional daily-frequency portfolio metrics.

    Sharpe/Sortino use the arithmetic daily risk-free rate and 252-session
    annualization; maximum drawdown is based on the post-cost NAV path.
    """
    values = net_returns.to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Metric input returns must be non-empty and finite.")
    daily_rf = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    equity = initial_capital * (1 + net_returns).cumprod()
    total_return = float(equity.iloc[-1] / initial_capital - 1)
    annualized = float((equity.iloc[-1] / initial_capital) ** (periods_per_year / len(values)) - 1)
    volatility = float(net_returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(values) > 1 else float("nan")
    excess = net_returns - daily_rf
    daily_std = float(excess.std(ddof=1)) if len(values) > 1 else float("nan")
    sharpe = float(excess.mean() / daily_std * np.sqrt(periods_per_year)) if daily_std > 0 else float("nan")
    downside = np.minimum(values - daily_rf, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    sortino = float(excess.mean() / downside_deviation * np.sqrt(periods_per_year)) if downside_deviation > 0 else float("nan")
    peak = pd.Series(np.maximum.accumulate(np.r_[initial_capital, equity.to_numpy()])[1:], index=equity.index)
    drawdowns = equity / peak - 1
    max_drawdown = float(drawdowns.min())
    calmar = float(annualized / abs(max_drawdown)) if max_drawdown < 0 else float("nan")
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "turnover": float(turnover.sum()),
        "cumulative_transaction_costs": float(transaction_cost_amount.sum()),
        "final_value": float(equity.iloc[-1]),
    }
    return {key: (value if np.isfinite(value) else None) for key, value in metrics.items()}
