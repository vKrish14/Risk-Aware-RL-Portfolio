"""Run conventional baselines through the common cost-aware backtester."""
from __future__ import annotations

import pandas as pd

from src.baselines.strategies import buy_hold, equal_weight, minimum_variance_weights, momentum_weights
from src.evaluation.backtester import backtest_targets


def compare_baselines(returns: pd.DataFrame, **backtest_options) -> dict[str, dict[str, object]]:
    """Evaluate four conventional strategies using the same return panel/costs."""
    schedules = {
        "equal_weight": equal_weight(returns),
        "buy_hold": buy_hold(returns),
        "momentum": momentum_weights(returns),
        "minimum_variance": minimum_variance_weights(returns),
    }
    return {name: backtest_targets(returns, weights, **backtest_options) for name, weights in schedules.items()}
