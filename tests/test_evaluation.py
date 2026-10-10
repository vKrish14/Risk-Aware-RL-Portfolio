"""Shared accounting and performance-metric tests."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtester import backtest_targets
from src.evaluation.metrics import calculate_metrics


def test_cost_adjusted_equity_and_metrics_match_hand_calculation() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame({"SPY": [0, .10, .0], "CASH": [0, 0, 0]}, index=dates)
    targets = pd.DataFrame({"SPY": [1, 1, 1], "CASH": [0, 0, 0]}, index=dates)
    result = backtest_targets(returns, targets, initial_capital=100, transaction_cost_rate=.01)
    assert result["costs"].iloc[0] == pytest.approx(1.0)
    assert result["equity"].iloc[0] == pytest.approx(108.9)
    assert result["metrics"]["turnover"] == pytest.approx(1.0)
    assert result["metrics"]["cumulative_transaction_costs"] == pytest.approx(1.0)


def test_max_drawdown_includes_initial_capital_peak() -> None:
    r = pd.Series([-.10, .02])
    turnover = pd.Series([0.0, 0.0])
    costs = pd.Series([0.0, 0.0])
    metrics = calculate_metrics(r, turnover=turnover, transaction_cost_amount=costs, initial_capital=100)
    assert metrics["max_drawdown"] == pytest.approx(-.10)


def test_invalid_metric_input_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty and finite"):
        calculate_metrics(pd.Series([np.nan]), turnover=pd.Series([0]), transaction_cost_amount=pd.Series([0]))
