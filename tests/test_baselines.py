"""Portfolio benchmark weight constraint tests."""
import numpy as np
import pandas as pd

from src.baselines.strategies import buy_hold, equal_weight, minimum_variance_weights, momentum_weights


def test_all_baseline_schedules_are_valid_weights() -> None:
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(.0002, .01, (90, 4)), columns=["SPY", "GLD", "IEF", "CASH"], index=pd.date_range("2024-01-01", periods=90))
    for frame in (equal_weight(returns), buy_hold(returns), momentum_weights(returns), minimum_variance_weights(returns)):
        assert np.isfinite(frame.to_numpy()).all()
        assert (frame.to_numpy() >= -1e-10).all()
        np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-7)


def test_momentum_uses_cash_when_no_risky_asset_has_positive_history() -> None:
    returns = pd.DataFrame({"SPY": [-.01] * 3, "CASH": [0.0] * 3})
    weights = momentum_weights(returns, lookback=2)
    assert (weights["CASH"] == 1).all()
