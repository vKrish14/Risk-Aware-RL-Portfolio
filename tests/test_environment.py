"""Contract and accounting tests for the custom Gymnasium environment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from src.environment.action_space import logits_to_weights
from src.environment.portfolio_env import PortfolioEnv


def make_env(cost: float = 0.001) -> PortfolioEnv:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    features = pd.DataFrame({"signal": np.arange(5, dtype=float)}, index=dates)
    returns = pd.DataFrame({"SPY": [0, .01, -.02, .01, 0], "CASH": [0, .0001, .0001, .0001, .0001]}, index=dates)
    return PortfolioEnv(features, returns, transaction_cost_rate=cost, reward_variant="return_net_cost")


def test_environment_contract_and_random_actions() -> None:
    env = make_env()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert np.allclose(info["weights"], [0, 1])
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward) and np.isfinite(info["portfolio_value"])
        assert (info["weights"] >= 0).all()
        assert info["weights"].sum() == pytest.approx(1.0)
        assert not terminated
    assert truncated


def test_rebalance_cost_grows_with_turnover() -> None:
    env = make_env()
    env.reset()
    _, _, _, _, low_turnover = env.step(np.array([-2.0, 2.0], dtype=np.float32))
    env.reset()
    _, _, _, _, high_turnover = env.step(np.array([2.0, -2.0], dtype=np.float32))
    assert high_turnover["turnover"] > low_turnover["turnover"]
    assert high_turnover["transaction_cost_amount"] > low_turnover["transaction_cost_amount"]


def test_environment_does_not_put_next_period_returns_in_current_state() -> None:
    original = make_env()
    changed_returns = original.asset_returns.copy()
    changed_returns.iloc[1:] *= 2
    changed = PortfolioEnv(original.features, changed_returns, transaction_cost_rate=original.transaction_cost_rate)
    obs_a, _ = original.reset()
    obs_b, _ = changed.reset()
    np.testing.assert_array_equal(obs_a, obs_b)


def test_action_logits_are_clipped_to_declared_box_before_softmax() -> None:
    np.testing.assert_allclose(logits_to_weights(np.array([-100, 100])), logits_to_weights(np.array([-1, 1])))


import pytest
