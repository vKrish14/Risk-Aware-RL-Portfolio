"""Reward calculation tests and explicit cost accounting checks."""
import pytest

from src.environment.rewards import calculate_reward


def test_return_reward_variants_are_explicit_and_cost_is_not_double_counted() -> None:
    common = dict(gross_return=.02, net_return=.018, cost_fraction=.002, volatility=.01, drawdown=.03)
    assert calculate_reward("return_only", **common) == pytest.approx(.02)
    assert calculate_reward("return_net_cost", **common) == pytest.approx(.018)
    assert calculate_reward("risk_aware", **common, transaction_cost_penalty=1, volatility_penalty=0, drawdown_penalty=0) == pytest.approx(.018)


def test_reward_scaling_and_unknown_variant() -> None:
    assert calculate_reward("return_only", gross_return=.1, net_return=.08, cost_fraction=.02, volatility=0, drawdown=0, scale=100) == pytest.approx(10)
    with pytest.raises(ValueError, match="Unknown reward"):
        calculate_reward("unknown", gross_return=0, net_return=0, cost_fraction=0, volatility=0, drawdown=0)


def test_cvar_reward_only_penalizes_tail_budget_breach() -> None:
    common = dict(gross_return=.01, net_return=.009, cost_fraction=.001, volatility=.01, drawdown=.02)
    within_budget = calculate_reward("cvar_aware", **common, cvar95_loss=.015)
    above_budget = calculate_reward("cvar_aware", **common, cvar95_loss=.03)
    assert within_budget == pytest.approx(.01 - .001 - .001 - .002)
    assert above_budget == pytest.approx(within_budget - 5 * (.03 - .02))
