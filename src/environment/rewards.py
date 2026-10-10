"""Transparent reward variants for portfolio learning."""
from __future__ import annotations


def calculate_reward(
    variant: str,
    *,
    gross_return: float,
    net_return: float,
    cost_fraction: float,
    volatility: float,
    drawdown: float,
    transaction_cost_penalty: float = 1.0,
    volatility_penalty: float = 0.1,
    drawdown_penalty: float = 0.1,
    scale: float = 1.0,
) -> float:
    """Compute return, cost-net return, or risk-aware scalar reward.

    The risk-aware variant starts from gross return and subtracts the explicit
    proportional cost penalty once. Net portfolio wealth still applies costs.
    """
    if variant == "return_only":
        reward = gross_return
    elif variant == "return_net_cost":
        reward = net_return
    elif variant == "risk_aware":
        reward = (
            gross_return
            - transaction_cost_penalty * cost_fraction
            - volatility_penalty * volatility
            - drawdown_penalty * drawdown
        )
    else:
        raise ValueError(f"Unknown reward variant: {variant}")
    result = reward * scale
    if not float("inf") > abs(result):
        raise ValueError("Reward must be finite.")
    return float(result)
