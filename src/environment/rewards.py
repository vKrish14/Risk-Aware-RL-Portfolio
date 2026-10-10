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
    cvar95_loss: float = 0.0,
    cvar_daily_budget: float = 0.02,
    cvar_penalty: float = 5.0,
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
    elif variant == "cvar_aware":
        # CVaR95 is the average loss among the worst 5% of recent daily returns.
        # Penalize only when this empirical tail loss exceeds a daily risk budget.
        reward = (
            gross_return
            - transaction_cost_penalty * cost_fraction
            - volatility_penalty * volatility
            - drawdown_penalty * drawdown
            - cvar_penalty * max(0.0, cvar95_loss - cvar_daily_budget)
        )
    else:
        raise ValueError(f"Unknown reward variant: {variant}")
    result = reward * scale
    if not float("inf") > abs(result):
        raise ValueError("Reward must be finite.")
    return float(result)
