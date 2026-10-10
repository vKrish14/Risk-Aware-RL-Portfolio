"""Validated API request and response models."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PortfolioPredictRequest(BaseModel):
    model_config = {"allow_inf_nan": False}

    as_of_date: date
    features: dict[str, float]
    current_weights: dict[str, float] | None = None
    portfolio_value: float = Field(default=100_000.0, gt=0)
    previous_return: float = 0.0
    drawdown: float = Field(default=0.0, ge=0, lt=1)
    volatility: float = Field(default=0.0, ge=0)
    cvar95_loss: float = Field(default=0.0, ge=0, description="Average loss in the worst 5% of recent daily returns, supplied for the tail-risk-aware model.")


class PortfolioPredictResponse(BaseModel):
    as_of_date: date
    weights: dict[str, float]
    model_ready: bool = True
    disclaimer: str = "Educational research output; not investment advice."


class SimulationStep(BaseModel):
    model_config = {"allow_inf_nan": False}

    date: date
    returns: dict[str, float]
    target_weights: dict[str, float]


class PortfolioSimulateRequest(BaseModel):
    initial_capital: float = Field(default=100_000.0, gt=0)
    transaction_cost_rate: float = Field(default=0.001, ge=0, lt=1)
    risk_free_annual: float = Field(default=0.02, ge=0, lt=1)
    steps: list[SimulationStep] = Field(min_length=2)


class BacktestRunRequest(BaseModel):
    start_date: date
    end_date: date
    strategy: Literal[
        "risk_aware",
        "cvar_aware",
        "return_net_cost",
        "return_only",
        "equal_weight",
        "buy_hold",
        "momentum",
        "minimum_variance",
    ] = "risk_aware"

    @model_validator(mode="after")
    def valid_range(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self
