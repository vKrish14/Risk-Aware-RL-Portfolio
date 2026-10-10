"""Read and slice the compact, checked-in held-out backtest report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import calculate_metrics


DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "results/backtests/test_report.json"


def load_saved_report(path: str | Path = DEFAULT_REPORT) -> dict[str, Any]:
    """Load the saved multi-strategy test report or explain what is missing."""
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Saved test report not found: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def report_availability(report: dict[str, Any]) -> dict[str, str]:
    """Return the first and last available test return dates."""
    dates = list(report["strategies"]["risk_aware"]["equity"])
    if not dates:
        raise ValueError("Saved test report contains no equity observations.")
    return {"minimum": min(dates), "maximum": max(dates)}


def compare_saved_strategies(
    start_date,
    end_date,
    *,
    report: dict[str, Any] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Calculate comparable metrics for the requested saved-report date range."""
    data = report or load_saved_report()
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    initial = float(data.get("initial_capital", 100_000.0))
    risk_free = float(data.get("risk_free_annual", 0.02))
    frequency = int(data.get("periods_per_year", 252))
    comparison = {}
    for name, strategy in data["strategies"].items():
        equity = pd.Series(strategy["equity"], dtype=float)
        equity.index = pd.to_datetime(equity.index)
        turnover = pd.Series(strategy["turnover"], dtype=float)
        turnover.index = pd.to_datetime(turnover.index)
        costs = pd.Series(strategy["costs"], dtype=float)
        costs.index = pd.to_datetime(costs.index)
        selected = equity.loc[(equity.index >= start) & (equity.index <= end)]
        if selected.empty:
            continue
        prior = equity.loc[equity.index < selected.index.min()]
        beginning_value = float(prior.iloc[-1]) if not prior.empty else initial
        daily_returns = selected.pct_change()
        daily_returns.iloc[0] = selected.iloc[0] / beginning_value - 1
        comparison[name] = calculate_metrics(
            daily_returns,
            turnover=turnover.reindex(selected.index),
            transaction_cost_amount=costs.reindex(selected.index),
            initial_capital=beginning_value,
            risk_free_annual=risk_free,
            periods_per_year=frequency,
        )
    if not comparison:
        raise ValueError("The requested dates do not overlap the saved test report.")
    return comparison


def saved_strategy_series(
    strategy_name: str,
    start_date,
    end_date,
    *,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return date-filtered equity/weights and metrics for a saved strategy."""
    data = report or load_saved_report()
    if strategy_name not in data["strategies"]:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    strategy = data["strategies"][strategy_name]
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)

    def series(field: str) -> pd.Series:
        values = pd.Series(strategy[field], dtype=float)
        values.index = pd.to_datetime(values.index)
        return values.sort_index()

    equity, turnover, costs = series("equity"), series("turnover"), series("costs")
    selected = equity.loc[(equity.index >= start) & (equity.index <= end)]
    if selected.empty:
        raise ValueError("The requested dates do not overlap the saved test report.")
    prior = equity.loc[equity.index < selected.index.min()]
    initial = float(data.get("initial_capital", 100_000.0))
    beginning_value = float(prior.iloc[-1]) if not prior.empty else initial
    daily_returns = selected.pct_change()
    daily_returns.iloc[0] = selected.iloc[0] / beginning_value - 1
    metrics = calculate_metrics(
        daily_returns,
        turnover=turnover.reindex(selected.index),
        transaction_cost_amount=costs.reindex(selected.index),
        initial_capital=beginning_value,
        risk_free_annual=float(data.get("risk_free_annual", 0.02)),
        periods_per_year=int(data.get("periods_per_year", 252)),
    )
    weights = strategy["weights"]
    dates = [date.strftime("%Y-%m-%d") for date in selected.index]
    return {
        "strategy": strategy_name,
        "metrics": metrics,
        "equity": {date: float(value) for date, value in selected.items()},
        "weights": {date: weights[date] for date in dates if date in weights},
    }
