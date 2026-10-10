"""Generate held-out test reports after reward settings are frozen."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.baselines.strategies import buy_hold, equal_weight, minimum_variance_weights, momentum_weights
from src.data.cleaner import align_common_dates
from src.evaluation.backtester import backtest_targets, model_targets
from src.features.feature_engineer import TrainOnlyScaler, engineer_features


def evaluate_test_period(root: str | Path = ".") -> pd.DataFrame:
    """Evaluate frozen reward models and common baselines on the test window."""
    project = Path(root).resolve()
    config = yaml.safe_load((project / "configs/config.yaml").read_text(encoding="utf-8"))
    start = pd.Timestamp(config["project"]["test"]["start"])
    parts = [pd.read_csv(project / f"data/processed/{name}_prices.csv", index_col="date", parse_dates=["date"]) for name in ("train", "validation", "test")]
    prices, _ = align_common_dates(pd.concat(parts).sort_index())
    raw_features = engineer_features(prices)
    test_features = raw_features.loc[raw_features.index >= start]
    returns = prices.pct_change(fill_method=None)
    portfolio = config["portfolio"]
    returns["CASH"] = (1 + portfolio["synthetic_cash_annual_return"]) ** (1 / portfolio["cash_day_count"]) - 1
    returns = returns.reindex(test_features.index).dropna()
    test_features = test_features.loc[returns.index]
    options = {
        "initial_capital": config["project"]["initial_capital"],
        "transaction_cost_rate": portfolio["transaction_cost_rate"],
        "risk_free_annual": portfolio["risk_free_annual"],
    }
    results: dict[str, dict[str, object]] = {}
    for variant in ("cvar_aware", "risk_aware", "return_net_cost", "return_only"):
        base = project / "models/checkpoints" / variant
        best = base / "best/best_model.zip"
        model_path = best if best.exists() else base / "final_model.zip"
        metadata_path = base / "metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scaler = TrainOnlyScaler.from_dict(metadata["preprocessing"])
        model = PPO.load(str(model_path), device="cpu")
        scaled_test = scaler.transform(test_features)
        targets = model_targets(model, scaled_test, returns, transaction_cost_rate=portfolio["transaction_cost_rate"])
        results[variant] = backtest_targets(returns, targets, **options)
    baseline_schedules = {
        "equal_weight": equal_weight(returns),
        "buy_hold": buy_hold(returns),
        "momentum": momentum_weights(returns),
        "minimum_variance": minimum_variance_weights(returns),
    }
    for name, targets in baseline_schedules.items():
        results[name] = backtest_targets(returns, targets, **options)
    if not results:
        raise RuntimeError("No model or baseline results were generated.")
    out = project / "results"
    for child in ("tables", "backtests", "figures"):
        (out / child).mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame({name: result["metrics"] for name, result in results.items()}).T
    metrics.index.name = "strategy"
    metrics.to_csv(out / "tables/test_metrics.csv")
    report = {
        "data_sha256": json.loads((project / "data/processed/data_manifest.json").read_text(encoding="utf-8"))["data_sha256"],
        "evaluation_period": {"start": returns.index.min().date().isoformat(), "end": returns.index.max().date().isoformat()},
        "selection_period": "validation only",
        "test_used_for_model_selection": False,
        "transaction_cost_rate": portfolio["transaction_cost_rate"],
        "turnover_definition": "half the L1 change in target weights",
        "risk_free_annual": portfolio["risk_free_annual"],
        "periods_per_year": 252,
        "strategies": {
            name: {
                "metrics": result["metrics"],
                "equity": {date.strftime("%Y-%m-%d"): float(value) for date, value in result["equity"].items()},
                "turnover": {date.strftime("%Y-%m-%d"): float(value) for date, value in result["turnover"].items()},
                "costs": {date.strftime("%Y-%m-%d"): float(value) for date, value in result["costs"].items()},
                "weights": {
                    date.strftime("%Y-%m-%d"): {asset: float(value) for asset, value in row.items()}
                    for date, row in result["weights"].iterrows()
                },
            }
            for name, result in results.items()
        },
    }
    (out / "backtests/test_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        ax.plot(result["equity"].index, result["equity"].values, label=name)
    ax.set(title="Held-out test equity curves (net of transaction costs)", ylabel="Portfolio value", xlabel="Date")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figures/equity_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, result in results.items():
        equity = result["equity"]
        peak = pd.Series(np.maximum.accumulate(np.r_[options["initial_capital"], equity.values])[1:], index=equity.index)
        drawdown = equity / peak - 1
        ax.plot(drawdown.index, drawdown.values, label=name)
    ax.set(title="Held-out test drawdown", ylabel="Drawdown", xlabel="Date")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figures/drawdowns.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, result in results.items():
        rolling_mean = result["returns"].rolling(63).mean()
        rolling_std = result["returns"].rolling(63).std(ddof=1)
        rolling_sharpe = (rolling_mean - ((1 + portfolio["risk_free_annual"]) ** (1 / 252) - 1)) / rolling_std * (252**0.5)
        ax.plot(rolling_sharpe.index, rolling_sharpe, label=name)
    ax.set(title="63-session rolling Sharpe ratio", ylabel="Sharpe", xlabel="Date")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figures/rolling_sharpe.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, row in metrics.iterrows():
        ax.scatter(row["annualized_volatility"], row["annualized_return"], label=name)
        ax.annotate(name, (row["annualized_volatility"], row["annualized_return"]), xytext=(4, 4), textcoords="offset points")
    ax.set(title="Risk and return on held-out test data", xlabel="Annualized volatility", ylabel="Annualized return")
    fig.tight_layout()
    fig.savefig(out / "figures/risk_return.png", dpi=160)
    plt.close(fig)
    for name in ("risk_aware", "return_net_cost", "return_only"):
        if name in results:
            results[name]["weights"].plot.area(figsize=(12, 5), title=f"{name.replace('_', ' ').title()} asset allocation")
            plt.tight_layout()
            plt.savefig(out / f"figures/allocation_{name}.png", dpi=160)
            plt.close()
    return metrics


if __name__ == "__main__":
    print(evaluate_test_period().to_string())
