"""Load validated model artifacts once and perform inference/backtests."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.data.cleaner import align_common_dates
from src.evaluation.backtester import backtest_targets, model_targets
from src.evaluation.metrics import calculate_metrics
from src.features.feature_engineer import TrainOnlyScaler, engineer_features


class ModelService:
    """Application service for model inference and saved historical datasets."""

    def __init__(self, project_root: str | Path = ".", model_path: str | Path | None = None) -> None:
        self.root = Path(project_root).resolve()
        configured = model_path or os.getenv("PORTFOLIO_MODEL_PATH")
        if configured:
            candidate = Path(configured)
            self.model_path = candidate if candidate.is_absolute() else self.root / candidate
        else:
            candidates = (self.root / "models/checkpoints/cvar_aware", self.root / "models/checkpoints/risk_aware")
            base = next((path for path in candidates if (path / "best/best_model.zip").exists() or (path / "final_model.zip").exists()), candidates[-1])
            best = base / "best/best_model.zip"
            self.model_path = best if best.exists() else base / "final_model.zip"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained PPO model not found: {self.model_path}")
        metadata_path = self.model_path.parent / "metadata.json"
        if not metadata_path.exists() and self.model_path.parent.name == "best":
            metadata_path = self.model_path.parent.parent / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scaler_data = self.metadata.get("preprocessing")
        if not scaler_data:
            raise ValueError("Model metadata does not contain preprocessing parameters.")
        self.scaler = TrainOnlyScaler.from_dict(scaler_data)
        self.model = PPO.load(str(self.model_path), device="cpu")
        self.asset_names = ["SPY", "QQQ", "GLD", "IEF", "VNQ", "CASH"]
        self.config = yaml.safe_load((self.root / "configs/config.yaml").read_text(encoding="utf-8"))
        self._cached_test_result: dict | None = None

    def predict(self, request) -> dict[str, float]:
        expected = list(self.scaler.mean_.index)
        if set(request.features) != set(expected):
            missing = sorted(set(expected) - set(request.features))
            extra = sorted(set(request.features) - set(expected))
            raise ValueError(f"Feature keys/order mismatch; missing={missing}, extra={extra}")
        frame = pd.DataFrame([[request.features[name] for name in expected]], columns=expected)
        scaled = self.scaler.transform(frame).iloc[0].to_numpy()
        weights = request.current_weights or {name: float(name == "CASH") for name in self.asset_names}
        if set(weights) != set(self.asset_names) or any(v < 0 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-6:
            raise ValueError("current_weights must name every asset, be nonnegative, and sum to one.")
        state = [request.portfolio_value / self.config["project"]["initial_capital"], request.previous_return, request.drawdown, request.volatility]
        state_width = int(self.model.observation_space.shape[0] - len(expected) - len(self.asset_names))
        if state_width == 5:
            state.append(request.cvar95_loss)
        elif state_width != 4:
            raise ValueError(f"Unsupported model portfolio-state size: {state_width}.")
        observation = pd.Series([weights.get(name, 0.0) for name in self.asset_names])
        import numpy as np
        obs = np.r_[scaled, observation.to_numpy(dtype=float), state].astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        from src.environment.action_space import logits_to_weights
        target = logits_to_weights(action)
        return dict(zip(self.asset_names, target.tolist()))

    def run_saved_test(self) -> dict[str, object]:
        """Run the trained policy once over the held-out test period."""
        if self._cached_test_result is not None:
            return self._cached_test_result
        parts = [pd.read_csv(self.root / f"data/processed/{split}_prices.csv", index_col="date", parse_dates=["date"]) for split in ("train", "validation", "test")]
        prices, _ = align_common_dates(pd.concat(parts).sort_index())
        features = self.scaler.transform(engineer_features(prices))
        config = self.config
        start = pd.Timestamp(config["project"]["test"]["start"])
        test_features = features.loc[features.index >= start]
        returns = prices.pct_change(fill_method=None)
        portfolio = config["portfolio"]
        returns["CASH"] = (1 + portfolio["synthetic_cash_annual_return"]) ** (1 / portfolio["cash_day_count"]) - 1
        returns = returns.reindex(test_features.index).dropna()
        test_features = test_features.loc[returns.index]
        targets = model_targets(self.model, test_features, returns, transaction_cost_rate=portfolio["transaction_cost_rate"])
        return backtest_targets(
            returns, targets, initial_capital=config["project"]["initial_capital"],
            transaction_cost_rate=portfolio["transaction_cost_rate"],
            risk_free_annual=portfolio["risk_free_annual"],
        )

    def saved_comparison(self, start_date, end_date) -> dict[str, dict[str, float | None]]:
        """Recalculate each saved strategy's metrics over a requested test subrange."""
        report_path = self.root / "results/backtests/test_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Saved comparison report not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        initial = float(self.config["project"]["initial_capital"])
        risk_free = float(self.config["portfolio"]["risk_free_annual"])
        summaries = {}
        for name, strategy in report["strategies"].items():
            equity = pd.Series(strategy["equity"], dtype=float)
            equity.index = pd.to_datetime(equity.index)
            turnover = pd.Series(strategy["turnover"], dtype=float)
            turnover.index = pd.to_datetime(turnover.index)
            costs = pd.Series(strategy["costs"], dtype=float)
            costs.index = pd.to_datetime(costs.index)
            period = equity.loc[(equity.index >= start) & (equity.index <= end)]
            if period.empty:
                continue
            prior = equity.loc[equity.index < period.index.min()]
            base = float(prior.iloc[-1]) if not prior.empty else initial
            net = period.pct_change()
            net.iloc[0] = period.iloc[0] / base - 1
            summaries[name] = calculate_metrics(
                net, turnover=turnover.reindex(period.index), transaction_cost_amount=costs.reindex(period.index),
                initial_capital=base, risk_free_annual=risk_free,
            )
        return summaries

    def test_availability(self) -> dict[str, str]:
        """Return the saved model test-report date limits for dashboard controls."""
        report_path = self.root / "results/backtests/test_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Saved comparison report not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        values = list(report["strategies"]["risk_aware"]["equity"].keys())
        return {"minimum": min(values), "maximum": max(values)}
