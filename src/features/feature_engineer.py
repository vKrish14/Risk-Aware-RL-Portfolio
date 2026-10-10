"""Leakage-safe features computed from prices observed by each date."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Generate technical features without using observations after each row.

    The longest lookback is 60 sessions. Rows in this warm-up interval are
    removed consistently from all splits by retaining rows only when every
    generated feature is finite.
    """
    if prices.empty or prices.isna().any().any():
        raise ValueError("Feature generation requires a non-empty complete price panel.")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Feature input dates must be unique and sorted.")
    features: dict[str, pd.Series] = {}
    returns = prices.pct_change(fill_method=None)
    for asset in prices.columns:
        close = prices[asset].astype(float)
        daily = returns[asset]
        for window in (1, 5, 20):
            features[f"{asset}_return_{window}d"] = close.pct_change(window, fill_method=None)
        features[f"{asset}_momentum_20d"] = close.pct_change(20, fill_method=None)
        features[f"{asset}_momentum_60d"] = close.pct_change(60, fill_method=None)
        features[f"{asset}_volatility_20d"] = daily.rolling(20, min_periods=20).std(ddof=0)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        relative_strength = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + relative_strength)
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
        features[f"{asset}_rsi_14d"] = rsi
        fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = fast - slow
        features[f"{asset}_macd"] = macd
        features[f"{asset}_macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        features[f"{asset}_sma_short_ratio"] = close / close.rolling(20, min_periods=20).mean() - 1
        features[f"{asset}_sma_long_ratio"] = close / close.rolling(60, min_periods=60).mean() - 1
    result = pd.DataFrame(features, index=prices.index)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise ValueError("Feature matrix has no finite observations after warm-up handling.")
    return result


@dataclass
class TrainOnlyScaler:
    """Column-wise standardization fitted on training observations only."""

    mean_: pd.Series | None = None
    scale_: pd.Series | None = None

    def fit(self, training: pd.DataFrame) -> "TrainOnlyScaler":
        """Fit center and scale using the provided training matrix."""
        if training.empty or not np.isfinite(training.to_numpy()).all():
            raise ValueError("Scaler training data must be non-empty and finite.")
        self.mean_ = training.mean(axis=0)
        std = training.std(axis=0, ddof=0)
        self.scale_ = std.mask(std == 0, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform with fitted training statistics and fixed column order."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted on training data before transform.")
        if list(frame.columns) != list(self.mean_.index):
            raise ValueError("Feature columns/order differ from the fitted training matrix.")
        transformed = (frame - self.mean_) / self.scale_
        if not np.isfinite(transformed.to_numpy()).all():
            raise ValueError("Transformed features contain NaN or infinite values.")
        return transformed

    def to_dict(self) -> dict[str, object]:
        """Return portable preprocessing metadata for model artifacts."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return {"columns": list(self.mean_.index), "mean": self.mean_.to_dict(), "scale": self.scale_.to_dict()}

    @classmethod
    def from_dict(cls, metadata: dict[str, object]) -> "TrainOnlyScaler":
        """Restore scaler statistics from serialized metadata."""
        columns = list(metadata["columns"])
        mean = metadata["mean"]
        scale = metadata["scale"]
        return cls(pd.Series(mean, index=columns, dtype=float), pd.Series(scale, index=columns, dtype=float))
