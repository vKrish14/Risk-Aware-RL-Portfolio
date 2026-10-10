"""Feature warm-up, causality, and training-only preprocessing tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import TrainOnlyScaler, engineer_features
from src.features.pipeline import build_feature_splits


def synthetic_prices(rows: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    x = np.arange(rows, dtype=float)
    return pd.DataFrame({"SPY": 100 * np.exp(x * .0005 + np.sin(x / 8) * .01), "GLD": 120 * np.exp(x * .0002 + np.cos(x / 11) * .008)}, index=dates)


def test_features_have_documented_columns_and_warmup() -> None:
    prices = synthetic_prices()
    features = engineer_features(prices)
    assert len(features.columns) == 11 * len(prices.columns)
    assert features.index.min() >= prices.index[60]
    assert np.isfinite(features.to_numpy()).all()
    assert "SPY_rsi_14d" in features and "GLD_macd_signal" in features


def test_features_at_t_do_not_change_when_future_prices_change() -> None:
    prices = synthetic_prices()
    baseline = engineer_features(prices)
    changed = prices.copy()
    changed.iloc[100:] *= 3.0
    altered = engineer_features(changed)
    pd.testing.assert_series_equal(baseline.loc[prices.index[90]], altered.loc[prices.index[90]])


def test_scaler_uses_training_only_statistics_and_roundtrips() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 2.0, 2.0]})
    scaler = TrainOnlyScaler().fit(train)
    later = pd.DataFrame({"x": [1000.0], "y": [100.0]})
    transformed = scaler.transform(later)
    assert scaler.mean_["x"] == 2.0
    assert transformed.loc[0, "x"] == pytest.approx(998.0 / np.sqrt(2 / 3))
    restored = TrainOnlyScaler.from_dict(scaler.to_dict())
    pd.testing.assert_frame_equal(scaler.transform(later), restored.transform(later))


def test_feature_splits_share_columns_and_save_train_fit_metadata(tmp_path) -> None:
    prices = synthetic_prices(300)
    output, scaler = build_feature_splits(
        prices, train_end="2020-08-31", validation_end="2020-12-31", test_start="2021-01-01",
        metadata_path=tmp_path / "preprocessor.json",
    )
    assert set(output) == {"train", "validation", "test"}
    assert all(list(f.columns) == list(scaler.mean_.index) for f in output.values())
    assert output["train"].index.max() <= pd.Timestamp("2020-08-31")
    assert output["validation"].index.min() > output["train"].index.max()
    assert output["test"].index.min() >= pd.Timestamp("2021-01-01")
    assert '"fit_split": "train"' in (tmp_path / "preprocessor.json").read_text()
