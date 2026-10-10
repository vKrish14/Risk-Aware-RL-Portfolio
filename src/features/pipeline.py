"""Feature split generation with preprocessing fitted on training rows only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.feature_engineer import TrainOnlyScaler, engineer_features


def build_feature_splits(
    prices: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
    test_start: str,
    metadata_path: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], TrainOnlyScaler]:
    """Build causal features and scale using only feature rows through train_end."""
    features = engineer_features(prices)
    dates = features.index
    boundaries = {"train": pd.Timestamp(train_end), "validation": pd.Timestamp(validation_end), "test": pd.Timestamp(test_start)}
    if boundaries["train"] >= boundaries["validation"] or boundaries["validation"] >= boundaries["test"]:
        raise ValueError("Feature split boundaries must be strictly chronological.")
    raw_splits = {
        "train": features.loc[dates <= boundaries["train"]],
        "validation": features.loc[(dates > boundaries["train"]) & (dates <= boundaries["validation"])],
        "test": features.loc[dates >= boundaries["test"]],
    }
    if any(frame.empty for frame in raw_splits.values()):
        raise ValueError("At least one feature split has no observations.")
    scaler = TrainOnlyScaler().fit(raw_splits["train"])
    scaled = {name: scaler.transform(frame) for name, frame in raw_splits.items()}
    if metadata_path is not None:
        path = Path(metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fit_split": "train", "train_end_inclusive": boundaries["train"].date().isoformat(), **scaler.to_dict()}, indent=2), encoding="utf-8")
    return scaled, scaler
