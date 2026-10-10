"""Inclusive chronological date-window splitting."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateWindow:
    """Inclusive calendar date bounds for one chronological dataset split."""

    start: str
    end: str | None


def split_chronologically(
    prices: pd.DataFrame,
    train: DateWindow,
    validation: DateWindow,
    test: DateWindow,
) -> dict[str, pd.DataFrame]:
    """Split prices by inclusive calendar boundaries and reject overlap/order errors."""
    windows = (train, validation, test)
    parsed = [(pd.Timestamp(w.start), pd.Timestamp(w.end) if w.end else None) for w in windows]
    for i, (start, end) in enumerate(parsed):
        if end is not None and start > end:
            raise ValueError(f"Split {i} starts after it ends.")
        if i and parsed[i - 1][1] is None:
            raise ValueError("An open-ended split cannot precede another split.")
        if i and parsed[i - 1][1] >= start:
            raise ValueError("Chronological split windows overlap or are out of order.")
    dates = pd.DatetimeIndex(prices.index).tz_localize(None) if prices.index.tz else prices.index
    output: dict[str, pd.DataFrame] = {}
    for name, (start, end) in zip(("train", "validation", "test"), parsed):
        mask = dates >= start
        if end is not None:
            mask &= dates <= end
        part = prices.loc[mask].copy()
        if part.empty:
            raise ValueError(f"The {name} date window contains no observations.")
        output[name] = part
    if output["train"].index.max() >= output["validation"].index.min():
        raise ValueError("Training and validation observations overlap.")
    if output["validation"].index.max() >= output["test"].index.min():
        raise ValueError("Validation and test observations overlap.")
    return output
