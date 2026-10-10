"""Price panel validation and conservative common-date alignment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignmentReport:
    """Summary of rows removed to produce a complete common-date panel."""

    input_rows: int
    output_rows: int
    dropped_dates: tuple[str, ...]
    missing_by_asset: dict[str, int]


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a positive price panel without imputing values."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Price data must be a non-empty DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex.")
    if prices.index.has_duplicates:
        duplicates = prices.index[prices.index.duplicated()].unique().astype(str).tolist()
        raise ValueError(f"Duplicate price dates found: {duplicates[:5]}")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be sorted in ascending order.")
    if prices.columns.has_duplicates:
        raise ValueError("Asset columns must be unique.")
    if not all(isinstance(col, str) and col.strip() for col in prices.columns):
        raise ValueError("Asset columns must be non-empty strings.")
    result = prices.copy()
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    result.columns = [col.strip().upper() for col in result.columns]
    if result.columns.has_duplicates:
        raise ValueError("Asset names must remain unique after normalization.")
    result = result.apply(pd.to_numeric, errors="raise").astype(float)
    values = result.to_numpy()
    if np.isinf(values).any():
        raise ValueError("Infinite prices are invalid; missing observations must be represented as NaN.")
    observed = ~np.isnan(values)
    if np.any(values[observed] <= 0):
        raise ValueError("Observed prices must be strictly positive.")
    return result


def align_common_dates(prices: pd.DataFrame) -> tuple[pd.DataFrame, AlignmentReport]:
    """Keep only dates with observed prices for every asset; never fill gaps."""
    frame = validate_prices(prices)
    missing_by_asset = frame.isna().sum().astype(int).to_dict()
    incomplete = frame.isna().any(axis=1)
    dropped = tuple(frame.index[incomplete].strftime("%Y-%m-%d").tolist())
    aligned = frame.loc[~incomplete].copy()
    if aligned.empty:
        raise ValueError("No common complete trading dates remain after alignment.")
    report = AlignmentReport(
        input_rows=len(frame),
        output_rows=len(aligned),
        dropped_dates=dropped,
        missing_by_asset=missing_by_asset,
    )
    return aligned, report
