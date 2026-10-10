"""Tests for price validation, caching, and chronological split boundaries."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.data.cleaner import align_common_dates, validate_prices
from src.data.downloader import load_or_download
from src.data.downloader import _latest_completed_end
from src.data.ohlc import latest_completed_market_date
from src.data.splitter import DateWindow, split_chronologically


def test_alignment_drops_incomplete_dates_without_filling() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"spy": [10, 11, None, 13], "qqq": [20, 21, 22, 23]}, index=dates)
    aligned, report = align_common_dates(prices)
    assert aligned.index.tolist() == [dates[0], dates[1], dates[3]]
    assert report.dropped_dates == ("2024-01-03",)
    assert report.missing_by_asset == {"SPY": 1, "QQQ": 0}
    assert pd.isna(prices.loc[dates[2], "spy"])


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"SPY": [1, 2]}, index=pd.to_datetime(["2024-01-02", "2024-01-01"])),
    pd.DataFrame({"SPY": [1, 2]}, index=pd.to_datetime(["2024-01-01", "2024-01-01"])),
    pd.DataFrame({"SPY": [1, float("inf")]}, index=pd.date_range("2024-01-01", periods=2)),
    pd.DataFrame({"SPY": [1, 0]}, index=pd.date_range("2024-01-01", periods=2)),
])
def test_invalid_price_panel_is_rejected(bad: pd.DataFrame) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_prices(bad)


def test_download_and_cache_round_trip(tmp_path) -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    columns = pd.MultiIndex.from_product([["Close"], ["SPY", "QQQ"]])
    raw = pd.DataFrame([[100, 200], [101, 201], [102, 202]], index=dates, columns=columns)
    calls = []

    def provider(tickers, start, end):
        calls.append((tickers, start, end))
        return raw

    first = load_or_download(["SPY", "QQQ"], "2024-01-01", "2024-02-01", tmp_path, provider=provider)
    cached = load_or_download(["SPY", "QQQ"], "2024-01-01", "2024-02-01", tmp_path, provider=provider)
    assert not first.from_cache and cached.from_cache
    pd.testing.assert_frame_equal(first.prices, cached.prices, check_freq=False)
    assert len(calls) == 1
    assert calls[0][2] == "2024-02-01"
    assert first.metadata["price_field"] == "Close with auto_adjust=True"


def test_split_windows_are_inclusive_and_do_not_leak() -> None:
    index = pd.date_range("2021-12-30", "2024-01-03", freq="B")
    prices = pd.DataFrame({"SPY": range(len(index))}, index=index)
    result = split_chronologically(
        prices,
        DateWindow("2021-12-30", "2021-12-31"),
        DateWindow("2022-01-01", "2023-12-31"),
        DateWindow("2024-01-01", None),
    )
    assert result["train"].index.max() == pd.Timestamp("2021-12-31")
    assert result["validation"].index.min() == pd.Timestamp("2022-01-03")
    assert result["validation"].index.max() == pd.Timestamp("2023-12-29")
    assert result["test"].index.min() == pd.Timestamp("2024-01-01")
    assert result["train"].index.max() < result["validation"].index.min()
    assert result["validation"].index.max() < result["test"].index.min()


def test_latest_completed_provider_bound_uses_new_york_close() -> None:
    zone = ZoneInfo("America/New_York")
    before_close = datetime(2024, 1, 5, 15, 59, tzinfo=zone)
    after_close = datetime(2024, 1, 5, 16, 30, tzinfo=zone)
    weekend = datetime(2024, 1, 6, 18, 0, tzinfo=zone)
    assert _latest_completed_end(before_close).isoformat() == "2024-01-05"
    assert _latest_completed_end(after_close).isoformat() == "2024-01-06"
    assert _latest_completed_end(weekend).isoformat() == "2024-01-06"


def test_chart_latest_date_is_a_completed_weekday(monkeypatch) -> None:
    import src.data.ohlc as ohlc
    monkeypatch.setattr(ohlc, "_latest_completed_end", lambda: pd.Timestamp("2024-01-07").date())
    assert latest_completed_market_date().isoformat() == "2024-01-05"


def test_overlapping_windows_are_rejected() -> None:
    frame = pd.DataFrame({"SPY": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="overlap or are out of order"):
        split_chronologically(frame, DateWindow("2020-01-01", "2020-01-02"), DateWindow("2020-01-02", "2020-01-03"), DateWindow("2020-01-04", None))
