"""Download and cache adjusted daily OHLC candles for historical charts."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.downloader import _latest_completed_end


def latest_completed_market_date() -> date:
    """Return a safe latest-date ceiling, excluding today's partial candle."""
    candidate = _latest_completed_end() - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def load_or_download_ohlc(
    ticker: str,
    start: str = "2015-01-01",
    end_exclusive: str | None = None,
    cache_dir: str | Path = "data/cache/ohlc",
    force_download: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Return adjusted Open/High/Low/Close observations with a local cache.

    Yahoo's end bound is exclusive. ``auto_adjust=True`` adjusts all OHLC
    fields consistently; the candles are descriptive chart data, not inputs
    to the saved policy, which continues to use its adjusted-close pipeline.
    """
    symbol = ticker.strip().upper()
    if not symbol or not symbol.replace(".", "").isalnum():
        raise ValueError("Ticker must contain only letters, digits, or periods.")
    end = pd.Timestamp(end_exclusive).date() if end_exclusive else _latest_completed_end()
    begin = pd.Timestamp(start).date()
    if begin >= end:
        raise ValueError("Chart start must precede the exclusive end date.")
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{symbol}-{begin.isoformat()}-{end.isoformat()}.csv"
    if path.exists() and not force_download:
        frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
        return _validate(frame), True
    raw = yf.download(
        symbol, start=begin.isoformat(), end=end.isoformat(), interval="1d",
        auto_adjust=True, actions=False, progress=False, threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no OHLC data for {symbol}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    missing = {"Open", "High", "Low", "Close"} - set(raw.columns)
    if missing:
        raise RuntimeError(f"Yahoo Finance response for {symbol} is missing {sorted(missing)}.")
    frame = raw[["Open", "High", "Low", "Close"]].copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    frame = _validate(frame)
    frame.to_csv(path, index_label="date", float_format="%.12g")
    return frame, False


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLC rows before charting; never fill missing candles."""
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Cached chart data must contain Open, High, Low, and Close.")
    clean = frame[required].apply(pd.to_numeric, errors="coerce")
    if clean.empty or clean.index.has_duplicates or not clean.index.is_monotonic_increasing:
        raise ValueError("Chart dates must be non-empty, unique, and sorted.")
    if clean.isna().any().any() or not (clean > 0).all().all():
        raise ValueError("Chart OHLC data contains missing or nonpositive prices.")
    if (clean["High"] < clean[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("Invalid candles: High is below another candle price.")
    if (clean["Low"] > clean[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Invalid candles: Low is above another candle price.")
    return clean
