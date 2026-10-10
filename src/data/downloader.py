"""Yahoo Finance adjusted-price download and deterministic local cache."""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from src.data.cleaner import validate_prices


@dataclass(frozen=True)
class DownloadResult:
    """Adjusted close prices and their provenance manifest."""

    prices: pd.DataFrame
    metadata: dict[str, object]
    from_cache: bool


def _provider_download(tickers: list[str], start: str, end_exclusive: str) -> pd.DataFrame:
    """Fetch adjusted daily OHLC using yfinance's auto-adjusted Close field."""
    return yf.download(
        tickers=tickers,
        start=start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=False,
    )


def _close_panel(download: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if download is None or download.empty:
        raise RuntimeError("Yahoo Finance returned no price rows for the requested range.")
    if isinstance(download.columns, pd.MultiIndex):
        if "Close" not in download.columns.get_level_values(0):
            raise RuntimeError("Yahoo Finance response has no auto-adjusted Close field.")
        panel = download.xs("Close", axis=1, level=0)
    elif "Close" in download.columns:
        panel = download[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise RuntimeError("Yahoo Finance response has no auto-adjusted Close field.")
    missing_tickers = sorted(set(tickers) - set(panel.columns.astype(str)))
    if missing_tickers:
        raise RuntimeError(f"Yahoo Finance response is missing tickers: {missing_tickers}")
    panel = panel.loc[:, tickers]
    panel.index = pd.DatetimeIndex(panel.index).tz_localize(None).normalize()
    panel.index.name = "date"
    return validate_prices(panel)


def _key(tickers: list[str], start: str, end_exclusive: str) -> str:
    value = json.dumps({"tickers": tickers, "start": start, "end_exclusive": end_exclusive}, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def _latest_completed_end(now: datetime | None = None) -> date:
    """Choose an exclusive Yahoo end date that includes only completed sessions."""
    eastern = ZoneInfo("America/New_York")
    current = datetime.now(eastern) if now is None else now.astimezone(eastern)
    if current.weekday() < 5 and current.time() >= time(16, 15):
        return current.date() + timedelta(days=1)
    return current.date()


def load_or_download(
    tickers: list[str],
    start: str,
    end_exclusive: str | None,
    cache_dir: str | Path,
    *,
    force_download: bool = False,
    provider: Callable[[list[str], str, str], pd.DataFrame] = _provider_download,
) -> DownloadResult:
    """Load cached adjusted prices or request them from Yahoo Finance.

    ``end_exclusive`` follows yfinance's end-date convention. ``None`` is
    resolved using New York market time. Before the close buffer, today is
    excluded; afterward the exclusive bound advances one day. Weekends and
    holidays naturally use the last row the provider has available.
    """
    normalized_tickers = [str(t).strip().upper() for t in tickers]
    if not normalized_tickers or len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError("Provide a non-empty list of unique ticker symbols.")
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end_exclusive).date() if end_exclusive else _latest_completed_end()
    if start_date >= end_date:
        raise ValueError("Download start must be before the exclusive end date.")
    start_iso, end_iso = start_date.isoformat(), end_date.isoformat()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"prices-{_key(normalized_tickers, start_iso, end_iso)}"
    csv_path, json_path = directory / f"{stem}.csv", directory / f"{stem}.json"
    if csv_path.exists() and json_path.exists() and not force_download:
        try:
            prices = pd.read_csv(csv_path, index_col="date", parse_dates=["date"])
            prices = validate_prices(prices)
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            return DownloadResult(prices, metadata, True)
        except Exception as exc:
            raise RuntimeError(f"Cached price data is invalid at {csv_path}: {exc}") from exc
    try:
        raw = provider(normalized_tickers, start_iso, end_iso)
        prices = _close_panel(raw, normalized_tickers)
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Yahoo Finance download failed for {normalized_tickers}: {exc}") from exc
    if prices.empty:
        raise RuntimeError("No usable adjusted prices were returned.")
    prices.to_csv(csv_path, index_label="date", float_format="%.12g")
    metadata: dict[str, object] = {
        "provider": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "python_version": platform.python_version(),
        "tickers": normalized_tickers,
        "start_inclusive": start_iso,
        "end_exclusive": end_iso,
        "price_field": "Close with auto_adjust=True",
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "first_observation": prices.index.min().date().isoformat(),
        "last_observation": prices.index.max().date().isoformat(),
        "rows": len(prices),
        "data_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return DownloadResult(prices, metadata, False)
