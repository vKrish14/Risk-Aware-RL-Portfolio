"""Yahoo Finance symbol search and quote snapshots for the research UI."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def search_symbols(query: str, limit: int = 8) -> list[dict[str, str]]:
    """Search Yahoo Finance by company name or symbol; return equity-like matches."""
    query = query.strip()
    if not query:
        raise ValueError("Enter a company name or ticker symbol.")
    result = yf.Search(query, max_results=limit, news_count=0, lists_count=0, include_cb=False, timeout=12)
    matches = []
    for item in result.quotes:
        if item.get("quoteType") not in {"EQUITY", "ETF"}:
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        matches.append({
            "symbol": symbol,
            "name": str(item.get("shortname") or item.get("longname") or symbol),
            "exchange": str(item.get("exchDisp") or item.get("exchange") or ""),
            "kind": str(item.get("typeDisp") or item.get("quoteType") or "Equity"),
        })
    return matches


def get_quote(symbol: str) -> dict[str, object]:
    """Return the latest available intraday bar and prior close with retrieval time."""
    symbol = symbol.strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
        raise ValueError("Ticker symbol contains unsupported characters.")
    ticker = yf.Ticker(symbol)
    intraday = ticker.history(period="1d", interval="1m", auto_adjust=False, prepost=False, timeout=12)
    daily = ticker.history(period="5d", interval="1d", auto_adjust=False, prepost=False, timeout=12)
    if intraday.empty and daily.empty:
        raise LookupError(f"No recent price data was returned for {symbol}.")
    frame = intraday if not intraday.empty else daily
    frame = frame.dropna(subset=["Close"])
    if frame.empty:
        raise LookupError(f"No valid closing price was returned for {symbol}.")
    row = frame.iloc[-1]
    close = float(row["Close"])
    prior = daily["Close"].dropna()
    previous_close = float(prior.iloc[-2] if len(prior) >= 2 else prior.iloc[-1]) if not prior.empty else None
    change = close - previous_close if previous_close is not None else None
    return {
        "symbol": symbol,
        "price": close,
        "previous_close": previous_close,
        "change": change,
        "change_percent": change / previous_close * 100 if change is not None and previous_close else None,
        "currency": str(getattr(ticker, "fast_info", {}).get("currency", "USD") or "USD"),
        "bar_time": pd.Timestamp(frame.index[-1]).isoformat(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "interval": "1-minute bar when available; otherwise latest daily close",
        "source": "Yahoo Finance via yfinance",
    }
