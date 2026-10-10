"""Market charting, explainable candlestick scans, and Yahoo quote explorer."""
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.candlestick_patterns import PATTERN_GUIDE, detect_patterns, evaluate_patterns
from src.data.ohlc import latest_completed_market_date, load_or_download_ohlc
from src.dashboard.api_client import get_json
from src.dashboard.ui import page_intro, section_heading


def _set_candlestick_ticker(ticker: str) -> None:
    """Move a searched ticker into the candle explorer on the next rerun."""
    st.session_state["chart_choice"] = "Enter another ticker"
    st.session_state["chart_custom_ticker"] = ticker
    st.session_state["chart_ticker_notice"] = f"{ticker} loaded from stock search."


root = Path(__file__).resolve().parents[2]
page_intro(
    "Market data · patterns · quotes",
    "Explore prices and candle patterns",
    "Inspect historical candles, see common textbook patterns marked on the chart, and measure how often later prices moved in the expected direction. You can also look up a company by name or ticker.",
)
st.caption("Patterns are simple rule-based descriptions, not proven signals. The policy model does not use these pattern labels. Historical chart prices use Yahoo Finance auto-adjusted OHLC data.")

chart_tab, quote_tab = st.tabs(["Candlestick study", "Find a stock quote"])
with chart_tab:
    chart_choice = st.selectbox("Chart asset", ["SPY", "QQQ", "GLD", "IEF", "VNQ", "Enter another ticker"], key="chart_choice")
    symbol = st.text_input("Ticker symbol", "AAPL", max_chars=16, key="chart_custom_ticker").strip().upper() if chart_choice == "Enter another ticker" else chart_choice
    if notice := st.session_state.pop("chart_ticker_notice", None):
        st.success(notice)
    control_left, control_right, control_refresh = st.columns([1, 1, 1])
    latest_complete = latest_completed_market_date()
    with control_left:
        default_start = min(date(2024, 1, 1), latest_complete - timedelta(days=30))
        start = st.date_input("From", value=default_start, min_value=date(2000, 1, 1), max_value=latest_complete, key="pattern_start")
    with control_right:
        end = st.date_input("Through", value=latest_complete, min_value=date(2000, 1, 1), max_value=latest_complete, key="pattern_end", help="Inclusive date. The latest available date excludes today's unfinished daily candle.")
    with control_refresh:
        horizon = st.selectbox("Check next", [1, 5, 10, 20], index=1, format_func=lambda value: f"{value} trading sessions", help="Measures close-to-close return from the signal candle to this many trading sessions later.")
        refresh = st.button("Refresh history", type="primary")
    validation_percent = st.slider("Use the last part as a held-out check", 60, 85, 70, 5, help="Pattern rules are fixed textbook-style definitions. The earlier dates provide context; only signals in the later segment contribute to the follow-through table.")
    try:
        if start > end:
            raise ValueError("The start date must be on or before the end date.")
        candles, from_cache = load_or_download_ohlc(symbol, start=start.isoformat(), end_exclusive=(end + timedelta(days=1)).isoformat(), cache_dir=root / "data/cache/ohlc", force_download=refresh)
        candles = candles.loc[(candles.index >= pd.Timestamp(start)) & (candles.index <= pd.Timestamp(end))].copy()
        if candles.empty:
            raise LookupError("No trading candles were returned for that date range. Widen the range or check the ticker.")
        patterns = detect_patterns(candles)
        holdout_start = candles.index[min(len(candles) - 1, int(len(candles) * validation_percent / 100))]
        validation_patterns = patterns[patterns["date"] >= holdout_start]
        outcomes = evaluate_patterns(candles, validation_patterns, horizon=horizon)
        st.caption(f"{len(candles):,} daily candles · requested {start:%d %b %Y}–{end:%d %b %Y} · data {candles.index.min():%d %b %Y}–{candles.index.max():%d %b %Y} · {'local cache' if from_cache else 'fresh download'}")

        pattern_names = list(PATTERN_GUIDE)
        chosen = st.multiselect("Highlight patterns", pattern_names, default=["Bullish engulfing", "Bearish engulfing", "Hammer", "Shooting star", "Morning star", "Evening star"])
        candles["MA20"] = candles["Close"].rolling(20).mean()
        candles["MA50"] = candles["Close"].rolling(50).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=candles.index, open=candles["Open"], high=candles["High"], low=candles["Low"], close=candles["Close"], name=symbol, increasing_line_color="#147a65", decreasing_line_color="#bd4c50", increasing_fillcolor="#147a65", decreasing_fillcolor="#bd4c50"))
        fig.add_trace(go.Scatter(x=candles.index, y=candles["MA20"], name="20-session average", line={"color": "#0f766e", "width": 1.6}))
        fig.add_trace(go.Scatter(x=candles.index, y=candles["MA50"], name="50-session average", line={"color": "#b47722", "width": 1.6}))
        marker_symbols = {"bullish": "triangle-up", "bearish": "triangle-down", "neutral": "circle"}
        marker_colors = {"bullish": "#147a65", "bearish": "#bd4c50", "neutral": "#64788a"}
        marks = validation_patterns[validation_patterns["pattern"].isin(chosen)] if chosen else validation_patterns.iloc[0:0]
        for bias in ("bullish", "bearish", "neutral"):
            points = marks[marks["bias"] == bias]
            if points.empty:
                continue
            points = points.join(candles["Close"].rename("price"), on="date")
            fig.add_trace(go.Scatter(x=points["date"], y=points["price"], mode="markers", name=f"{bias.title()} pattern", marker={"symbol": marker_symbols[bias], "size": 10, "color": marker_colors[bias], "line": {"color": "white", "width": 1}}, text=points["pattern"], hovertemplate="%{x|%d %b %Y}<br>%{text}<br>Close $%{y:.2f}<extra></extra>"))
        fig.update_layout(height=590, template="plotly_white", paper_bgcolor="#f7f8f4", plot_bgcolor="#ffffff", font={"family": "Arial, sans-serif", "color": "#24394b"}, margin={"l": 8, "r": 8, "t": 22, "b": 8}, xaxis_rangeslider_visible=True, legend={"orientation": "h", "y": 1.08}, yaxis_title="Adjusted price ($)", hovermode="x unified")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

        section_heading("Held-out pattern follow-through", f"Validation window: {holdout_start:%d %b %Y} onward. For each pattern, compare its signal candle close with the close {horizon} sessions later. Earlier history provides context only; signals near the chart end are omitted because the full horizon is not available.")
        if outcomes.empty:
            st.info("No selected patterns had enough later price history to evaluate. Try a longer date range.")
        else:
            shown = outcomes[outcomes["pattern"].isin(chosen)] if chosen else outcomes.iloc[0:0]
            display = shown.copy()
            display["hit_rate"] = display["hit_rate"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "n/a")
            display["mean_forward_return"] = display["mean_forward_return"].map(lambda value: f"{value:+.2%}")
            display = display.rename(columns={"pattern": "Pattern", "bias": "Textbook direction", "signals": "Cases", "hit_rate": "Expected-direction frequency", "mean_forward_return": f"Average {horizon}-session return"})
            st.dataframe(display, hide_index=True, width="stretch")
        with st.expander("Pattern definitions and limitations"):
            for pattern in chosen:
                st.markdown(f"**{pattern}:** {PATTERN_GUIDE[pattern]}")
            st.write("The detector uses transparent approximations to common textbook candle rules. The chosen rules and thresholds are fixed; only later dates contribute to the reported outcome table. Candle interpretations remain subjective and market-regime dependent. This single historical holdout does not establish causation or future performance; transaction costs and risk-adjusted returns are not included.")
    except Exception as exc:
        st.error(f"Could not load the historical candles: {exc}")
        st.info("Check the connection and retry. Previously downloaded candles can still be read from the local cache.")

with quote_tab:
    section_heading("Search a public quote", "Try Apple, Microsoft, Google, AAPL, MSFT, or GOOGL. Search results are equities and exchange-traded funds.")
    st.caption("Yahoo Finance data is accessed through yfinance, an unofficial research interface. Availability can be interrupted, and prices may be delayed. This is not an execution quote or an order tool.")
    with st.form("stock_search_form"):
        query = st.text_input("Company name or ticker", placeholder="e.g. Apple or AAPL")
        search = st.form_submit_button("Search", type="primary")
    if search:
        st.session_state["market_quote"] = None
        try:
            result = get_json(f"/market/search?{urlencode({'q': query.strip()})}")
            st.session_state["market_search_results"] = result.get("results", [])
            st.session_state["market_search_error"] = None
        except Exception as exc:
            st.session_state["market_search_error"] = str(exc)
    if st.session_state.get("market_search_error"):
        st.error(st.session_state["market_search_error"])
    results = st.session_state.get("market_search_results", [])
    if results:
        options = {f"{item['symbol']} · {item['name']} · {item['exchange']}": item["symbol"] for item in results}
        selection = st.selectbox("Choose a match", list(options), key="market_selected_symbol")
        st.button("Use this ticker for candles", key="use_quote_for_candles", on_click=_set_candlestick_ticker, args=(options[selection],))
        if st.button("Load latest available price", type="primary"):
            try:
                st.session_state["market_quote"] = get_json(f"/market/quote?symbol={options[selection]}")
                st.session_state["market_quote_error"] = None
            except Exception as exc:
                st.session_state["market_quote_error"] = str(exc)
        if st.session_state.get("market_quote_error"):
            st.error(st.session_state["market_quote_error"])
        quote = st.session_state.get("market_quote")
        if quote:
            st.subheader(f"{quote['symbol']} · {quote['currency']}")
            price, delta, previous = st.columns(3)
            price.metric("Latest available price", f"{quote['price']:,.2f}")
            delta.metric("Change vs prior close", "—" if quote["change"] is None else f"{quote['change']:+,.2f}", "—" if quote["change_percent"] is None else f"{quote['change_percent']:+.2f}%")
            previous.metric("Previous close", "—" if quote["previous_close"] is None else f"{quote['previous_close']:,.2f}")
            st.caption(f"Latest bar: {quote['bar_time']} · Retrieved (UTC): {quote['retrieved_at_utc']} · {quote['interval']}")
            st.caption(f"Source: {quote['source']}. Price availability and timing depend on the provider; verify independently before making decisions.")
    elif search and query.strip():
        st.info("No equity or ETF matches were returned. Try a ticker symbol or a shorter company name.")
