"""Date-filtered held-out test equity curve."""
from datetime import date

import streamlit as st

from src.dashboard.api_client import get_json, post_json
from src.dashboard.components import show_equity, show_metrics

st.title("Backtesting")
try:
    available = get_json("/backtest/availability")
    earliest, latest = date.fromisoformat(available["minimum"]), date.fromisoformat(available["maximum"])
    left, right = st.columns(2)
    start = left.date_input("Start", value=earliest, min_value=earliest, max_value=latest)
    end = right.date_input("End", value=latest, min_value=earliest, max_value=latest)
    result = post_json("/backtest/run", {"start_date": start.isoformat(), "end_date": end.isoformat()})
    show_metrics(result["metrics"])
    st.subheader("Strategy comparison for the selected test window")
    st.dataframe(result["comparison"])
    show_equity(result["equity"])
except Exception as exc:
    st.error(f"Saved test backtest is unavailable for that range: {exc}")
