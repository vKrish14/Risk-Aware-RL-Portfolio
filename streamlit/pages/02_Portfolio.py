"""Saved portfolio-level metrics and test equity curve."""
import streamlit as st
from datetime import date

from src.dashboard.api_client import post_json
from src.dashboard.components import show_equity, show_metrics, show_weights

st.title("Portfolio")
try:
    end = date.today()
    payload = post_json("/backtest/run", {"start_date": "2024-01-02", "end_date": end.isoformat()})
    show_metrics(payload["metrics"])
    show_equity(payload["equity"])
    show_weights(payload["weights"])
    st.caption("Historical test portfolio path; the figure shows holdings after returns and drift.")
except Exception as exc:
    st.error(f"Portfolio results unavailable: {exc}")
