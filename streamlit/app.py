"""Educational overview dashboard for the local risk-aware PPO project."""
import streamlit as st

from src.dashboard.api_client import get_json
from src.dashboard.components import show_metrics


st.set_page_config(page_title="Risk-Aware RL Portfolio", layout="wide")
st.title("Risk-Aware RL Portfolio Management")
st.caption("Educational research project. Historical results are not a forecast or investment advice.")
try:
    health = get_json("/health")
    st.metric("PPO model", "Ready" if health.get("model_ready") else "Unavailable")
    if health.get("model_ready"):
        period = get_json("/backtest/availability")
        st.write(f"Held-out evaluation window: {period['minimum']} to {period['maximum']}")
        metrics = get_json("/portfolio/metrics")
        st.subheader("Held-out test summary")
        show_metrics(metrics)
        st.caption("Metrics summarize saved historical test data and include the configured transaction-cost convention.")
except Exception as exc:
    st.error(f"Could not reach the local API: {exc}")
