"""Date-filtered strategy leaderboard and equity comparison."""
from datetime import date

import pandas as pd
import streamlit as st

from src.dashboard.api_client import get_json, post_json
from src.dashboard.components import show_equity, show_metrics, show_weights
from src.dashboard.ui import page_intro, section_heading


page_intro(
    "Explore results",
    "Compare strategies",
    "Rank all saved strategies on the same date range, then inspect the equity curve and allocation history for one strategy.",
)
st.caption("This page reads the saved held-out report; it does not retrain or choose a checkpoint based on test performance.")
strategies = {
    "Experimental tail-risk PPO": "cvar_aware",
    "Risk-aware PPO": "risk_aware",
    "PPO · return net of cost": "return_net_cost",
    "PPO · return only": "return_only",
    "Equal weight": "equal_weight",
    "Buy and hold": "buy_hold",
    "Momentum": "momentum",
    "Minimum variance": "minimum_variance",
}
metric_labels = {
    "total_return": "Total return",
    "annualized_return": "Annualized return",
    "sharpe_ratio": "Sharpe ratio",
    "sortino_ratio": "Sortino ratio",
    "max_drawdown": "Maximum drawdown",
    "annualized_volatility": "Annualized volatility",
    "turnover": "Turnover sum",
    "cumulative_transaction_costs": "Transaction costs",
}
try:
    limits = get_json("/backtest/availability")
except Exception as exc:
    st.error(f"Saved backtest report is unavailable: {exc}")
    st.info("The checked-in report is required. Restore `results/backtests/test_report.json` and restart the API.")
    st.stop()

minimum = date.fromisoformat(limits["minimum"])
maximum = date.fromisoformat(limits["maximum"])
with st.form("comparison_controls"):
    section_heading("1 · Set the comparison")
    period_col, strategy_col, metric_col = st.columns([1, 1, 1])
    start_date = period_col.date_input("From", minimum, min_value=minimum, max_value=maximum)
    end_date = period_col.date_input("Through", maximum, min_value=minimum, max_value=maximum)
    selected_label = strategy_col.selectbox("Show equity for", strategies)
    metric_key = metric_col.selectbox("Rank by", list(metric_labels), format_func=lambda key: metric_labels[key])
    direction = metric_col.radio("Rank direction", ("Highest first", "Lowest first"), horizontal=True)
    submitted = st.form_submit_button("Run comparison", type="primary")

if submitted:
    if start_date > end_date:
        st.error("The start date must be on or before the end date.")
        st.stop()
    try:
        result = post_json("/backtest/run", {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "strategy": strategies[selected_label],
        })
    except Exception as exc:
        st.error(f"Could not compare this date range: {exc}")
        st.stop()

    comparison = pd.DataFrame(result["comparison"]).T
    ascending = direction == "Lowest first"
    ranked = comparison.sort_values(metric_key, ascending=ascending)
    section_heading(f"2 · Leaderboard by {metric_labels[metric_key]}")
    st.dataframe(ranked)
    st.bar_chart(ranked[metric_key].rename(metric_labels[metric_key]))

    section_heading(f"3 · Inspect {selected_label}")
    show_metrics(result["metrics"])
    show_equity(result["equity"])
    show_weights(result["weights"])
    st.download_button(
        "Download comparison as CSV",
        data=ranked.to_csv().encode("utf-8"),
        file_name=f"strategy_comparison_{start_date}_{end_date}.csv",
        mime="text/csv",
    )
else:
    st.info(f"Select the metric, strategy, and dates ({minimum} to {maximum}), then run the comparison.")
