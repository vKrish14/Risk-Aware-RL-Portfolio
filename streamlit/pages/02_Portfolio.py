"""Date-selectable portfolio history and benchmark comparison."""
from datetime import date

import pandas as pd
import streamlit as st

from src.dashboard.api_client import get_json, post_json
from src.dashboard.components import show_equity, show_metrics, show_weights
from src.dashboard.ui import page_intro, section_heading


page_intro(
    "Explore results",
    "Portfolio history",
    "Select one strategy and one date window to inspect its saved equity path, portfolio weights, and metrics.",
)
st.caption("These are historical backtest results after the project's modeled transaction costs. No live prices or trades are used here.")
try:
    available = get_json("/backtest/availability")
except Exception as exc:
    st.error(f"Historical portfolio is unavailable: {exc}")
    st.info("Start the API and ensure a trained model, processed prices, and saved test report are available.")
    st.stop()

minimum = date.fromisoformat(available["minimum"])
maximum = date.fromisoformat(available["maximum"])
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
with st.form("portfolio_period"):
    section_heading("1 · Choose the comparison")
    strategy_label = st.selectbox("Strategy", strategies)
    first, last = st.columns(2)
    start_date = first.date_input("From", value=minimum, min_value=minimum, max_value=maximum)
    end_date = last.date_input("Through", value=maximum, min_value=minimum, max_value=maximum)
    submitted = st.form_submit_button("Show portfolio history", type="primary")

if submitted:
    if start_date > end_date:
        st.error("The start date must be on or before the end date.")
    else:
        try:
            result = post_json("/backtest/run", {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "strategy": strategies[strategy_label],
            })
            section_heading(strategy_label)
            show_metrics(result["metrics"])
            show_equity(result["equity"])
            st.caption("Saved portfolio value after modeled daily transaction costs. Holdings shown below are post-return, drifted weights.")
            show_weights(result["weights"])
            comparison = pd.DataFrame(result["comparison"]).T
            if not comparison.empty:
                st.subheader("Benchmarks for the same date range")
                st.dataframe(comparison.style.format({
                    "total_return": "{:.1%}",
                    "annualized_return": "{:.1%}",
                    "annualized_volatility": "{:.1%}",
                    "sharpe_ratio": "{:.2f}",
                    "max_drawdown": "{:.1%}",
                    "cumulative_transaction_costs": "${:,.0f}",
                }, na_rep="—"))
        except Exception as exc:
            st.error(f"Could not load the selected period: {exc}")
else:
    st.info(f"Choose a period between {minimum} and {maximum}, then load the historical results.")
