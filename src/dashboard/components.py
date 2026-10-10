"""Reusable Streamlit metrics and chart presentation."""
import pandas as pd
import streamlit as st


def show_metrics(metrics: dict) -> None:
    """Display a consistent set of portfolio measures with plain-language help."""
    fields = [
        ("Total return", "total_return", ".1%", "Net growth over the selected period, after modeled trading costs."),
        ("Annualized return", "annualized_return", ".1%", "Geometric return scaled to a one-year rate."),
        ("Annualized volatility", "annualized_volatility", ".1%", "Annualized standard deviation of daily net returns; higher means more variable returns."),
        ("Sharpe ratio", "sharpe_ratio", ".2f", "Annualized excess return divided by return volatility, using the configured risk-free rate."),
        ("Sortino ratio", "sortino_ratio", ".2f", "Excess return divided by downside deviation; focuses on below-target variation."),
        ("Maximum drawdown", "max_drawdown", ".1%", "Largest peak-to-trough decline in the selected portfolio path."),
        ("Daily CVaR (95%)", "daily_cvar_95_loss", ".2%", "Average daily loss among the worst 5% of observed daily returns; a historical tail-risk summary."),
        ("Calmar ratio", "calmar_ratio", ".2f", "Annualized return divided by the absolute maximum drawdown."),
        ("One-way turnover", "turnover", ".1f", "Sum of daily half-L1 target-weight changes. This is a turnover total, not a percent of account value."),
        ("Trading costs", "cumulative_transaction_costs", ",.0f", "Cumulative dollars deducted under the configured proportional turnover-cost assumption."),
    ]
    for start in range(0, len(fields), 3):
        cols = st.columns(3)
        for i, (label, key, fmt, help_text) in enumerate(fields[start : start + 3]):
            value = metrics.get(key)
            display = "Not available" if value is None else format(value, fmt)
            if value is not None and key == "cumulative_transaction_costs":
                display = f"${display}"
            cols[i].metric(label, display, help=help_text)


def show_equity(equity: dict) -> None:
    if equity:
        series = pd.Series(equity, dtype=float)
        series.index = pd.to_datetime(series.index)
        st.line_chart(series.sort_index().rename("Portfolio value ($)"), y_label="Portfolio value ($)")


def show_weights(weights: dict) -> None:
    if weights:
        frame = pd.DataFrame.from_dict(weights, orient="index").sort_index()
        frame.index = pd.to_datetime(frame.index)
        st.area_chart(frame, y_label="Portfolio weight")
