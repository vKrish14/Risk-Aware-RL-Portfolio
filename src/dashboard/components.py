"""Reusable Streamlit metrics and chart presentation."""
import pandas as pd
import streamlit as st


def show_metrics(metrics: dict) -> None:
    fields = [
        ("Total return", "total_return", ".1%"), ("Annualized return", "annualized_return", ".1%"),
        ("Volatility", "annualized_volatility", ".1%"), ("Sharpe", "sharpe_ratio", ".2f"),
        ("Max drawdown", "max_drawdown", ".1%"), ("Turnover", "turnover", ".1f"),
    ]
    cols = st.columns(3)
    for i, (label, key, fmt) in enumerate(fields):
        value = metrics.get(key)
        cols[i % 3].metric(label, "Unavailable" if value is None else format(value, fmt))


def show_equity(equity: dict) -> None:
    if equity:
        series = pd.Series(equity, dtype=float)
        series.index = pd.to_datetime(series.index)
        st.line_chart(series.rename("Portfolio value"))


def show_weights(weights: dict) -> None:
    if weights:
        st.area_chart(pd.DataFrame.from_dict(weights, orient="index").sort_index())
