"""Plain-language finance glossary and small assumption-driven calculators."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.finance_kit import cagr, future_value
from src.dashboard.ui import page_intro, section_heading


page_intro(
    "Quick reference · plain language",
    "Finance kit",
    "Look up abbreviations used in the project, or try simple growth calculators. Calculations are educational examples, not forecasts or recommendations.",
)

definitions = pd.DataFrame([
    ("ETF", "Exchange-traded fund", "A fund that holds a basket of investments and trades on an exchange. SPY, QQQ, GLD, IEF, and VNQ are ETFs in this project."),
    ("RL", "Reinforcement learning", "A way to train a decision-making program through simulated actions and feedback."),
    ("PPO", "Proximal Policy Optimization", "The reinforcement-learning method used to update the portfolio policy in this project."),
    ("NAV", "Net asset value", "The current value of a portfolio after gains, losses, and modeled costs."),
    ("OHLC", "Open, high, low, close", "The four prices shown by a candlestick for one trading session."),
    ("RSI", "Relative Strength Index", "A momentum indicator scaled from 0 to 100. It summarizes recent price changes; it does not predict future prices."),
    ("MACD", "Moving Average Convergence/Divergence", "A momentum indicator based on the difference between two moving averages."),
    ("CAGR", "Compound annual growth rate", "The steady yearly growth rate that would connect a starting value to an ending value over a chosen period."),
    ("CVaR 95%", "Conditional Value at Risk at 95%", "The average loss among the worst 5% of returns in the chosen sample. Here it summarizes recent historical daily returns."),
    ("VaR 95%", "Value at Risk at 95%", "A loss threshold that was exceeded by roughly 5% of observations in the chosen sample. It is not a maximum-loss guarantee."),
    ("Sharpe ratio", "Excess return per unit of volatility", "A risk-adjusted summary. This project uses a 2% annual reference rate and 252 daily periods."),
    ("Sortino ratio", "Excess return per unit of downside variation", "Like Sharpe, but focuses on returns below the reference rate."),
    ("Drawdown", "Fall from a previous portfolio high", "A 10% drawdown means the portfolio is 10% below its earlier peak."),
    ("Volatility", "Variation in returns", "A statistical measure of how much returns move around their average; it does not say which direction comes next."),
    ("Turnover", "How much the portfolio is reallocated", "This project uses one-way turnover: half the sum of absolute weight changes."),
    ("Annualized", "Scaled to a one-year rate", "A way to express a multi-day or multi-year measurement on a yearly basis. It does not mean the same rate will continue."),
], columns=["Abbreviation / term", "Expanded name", "In everyday language"])

section_heading("Find a term", "Search by abbreviation or plain-English meaning.")
query = st.text_input("Search the finance glossary", placeholder="Try Sharpe, ETF, or drawdown")
if query.strip():
    matches = definitions.apply(lambda column: column.astype(str).str.contains(query, case=False, regex=False)).any(axis=1)
    filtered = definitions.loc[matches]
else:
    filtered = definitions
st.dataframe(filtered, hide_index=True, width="stretch")

section_heading("Compound growth calculator", "Change the assumptions to see how timing and rate affect a hypothetical value.")
st.info("Assumptions: one fixed annual rate, contributions at each year-end, no taxes, and no fees. Real returns vary and may be negative.")
left, right = st.columns([1, 1.25])
with left:
    starting_value = st.number_input("Starting amount ($)", min_value=0.0, value=10_000.0, step=1_000.0)
    annual_contribution = st.number_input("Contribution at each year-end ($)", min_value=0.0, value=2_000.0, step=500.0)
    annual_rate_percent = st.slider("Assumed annual rate (%)", min_value=-100.0, max_value=30.0, value=5.0, step=0.5)
    years = st.slider("Number of years", min_value=1, max_value=40, value=10)

annual_rate = annual_rate_percent / 100
values = [future_value(starting_value, annual_contribution, annual_rate, year) for year in range(years + 1)]
contributed = starting_value + annual_contribution * years
final_value = values[-1]
with right:
    result_cols = st.columns(2)
    result_cols[0].metric("Illustrative ending value", f"${final_value:,.0f}", help="A mathematical scenario under the fixed assumptions entered; it is not an expected or guaranteed result.")
    result_cols[1].metric("Total contributions", f"${contributed:,.0f}")
    chart_data = pd.DataFrame({"Year": range(years + 1), "Illustrative value ($)": values})
    figure = px.line(chart_data, x="Year", y="Illustrative value ($)", markers=True)
    figure.update_layout(height=330, template="plotly_white", margin={"l": 8, "r": 8, "t": 10, "b": 8}, yaxis_title="Value ($)")
    st.plotly_chart(figure, width="stretch")

with st.expander("Calculate an annualized growth rate from two values"):
    first, second, duration = st.columns(3)
    beginning = first.number_input("Beginning value ($)", min_value=1.0, value=10_000.0, step=1_000.0, key="cagr_beginning")
    ending = second.number_input("Ending value ($)", min_value=0.0, value=15_000.0, step=1_000.0, key="cagr_ending")
    elapsed_years = duration.number_input("Elapsed years", min_value=0.1, value=5.0, step=0.5, key="cagr_years")
    st.metric("Annualized growth rate (CAGR)", f"{cagr(beginning, ending, elapsed_years):.2%}", help="A smoothed rate connecting the two values. It hides the path between them and does not predict future growth.")

st.caption("All calculators run locally in the dashboard. No data is sent to a trading service and no trades are placed.")
