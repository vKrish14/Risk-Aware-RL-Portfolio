"""Beginner-friendly project overview and honest held-out summary."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.components import show_metrics
from src.dashboard.ui import page_intro, section_heading


root = Path(__file__).resolve().parents[2]
page_intro(
    "Start here · portfolio research",
    "A portfolio model you can understand",
    "This project teaches a computer to change a six-part portfolio using past market data. Explore what it learned, how it was checked, and where it did not beat simple alternatives.",
)
st.info("For learning and research only. Historical results are not promises. Allocation suggestions are hypothetical; this app does not place trades and is not investment advice.")

section_heading("In one minute", "No finance or machine-learning background required.")
cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        st.markdown("#### 1 · Look at the past")
        st.write("The program reads daily adjusted prices for five exchange-traded funds and a modeled cash asset. It calculates signals such as recent returns and volatility using information available up to that date.")
with cols[1]:
    with st.container(border=True):
        st.markdown("#### 2 · Choose a mix")
        st.write("A neural network suggests percentages for SPY, QQQ, GLD, IEF, VNQ, and cash. The percentages cannot be negative and always add up to 100%.")
with cols[2]:
    with st.container(border=True):
        st.markdown("#### 3 · Learn from a simulation")
        st.write("The simulated portfolio earns the next day’s return and pays a small modeled cost when its mix changes. The learning algorithm adjusts its choices to improve the chosen reward.")

section_heading("What does ‘reinforcement learning’ mean here?")
st.write("Reinforcement learning (RL) is learning by trial and feedback. Imagine a student choosing a portfolio, seeing what happened next, and receiving points for the outcome. The student is the **agent**, the information it sees is the **state**, its chosen percentages are the **action**, and the points are the **reward**. Here, each trial happens inside a historical-data simulator—not in a brokerage account.")
st.markdown("**PPO** means *Proximal Policy Optimization*. It is the learning method used to update the model. ‘Proximal’ means each update is limited so the strategy does not change too drastically in one training step. The policy itself is a small neural network (an `MlpPolicy`) that maps market and portfolio information to allocation choices.")

metrics_path = root / "results/tables/test_metrics.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path, index_col="strategy")
    if "risk_aware" in metrics.index:
        section_heading("A real held-out result", "Test data covers 2 Jan 2024–9 Oct 2026 in the saved run. This period was not used to choose the checkpoint.")
        show_metrics(metrics.loc["risk_aware"].to_dict())
        comparison = metrics.loc[[name for name in ("cvar_aware", "risk_aware", "equal_weight", "buy_hold", "momentum", "minimum_variance") if name in metrics.index], ["total_return", "sharpe_ratio", "max_drawdown", "daily_cvar_95_loss", "cumulative_transaction_costs"]].rename(index={"cvar_aware": "Experimental tail-risk PPO", "risk_aware": "Risk-aware PPO"}, columns={
            "total_return": "Total return", "sharpe_ratio": "Sharpe ratio", "max_drawdown": "Worst peak-to-trough fall", "cumulative_transaction_costs": "Modeled trading costs ($)",
            "daily_cvar_95_loss": "Daily tail loss (95% CVaR)",
        })
        st.dataframe(comparison.style.format({"Total return": "{:.1%}", "Sharpe ratio": "{:.2f}", "Worst peak-to-trough fall": "{:.1%}", "Daily tail loss (95% CVaR)": "{:.2%}", "Modeled trading costs ($)": "${:,.0f}"}), width="stretch")
        st.warning("In this one test period, the risk-aware model did not beat every simple strategy. That is a useful research result: a learned model is not automatically better.")
else:
    st.warning("No saved results found. Generate a report with `python -m src.evaluation.run_evaluation` after training.")

section_heading("What should I open next?")
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("#### Learn the mechanics")
        st.write("Open **Project map · Start here** for the complete system flow, or **How the strategy learns** for the equations and beginner-friendly glossary.")
with right:
    with st.container(border=True):
        st.markdown("#### Explore and try it")
        st.write("Open **Market explorer** for searched stocks and date-ranged candles, or **What-if sandbox** to see how hypothetical results respond to costs.")

with st.expander("How to interpret common numbers"):
    st.markdown("- **Total return:** overall change in portfolio value, after the modeled costs.\n- **Annualized return:** the average yearly pace implied by the full period; it is not a forecast.\n- **Volatility:** how widely daily returns varied.\n- **Sharpe ratio:** average excess return relative to volatility; the calculation uses a 2% annual reference rate and 252 trading days.\n- **Maximum drawdown:** the largest peak-to-trough decline.\n- **Turnover:** one-way fraction of the account reallocated. The cost assumption is 0.1% per turnover unit.")
