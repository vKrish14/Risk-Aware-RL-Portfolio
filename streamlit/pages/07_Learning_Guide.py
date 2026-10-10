"""Explain the data, RL setup, equations, and validation in accessible terms."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.ui import page_intro, section_heading


root = Path(__file__).resolve().parents[2]
page_intro("Model guide · no jargon assumed", "How the model learns", "Follow the project from market data to a checked historical result. The equations are optional; each has a plain-English explanation.")

section_heading("The full project pipeline")
steps = st.columns(4)
for col, title, body in zip(steps, ("1 · Prices", "2 · Signals", "3 · Practice", "4 · Check"), (
    "Yahoo Finance daily adjusted prices (SPY, QQQ, GLD, IEF, VNQ). Synthetic cash earns a configurable 2% annual rate.",
    "Pandas builds returns, momentum, volatility, RSI, MACD, and moving averages. Rolling calculations use only current and earlier dates.",
    "A Gymnasium simulator applies weights, next-day returns, drift, and 0.1% × one-way turnover costs. Stable-Baselines3 trains PPO.",
    "Choose checkpoints on 2022–2023 validation data. Freeze the model, then compare once on 2024 onward test data against four rules.",
)):
    with col:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(body)
st.caption("Training: 2015–2021 · Validation: 2022–2023 · Held-out test: 2024-01-02–2026-10-09. Date ranges are chronological; the test period is not used for model selection.")

section_heading("Technology map", "What each tool does in this project")
st.dataframe(pd.DataFrame([
    ("Python 3.13 locally", "Runs the research code; the repository targets a modern Python environment."),
    ("yfinance", "Downloads daily Yahoo Finance prices; locally cached data supports repeatable reruns."),
    ("pandas + NumPy", "Align dates, compute features, and perform transparent portfolio arithmetic."),
    ("Gymnasium", "Defines the simulator interface: observation, action, reward, and next step."),
    ("Stable-Baselines3 PPO + PyTorch", "Trains the neural-network policy and value estimator with PPO."),
    ("FastAPI + Pydantic", "Validates requests and serves model or saved historical results locally."),
    ("Streamlit + Plotly", "Provides these interactive explanations, comparisons, and charts."),
], columns=["Technology", "Role here"]), hide_index=True, width="stretch")

section_heading("What PPO does in this portfolio")
st.write("At each date, the policy receives a **state**: recent market signals, current holdings, account value, the previous return, current drawdown, and recent volatility. Its **action** is six numbers called logits. A softmax turns them into a valid portfolio mix. The simulator applies a cost, then the next period’s return. A **reward** summarizes whether that outcome was desirable under the chosen objective.")
st.latex(r"w_{i,t}=\frac{e^{a_{i,t}}}{\sum_{j=1}^{6}e^{a_{j,t}}},\qquad w_{i,t}\ge 0,\qquad \sum_i w_{i,t}=1")
st.caption("The softmax formula makes every allocation nonnegative and makes all six weights total 100%.")
st.latex(r"\tau_t=\frac{1}{2}\sum_i|w_{i,t}-w^{\mathrm{drift}}_{i,t}|,\qquad c_t=0.001\,\tau_t")
st.caption("Turnover is half the total absolute change in weights, so selling 10% of one holding and buying 10% of another is 10% one-way turnover. The modeled charge is 0.1% of that amount.")
st.latex(r"V_{t+1}=V_t(1-c_t)\left(1+\sum_i w_{i,t}r_{i,t+1}\right)")
st.caption("Portfolio value after trading cost and the next day’s weighted asset return. The policy chooses using date t; it earns returns from t to t+1.")

with st.expander("Optional: PPO's clipped learning objective"):
    st.write("PPO compares how likely an action is under the updated policy versus the previous policy. It uses an estimate of whether the action was better or worse than expected (the advantage). Clipping limits the influence of overly large policy changes.")
    st.latex(r"L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(q_t(\theta)\hat A_t,\;\operatorname{clip}(q_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right]")
    st.latex(r"q_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}")
    st.caption("Here, θ are neural-network parameters; π is the policy's probability of choosing an action; Â is estimated improvement; ε is the clip range (0.2 in the starting configuration). This is a simplified presentation of PPO's policy objective.")

section_heading("How was it validated?")
st.markdown("1. **Data checks:** duplicate or invalid prices are rejected; asset calendars are aligned on common valid observations.\n2. **Leakage controls:** rolling indicators are backward-looking; scaling is fitted on training dates only; date splits are chronological.\n3. **Training and selection:** PPO learns only from training dates. A periodic deterministic evaluation on 2022–2023 selects the saved best checkpoint.\n4. **Held-out comparison:** once selected, policies and equal-weight, buy-and-hold, momentum, and minimum-variance baselines use the same test dates, returns, and transaction-cost convention.\n5. **Interpretation:** this is one seed and one market window. It is a software demonstration, not statistical proof, a forecast, or evidence of future performance.")
metrics_path = root / "results/tables/test_metrics.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path, index_col="strategy")
    if "risk_aware" in metrics.index:
        st.markdown("**Saved risk-aware test measurements:**")
        st.dataframe(metrics.loc[["risk_aware"], ["total_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown", "daily_cvar_95_loss", "calmar_ratio", "turnover", "cumulative_transaction_costs"]].T.rename(columns={"risk_aware": "Measured value"}), width="stretch")

section_heading("The reward is a design choice")
st.markdown("- **Return only:** rewards gross portfolio return.\n- **Return after costs:** rewards the realized net return.\n- **Risk-aware:** starts from gross return and subtracts the explicit modeled trading-cost, trailing-volatility, and drawdown penalties. Portfolio accounting still deducts costs once from wealth; the extra reward cost is an incentive, not a second NAV charge.")
st.caption("The project-specific combination is an educational experiment, not a claim of a new general-purpose RL algorithm. A convincing research contribution requires multiple seeds, alternative periods, ablations, and statistical uncertainty analysis.")
st.markdown("**Experimental tail-risk PPO:** the policy also observes the empirical average loss in the worst 5% of its last 60 daily portfolio returns. Its reward adds a penalty when this estimated tail loss exceeds 2% in a day. This is a small, inspectable project extension; it is not proof of novelty or robustness. Its test results are shown only after training and use the same frozen held-out dates as the existing strategies.")

with st.expander("Glossary"):
    st.markdown("**Agent:** the decision-making learner. **Policy:** its rule for turning information into weights. **State:** the information available at a decision time. **Reward:** the score used for learning. **Drawdown:** decline from the account’s previous high. **Volatility:** variability of returns. **Sharpe ratio:** excess return divided by volatility. **Validation:** data used to select settings. **Test:** later, held-out data used for the final comparison.")

st.markdown("**Primary references:** [PPO paper by Schulman et al.](https://arxiv.org/abs/1707.06347) · [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) · [Gymnasium environment API](https://gymnasium.farama.org/api/env/)")
