"""Interactive comparison of saved PPO reward objectives and benchmarks."""
from pathlib import Path

import pandas as pd
import streamlit as st
from src.dashboard.ui import page_intro, section_heading


page_intro(
    "Explore results",
    "Reward analysis",
    "Compare four PPO reward objectives and see how return, risk, trading activity, and modeled costs moved together in the held-out run.",
)
root = Path(__file__).resolve().parents[2]
metrics_path = root / "results/tables/test_metrics.csv"
if not metrics_path.exists():
    st.warning("Saved test metrics are missing. Run `python -m src.evaluation.run_evaluation` to generate them.")
    st.stop()

metrics = pd.read_csv(metrics_path, index_col="strategy")
metric_labels = {
    "total_return": "Total return",
    "annualized_return": "Annualized return",
    "annualized_volatility": "Annualized volatility",
    "sharpe_ratio": "Sharpe ratio",
    "sortino_ratio": "Sortino ratio",
    "max_drawdown": "Maximum drawdown",
    "daily_cvar_95_loss": "Daily CVaR (95%)",
    "turnover": "Turnover sum",
    "cumulative_transaction_costs": "Cumulative transaction costs",
}
metric_key = st.selectbox("Rank strategies by", list(metric_labels), format_func=lambda key: metric_labels[key], index=3, help="Choose a metric to reorder the chart. Lower volatility, turnover, and costs are usually preferred; higher return and Sharpe are usually preferred.")
ascending = metric_key in {"annualized_volatility", "daily_cvar_95_loss", "turnover", "cumulative_transaction_costs"}
ranked = metrics.sort_values(metric_key, ascending=ascending)
section_heading(f"Held-out ranking · {metric_labels[metric_key]}")
st.bar_chart(ranked[metric_key].rename(metric_labels[metric_key]))

section_heading("PPO reward variants", "All variants use the same requested training budget; the experimental tail-risk policy changes the input state by one rolling risk measure.")
ppo_names = ["cvar_aware", "risk_aware", "return_net_cost", "return_only"]
ppo = metrics.loc[[name for name in ppo_names if name in metrics.index]]
st.dataframe(ppo.style.format({
    "total_return": "{:.1%}",
    "annualized_return": "{:.1%}",
    "annualized_volatility": "{:.1%}",
    "sharpe_ratio": "{:.2f}",
    "sortino_ratio": "{:.2f}",
    "max_drawdown": "{:.1%}",
    "daily_cvar_95_loss": "{:.2%}",
    "turnover": "{:.1f}",
    "cumulative_transaction_costs": "${:,.0f}",
}, na_rep="—"))

if {"risk_aware", "return_net_cost", "return_only"}.issubset(metrics.index):
    risk = metrics.loc["risk_aware"]
    st.markdown(
        f"In this saved run, risk-aware PPO's Sharpe was **{risk['sharpe_ratio']:.2f}** and its maximum drawdown was **{risk['max_drawdown']:.1%}**. "
        f"It used **{risk['turnover']:.1f}** cumulative one-way turnover and paid **${risk['cumulative_transaction_costs']:,.0f}** in modeled costs. "
        "Compare these values to the two return-focused reward variants above; this is one seed and one test window."
    )

with st.expander("Validation training curves and checkpoint metadata"):
    found = False
    for name in ppo_names:
        metadata_path = root / f"models/checkpoints/{name}/metadata.json"
        evaluations_path = root / f"results/reports/training/{name}/evaluations.npz"
        if metadata_path.exists():
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            st.write({key: metadata.get(key) for key in ("seed", "requested_timesteps", "actual_timesteps", "selected_using", "test_used_for_selection")})
            found = True
        if evaluations_path.exists():
            import numpy as np

            evaluation = np.load(evaluations_path)
            if "timesteps" in evaluation and "results" in evaluation:
                rewards = pd.Series(evaluation["results"].mean(axis=1), index=evaluation["timesteps"], name=name)
                st.line_chart(rewards)
                found = True
    if not found:
        st.info("Training logs and model metadata are local generated artifacts and are not present in this checkout. The saved held-out strategy table above remains available.")

st.caption("Test outcomes were not used for PPO checkpoint selection. Historical results do not predict future performance.")
