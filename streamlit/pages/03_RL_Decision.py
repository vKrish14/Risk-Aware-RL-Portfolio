"""Interactive historical-snapshot PPO allocation inference."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.api_client import post_json
from src.features.feature_engineer import engineer_features
from src.dashboard.ui import page_intro, section_heading


ASSETS = ("SPY", "QQQ", "GLD", "IEF", "VNQ", "CASH")
RISKY_ASSETS = ASSETS[:-1]
root = Path(__file__).resolve().parents[2]
page_intro(
    "Interactive tool",
    "PPO allocation",
    "Ask the trained policy for target weights using the latest saved market features and the portfolio state you enter below.",
)
st.info("The date shown is the latest date in the local data cache. The policy does not fetch today's prices, forecast returns, or place orders.")

try:
    candidates = [root / "models/checkpoints/cvar_aware/metadata.json", root / "models/checkpoints/risk_aware/metadata.json"]
    metadata_path = next(path for path in candidates if path.exists())
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pieces = [
        pd.read_csv(root / f"data/processed/{name}_prices.csv", index_col="date", parse_dates=["date"])
        for name in ("train", "validation", "test")
    ]
    prices = pd.concat(pieces).sort_index()
    features = engineer_features(prices).dropna()
    latest = features.iloc[-1]
except Exception as exc:
    st.error(f"Cannot prepare inference inputs: {exc}")
    st.info("Run the data pipeline and train the risk-aware PPO model first. See the README setup and training commands.")
    st.stop()

st.write(f"**Latest feature date:** {latest.name.date().isoformat()}")
with st.expander("Review input feature snapshot"):
    st.dataframe(latest.rename("Value").to_frame())

with st.form("inference_state"):
    section_heading("1 · Describe the current portfolio", "The first state variables are portfolio context; risky weights are entered as percentages and CASH is calculated as the remainder.")
    capital = st.number_input("Current portfolio value ($)", min_value=1.0, value=100_000.0, step=5_000.0)
    previous_return = st.number_input("Previous daily return (decimal)", min_value=-0.99, max_value=5.0, value=0.0, step=0.005, format="%.4f")
    current_drawdown_percent = st.slider(
        "Current fall from portfolio peak (%)",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        format="%d",
        help="Enter how far the portfolio is below its previous high. For example, enter 8 for an 8% fall.",
    )
    trailing_volatility = st.number_input("Trailing daily volatility (decimal)", min_value=0.0, max_value=5.0, value=0.0, step=0.002, format="%.4f")
    cvar95_loss = st.number_input("Recent worst-5% average daily loss (decimal)", min_value=0.0, max_value=1.0, value=0.0, step=0.002, format="%.4f", help="Only used by the experimental tail-risk-aware policy. Calculate from your last 60 portfolio returns; leave at zero if unknown.")
    st.caption("Set current risky-asset weights below. CASH is the remaining allocation; the five entries must total no more than 100%.")
    columns = st.columns(5)
    current_risky = {
        asset: columns[index].number_input(f"{asset} weight (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0, format="%.0f", key=f"current_{asset}") / 100.0
        for index, asset in enumerate(RISKY_ASSETS)
    }
    submitted = st.form_submit_button("Calculate target weights", type="primary")

if submitted:
    risky_total = sum(current_risky.values())
    if risky_total > 1.000001:
        st.error(f"Current risky weights total {risky_total:.1%}; reduce them to 100% or less.")
    else:
        current_weights = {**current_risky, "CASH": max(0.0, 1.0 - risky_total)}
        payload = {
            "as_of_date": latest.name.date().isoformat(),
            "features": {name: float(latest[name]) for name in latest.index},
            "current_weights": current_weights,
            "portfolio_value": capital,
            "previous_return": previous_return,
            "drawdown": current_drawdown_percent / 100.0,
            "volatility": trailing_volatility,
            "cvar95_loss": cvar95_loss,
        }
        try:
            response = post_json("/portfolio/predict", payload)
            section_heading("2 · Review the model output", "The target weights are long-only and sum to 100% across the five funds and CASH.")
            allocation = pd.Series(response["weights"], name="Target weight").reindex(ASSETS)
            left, right = st.columns([1, 2])
            left.metric("Largest target", allocation.idxmax(), f"{allocation.max():.1%}")
            left.metric("Cash target", f"{allocation['CASH']:.1%}")
            right.bar_chart(allocation)
            st.dataframe(allocation.to_frame().style.format("{:.1%}"))
            st.caption(f"Inference date: {response['as_of_date']}. The policy recommendation depends on the supplied portfolio state and saved model.")
        except Exception as exc:
            st.error(f"PPO inference failed: {exc}")
