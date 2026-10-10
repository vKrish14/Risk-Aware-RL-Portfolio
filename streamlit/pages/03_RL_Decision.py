"""Current historical snapshot PPO allocation output."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.api_client import post_json

st.title("RL Decision")
st.caption("Recommendation is a hypothetical allocation for the latest saved market observation.")
try:
    root = Path(__file__).resolve().parents[2]
    meta = json.loads((root / "models/checkpoints/risk_aware/metadata.json").read_text(encoding="utf-8"))
    scaler = meta["preprocessing"]
    raw_parts = [pd.read_csv(root / f"data/processed/{s}_prices.csv", index_col="date", parse_dates=["date"]) for s in ("train", "validation", "test")]
    prices = pd.concat(raw_parts).sort_index()
    from src.features.feature_engineer import engineer_features
    latest = engineer_features(prices).iloc[-1]
    st.subheader("Latest available feature snapshot")
    st.dataframe(latest.rename("value").to_frame())
    response = post_json("/portfolio/predict", {"as_of_date": latest.name.date().isoformat(), "features": latest.to_dict()})
    st.write(f"Data date: **{response['as_of_date']}**")
    st.bar_chart(pd.Series(response["weights"]).sort_values(ascending=False))
    st.caption("The model maps a historical feature snapshot to weights; it does not predict guaranteed profits.")
except Exception as exc:
    st.error(f"Could not produce a model decision: {exc}")
