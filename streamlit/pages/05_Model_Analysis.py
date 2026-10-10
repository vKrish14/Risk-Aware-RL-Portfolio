"""Training reward logs and model metadata."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Model Analysis")
root = Path(__file__).resolve().parents[2]
table = root / "results/tables/test_metrics.csv"
if table.exists():
    st.subheader("Held-out strategy comparison")
    st.dataframe(pd.read_csv(table, index_col="strategy"))
for figure in ("risk_return.png", "rolling_sharpe.png"):
    path = root / "results/figures" / figure
    if path.exists():
        st.image(str(path), caption=figure.replace("_", " ").replace(".png", "").title())
for name in ("risk_aware", "return_net_cost", "return_only"):
    metadata_path = root / f"models/checkpoints/{name}/metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.subheader(name.replace("_", " ").title())
        st.write({k: meta.get(k) for k in ("seed", "requested_timesteps", "actual_timesteps", "selected_using", "test_used_for_selection")})
        evaluations = root / f"results/reports/training/{name}/evaluations.npz"
        if evaluations.exists():
            data = dict(__import__("numpy").load(evaluations))
            if "timesteps" in data and "results" in data:
                st.line_chart(pd.Series(data["results"].mean(axis=1), index=data["timesteps"], name="Validation reward"))
if not any((root / f"models/checkpoints/{n}/metadata.json").exists() for n in ("risk_aware", "return_net_cost", "return_only")):
    st.info("Train one or more reward variants to populate model analysis.")
