"""User-driven what-if simulation for uploaded daily asset-return scenarios."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.api_client import post_json
from src.dashboard.components import show_equity, show_metrics
from src.dashboard.ui import page_intro, section_heading


ASSETS = ("SPY", "QQQ", "GLD", "IEF", "VNQ", "CASH")
RISKY_ASSETS = ASSETS[:-1]


def make_example_returns() -> pd.DataFrame:
    """Create a deterministic, clearly synthetic return series for trying the tool."""
    step = np.arange(63, dtype=float)
    frame = pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=len(step)).strftime("%Y-%m-%d"),
        "SPY": 0.00035 + 0.0060 * np.sin(step / 4.1),
        "QQQ": 0.00045 + 0.0085 * np.sin(step / 4.0 + 0.7),
        "GLD": 0.00018 + 0.0030 * np.sin(step / 6.0 + 1.2),
        "IEF": 0.00007 + 0.0020 * np.sin(step / 5.0 + 2.0),
        "VNQ": 0.00025 + 0.0065 * np.sin(step / 3.4 + 0.4),
        "CASH": (1.02 ** (1 / 252) - 1),
    })
    return frame


page_intro(
    "Interactive tool",
    "What-if portfolio simulator",
    "Test a fixed target allocation against daily returns that you provide. The simulator applies the selected trading cost and shows the resulting hypothetical portfolio path.",
)
st.warning("This tool does not fetch current prices or place orders. Example data is synthetic. Uploaded returns must be decimal returns (for example, 0.01 means +1%).")

section_heading("1 · Choose your input return data", "Prepare one row per date with the six named asset columns. The downloadable sample demonstrates the exact format.")
example = make_example_returns()
st.download_button(
    "Download synthetic CSV template",
    data=example.to_csv(index=False).encode("utf-8"),
    file_name="synthetic_portfolio_returns.csv",
    mime="text/csv",
)
uploaded = st.file_uploader("Upload daily simple returns as CSV", type=("csv",), help="Required columns: date, SPY, QQQ, GLD, IEF, VNQ, CASH. Values are decimals, not percentages.")

returns = None
if uploaded is not None:
    try:
        raw = pd.read_csv(BytesIO(uploaded.getvalue()))
        required = {"date", *ASSETS}
        missing, extra = required - set(raw.columns), set(raw.columns) - required
        if missing or extra:
            raise ValueError(f"Columns must be exactly date plus {', '.join(ASSETS)}. Missing: {sorted(missing)}; extra: {sorted(extra)}.")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        if raw["date"].isna().any():
            raise ValueError("Some dates could not be parsed.")
        if raw["date"].duplicated().any():
            raise ValueError("Each date must appear only once.")
        raw = raw.sort_values("date").set_index("date")
        returns = raw.loc[:, ASSETS].apply(pd.to_numeric, errors="raise").astype(float)
        if len(returns) < 2:
            raise ValueError("Upload at least two rows of daily returns.")
        if len(returns) > 1000:
            raise ValueError("Upload 1,000 rows or fewer per simulation.")
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("Return values must be finite; remove or correct missing cells.")
        if (returns.to_numpy() <= -1).any():
            raise ValueError("Asset returns must be greater than -100%.")
        st.success(f"Loaded {len(returns):,} daily observations from {returns.index.min().date()} through {returns.index.max().date()}.")
        with st.expander("Preview uploaded returns"):
            st.dataframe(returns.head(12).style.format("{:.2%}"))
    except Exception as exc:
        st.error(f"Could not use this CSV: {exc}")

with st.form("scenario_inputs"):
    section_heading("2 · Choose the assumptions", "Set the account value, transaction cost, risk-free rate, and target weights. Any unused allocation is assigned to CASH.")
    initial_capital = st.number_input("Starting value ($)", min_value=1.0, value=100_000.0, step=5_000.0)
    left, right = st.columns(2)
    cost_percent = left.number_input("Trading cost per one-way turnover (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05, format="%.2f")
    risk_free_percent = right.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=50.0, value=2.0, step=0.25, format="%.2f")
    st.caption("Target weights are held constant. The portfolio rebalances to them at each input date, so drift can create turnover and cost.")
    st.caption("The first return row is the starting decision date; the first portfolio return is applied on the next row, following the project's date-t to date-t+1 accounting convention.")
    columns = st.columns(5)
    target_risky_percent = {
        asset: columns[index].slider(f"{asset} target weight (%)", min_value=0, max_value=100, value=20, step=5, format="%d")
        for index, asset in enumerate(RISKY_ASSETS)
    }
    run = st.form_submit_button("Run what-if simulation", type="primary", disabled=returns is None)

if run and returns is not None:
    risky_total = sum(target_risky_percent.values())
    if risky_total > 100:
        st.error(f"Risky-asset weights total {risky_total}%. Reduce them to 100% or less. CASH is the remainder.")
    else:
        target = {asset: value / 100 for asset, value in target_risky_percent.items()}
        target["CASH"] = (100 - risky_total) / 100
        target_rows = {asset: target[asset] for asset in ASSETS}
        steps = [
            {"date": timestamp.date().isoformat(), "returns": row.to_dict(), "target_weights": target_rows}
            for timestamp, row in returns.iterrows()
        ]
        try:
            result = post_json("/portfolio/simulate", {
                "initial_capital": initial_capital,
                "transaction_cost_rate": cost_percent / 100,
                "risk_free_annual": risk_free_percent / 100,
                "steps": steps,
            })
            section_heading("3 · Read or export the result", "Target weights are reset at each input date. The first row is the starting decision date; the first portfolio return is on the next row.")
            show_metrics(result["metrics"])
            show_equity(result["equity"])
            st.write("Target allocation")
            st.bar_chart(pd.Series(result["weights"][next(iter(result["weights"]))], name="Weight"))
            csv = pd.DataFrame({"portfolio_value": result["equity"]}).rename_axis("date").to_csv().encode("utf-8")
            st.download_button("Download simulated portfolio path", csv, "portfolio_scenario.csv", "text/csv")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
elif uploaded is None:
    st.info("Upload a returns CSV to enable the simulator. Use the synthetic template to explore the controls immediately.")
