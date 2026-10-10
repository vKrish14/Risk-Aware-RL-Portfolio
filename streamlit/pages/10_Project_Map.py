"""Interactive map of the research workflow and dashboard capabilities."""
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.ui import page_intro, section_heading

page_intro(
    "Navigation · architecture · workflow",
    "Project map: from prices to portfolio research",
    "A guided map of what the project does, how its parts connect, and where to go next. Each stage is inspectable; the system is a historical research tool and never sends trades.",
)

st.info("Follow the data left to right: clean prices become features, features feed a simulated portfolio, and independent tests summarize what happened before the dashboard presents it.")

stages = [
    {"name": "Market data", "module": "src/data", "what": "Downloads or reads cached adjusted prices and daily candles. Checks the dates and values, and records the source.", "page": "Market explorer", "accent": "#0f766e"},
    {"name": "Market features", "module": "src/features", "what": "Builds returns, momentum, volatility, and technical indicators using only information available by each date.", "page": "How the strategy learns", "accent": "#147a65"},
    {"name": "Portfolio simulator", "module": "src/environment", "what": "Lets the policy choose long-only weights, applies market returns, and charges a transparent turnover cost.", "page": "What-if sandbox", "accent": "#b47722"},
    {"name": "PPO training", "module": "src/training · src/agents", "what": "Trains a neural-network policy with Proximal Policy Optimization and selects checkpoints using validation data.", "page": "Model research", "accent": "#426c9a"},
    {"name": "Benchmarks & tests", "module": "src/baselines · src/evaluation", "what": "Compares learned allocations with conventional strategies, then reports return, risk, drawdown, turnover, and costs.", "page": "Strategy scorecard", "accent": "#8b6fa8"},
    {"name": "Prediction service", "module": "api", "what": "FastAPI loads the saved model once and provides health, prediction, simulation, historical comparison, search, and quote endpoints.", "page": "Allocation lab", "accent": "#bd4c50"},
    {"name": "Research dashboard", "module": "streamlit", "what": "Presents the project guide, results, learning material, interactive market charts, stock search, and finance tools.", "page": "This dashboard", "accent": "#173b50"},
]

fig = go.Figure(go.Sankey(
    arrangement="fixed",
    node={
        "label": [stage["name"] for stage in stages],
        "x": [i / (len(stages) - 1) for i in range(len(stages))],
        "y": [0.45] * len(stages),
        "pad": 16,
        "thickness": 22,
        "line": {"color": "#ffffff", "width": 1},
        "color": [stage["accent"] for stage in stages],
        "hovertemplate": "%{label}<extra></extra>",
    },
    link={
        "source": list(range(len(stages) - 1)),
        "target": list(range(1, len(stages))),
        "value": [1] * (len(stages) - 1),
        "color": ["rgba(15,118,110,0.15)"] * (len(stages) - 1),
        "hovertemplate": "%{source.label} → %{target.label}<extra></extra>",
    },
))
fig.update_layout(
    height=210,
    margin={"l": 8, "r": 8, "t": 16, "b": 12},
    paper_bgcolor="#f7f8f4",
    font={"family": "Arial, sans-serif", "size": 12, "color": "#24394b"},
)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
st.caption("The arrows show the main research flow. The API serves the dashboard; the dashboard does not retrain the model.")

section_heading("Explore a stage", "Pick a stage to see its job and the dashboard page that demonstrates it.")
names = [stage["name"] for stage in stages]
selected_name = st.selectbox("Pipeline stage", names)
selected = next(stage for stage in stages if stage["name"] == selected_name)
left, right = st.columns([2, 1])
with left:
    st.markdown(f"### {selected['name']}")
    st.write(selected["what"])
with right:
    st.metric("Code area", selected["module"])
    st.metric("Dashboard destination", selected["page"])

section_heading("What you can do here")
cards = st.columns(3)
capabilities = [
    ("Understand", "Project guide, plain-English model walkthrough, mathematical formulation, and a finance glossary."),
    ("Inspect", "Historical portfolio and benchmark comparisons, training/reward analysis, market candles, and pattern follow-through."),
    ("Experiment", "Search a company, inspect its quote and candle history, request a hypothetical allocation, and change what-if assumptions."),
]
for column, (title, description) in zip(cards, capabilities, strict=True):
    with column, st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(description)

with st.expander("How the model decision is checked"):
    st.write("The training period is used to fit the policy and preprocessing. Validation data is used for checkpoint selection. A later held-out test period is used for the final strategy comparison. Market explorer quotes and candlestick patterns are separate analysis tools; they do not enter the trained portfolio policy.")
