"""Shared Streamlit shell, theme, and descriptive navigation."""
from pathlib import Path
import sys

import streamlit as st

# Streamlit may put this entrypoint's folder, rather than the repository root,
# at the front of sys.path. Keep `src` imports stable in Docker and cloud runs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.api_client import get_json
from src.dashboard.ui import apply_theme


st.set_page_config(
    page_title="Portfolio Research Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

st.sidebar.markdown(
    """
    <div class="brand-lockup">
      <div class="brand-mark">Portfolio research</div>
      <div class="brand-name">Risk-Aware RL Lab</div>
      <div class="brand-note">Learn · compare · simulate</div>
    </div>
    """,
    unsafe_allow_html=True,
)
try:
    health = get_json("/health")
    if health.get("model_ready"):
        active_model = "Experimental tail-risk model" if "cvar_aware" in str(health.get("model", "")) else "PPO allocation model"
        st.sidebar.success(f"{active_model} ready")
    else:
        st.sidebar.info("Saved results ready" if health.get("saved_results_ready") else "Model and saved results unavailable")
except Exception:
    st.sidebar.info("Offline view · saved results may still be available")

pages = {
    "Start here": [
        st.Page("pages/10_Project_Map.py", title="Project map · Start here", icon="🗺️"),
        st.Page("pages/01_Overview.py", title="Project at a glance", icon="🏠", default=True),
        st.Page("pages/07_Learning_Guide.py", title="How the strategy learns", icon="🧠"),
        st.Page("pages/09_Finance_Kit.py", title="Finance quick reference", icon="🧮"),
    ],
    "Explore results": [
        st.Page("pages/08_Market_Charts.py", title="Market explorer", icon="🕯️"),
        st.Page("pages/02_Portfolio.py", title="Portfolio performance", icon="📈"),
        st.Page("pages/04_Backtesting.py", title="Strategy scorecard", icon="📊"),
        st.Page("pages/05_Model_Analysis.py", title="Model research", icon="🔬"),
    ],
    "Interactive tools": [
        st.Page("pages/03_RL_Decision.py", title="Allocation lab", icon="🧭"),
        st.Page("pages/06_Scenario_Simulator.py", title="What-if sandbox", icon="🧪"),
    ],
}
page = st.navigation(pages, position="sidebar")
page.run()
