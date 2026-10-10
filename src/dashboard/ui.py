"""Shared visual theme and plain-language page framing for Streamlit."""
from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Apply small, consistent presentation refinements across every page."""
    st.markdown(
        """
        <style>
        :root {
          --navy: #102a43;
          --ink: #24394b;
          --muted: #64788a;
          --teal: #0f766e;
          --mint: #dff1ec;
          --canvas: #f7f8f4;
          --line: #dce4e5;
          --card: #ffffff;
          --positive: #147a65;
          --negative: #bd4c50;
          --amber: #b47722;
        }
        [data-testid="stAppViewContainer"] { background: var(--canvas); }
        /* Preserve normal document flow below Streamlit's sticky toolbar. */
        header[data-testid="stHeader"] { background: var(--canvas) !important; z-index: 1000; }
        [data-testid="stToolbar"] { background: transparent !important; }
        [data-testid="stAppDeployButton"], [data-testid="stAppDeployButton"] button, button[aria-label="Deploy"] { display: none !important; }
        [data-testid="stMainBlockContainer"] {
          max-width: 1480px;
          padding-top: 3.65rem !important;
          padding-bottom: 2.25rem;
        }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #102a43 0%, #173b50 100%);
          border-right: 1px solid rgba(255,255,255,.08);
        }
        section[data-testid="stSidebar"] * { color: #edf4f8; }
        section[data-testid="stSidebar"] a:hover { color: #8ed3c2 !important; }
        section[data-testid="stSidebar"] [data-testid="stAlert"] { background: #234650; border: 1px solid rgba(145,223,211,.22); }
        section[data-testid="stSidebar"] [data-testid="stAlert"] * { color: #e8f5f4; }
        section[data-testid="stSidebar"] [aria-current="page"] {
          background: rgba(72,202,185,.16);
          border-radius: 9px;
        }
        .brand-lockup { padding: .55rem .3rem 1.2rem; margin-bottom: .7rem; border-bottom: 1px solid rgba(255,255,255,.15); }
        .brand-lockup .brand-mark { color: #8ed3c2; font-size: .73rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
        .brand-lockup .brand-name { color: white; font-size: 1.12rem; font-weight: 750; line-height: 1.2; margin-top: .35rem; }
        .brand-lockup .brand-note { color: #bdcbd8; font-size: .77rem; margin-top: .35rem; }
        [data-testid="stMainBlockContainer"] h1 { color: var(--navy); font-size: clamp(1.9rem, 3vw, 2.65rem); font-weight: 760; letter-spacing: -.03em; line-height: 1.13; margin: .25rem 0 .45rem; overflow-wrap: anywhere; }
        [data-testid="stMainBlockContainer"] h5 { color: var(--teal); font-size: .72rem; font-weight: 800; letter-spacing: .13em; line-height: 1.4; margin: 0 0 .3rem; text-transform: uppercase; }
        [data-testid="stMainBlockContainer"] h2, [data-testid="stMainBlockContainer"] h3 { color: var(--navy); letter-spacing: -.015em; }
        [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] { color: var(--muted); }
        [data-testid="stMetric"] {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 1rem 1rem .9rem;
          box-shadow: 0 3px 14px rgba(16,42,67,.045);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .8rem; }
        [data-testid="stMetricValue"] { color: var(--navy); font-weight: 730; }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--line);
          border-radius: 15px;
          background: var(--card);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        [data-testid="stAlert"] { border-radius: 12px; }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
          background: var(--teal);
          border-color: var(--teal);
          color: white;
          border-radius: 9px;
          font-weight: 650;
        }
        .stButton > button, .stDownloadButton > button { border-radius: 9px; }
        div[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 14px; padding: 1.15rem; background: #fff; }
        div[data-testid="stExpander"] { border-color: var(--line); border-radius: 12px; background: #fff; }
        [data-testid="stPlotlyChart"] { background: transparent; }
        .finance-positive { color: var(--positive); font-weight: 700; }
        .finance-negative { color: var(--negative); font-weight: 700; }
        @media (max-width: 760px) {
          [data-testid="stMainBlockContainer"] { padding: 2.5rem .8rem 1.7rem !important; }
          [data-testid="stMainBlockContainer"] h1 { font-size: 1.85rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(eyebrow: str, title: str, description: str) -> None:
    """Render headings and body text with native Streamlit elements."""
    st.markdown(f"##### {eyebrow.upper()}")
    st.title(title)
    st.write(description)


def section_heading(title: str, note: str | None = None) -> None:
    """Render a small section label and optional one-line description."""
    st.subheader(title)
    if note:
        st.caption(note)
