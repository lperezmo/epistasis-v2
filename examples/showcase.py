"""Entrypoint for the epistasis-v2 Streamlit showcase.

Run with:
    streamlit run examples/showcase.py

Or with uv (no permanent install):
    uv run --with streamlit --with plotly streamlit run examples/showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="epistasis-v2",
    page_icon=":material/science:",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/lperezmo/epistasis-v2",
        "Report a bug": "https://github.com/lperezmo/epistasis-v2/issues",
        "About": (
            "epistasis-v2: high-order epistasis fits with a Rust hot path. "
            "Clean-break rewrite of harmslab/epistasis."
        ),
    },
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
      header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.html(
    """
    <div style="text-align:center; padding: 0.25rem 0 0.5rem;">
      <h2 style="margin: 0; font-weight: 700; font-size: 2.25rem; letter-spacing: -0.02em;
                 background: linear-gradient(90deg, #0d9488 0%, #6366f1 100%);
                 -webkit-background-clip: text; background-clip: text; color: transparent;">
        epistasis-v2
      </h2>
      <p style="margin: 0.25rem 0 0; color: #64748b; font-size: 0.95rem;">
        High-order epistasis fits with a Rust hot path and a Walsh-Hadamard fast solver.
      </p>
    </div>
    """
)

page = st.navigation(
    {
        "": [
            st.Page(
                "app_pages/intro.py",
                title="Overview",
                icon=":material/hub:",
                default=True,
            ),
        ],
        "Mechanics": [
            st.Page(
                "app_pages/design_matrix.py",
                title="Design matrix",
                icon=":material/grid_on:",
            ),
            st.Page(
                "app_pages/linear_fit.py",
                title="Linear fit",
                icon=":material/scatter_plot:",
            ),
        ],
        "Performance": [
            st.Page(
                "app_pages/fwht_speed.py",
                title="FWHT fast path",
                icon=":material/bolt:",
            ),
        ],
        "Workflows": [
            st.Page(
                "app_pages/simulate.py",
                title="Simulate",
                icon=":material/science:",
            ),
            st.Page(
                "app_pages/cross_validation.py",
                title="Cross validation",
                icon=":material/repeat:",
            ),
        ],
        "Meta": [
            st.Page(
                "app_pages/about.py",
                title="About",
                icon=":material/info:",
            ),
        ],
    },
    position="top",
)
page.run()
