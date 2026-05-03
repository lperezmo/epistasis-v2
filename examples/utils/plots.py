"""Plotly theming helpers for the epistasis-v2 showcase.

Tailwind teal-indigo palette. One `apply_theme` wrapper that every figure
goes through so the modebar, margins, fonts, and color sequence stay
consistent across pages. Charts adapt to the active Streamlit theme
(light or dark) via `_is_dark()`.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

TEAL = "#14b8a6"
TEAL_DARK = "#0d9488"
INDIGO = "#6366f1"
INDIGO_DARK = "#4f46e5"
AMBER = "#f59e0b"
ROSE = "#f43f5e"
EMERALD = "#10b981"
VIOLET = "#8b5cf6"
SLATE = "#64748b"

COLOR_SEQUENCE = [TEAL, INDIGO, AMBER, ROSE, EMERALD, VIOLET, SLATE]

HEATMAP_SIGNED = [
    [0.0, "#f43f5e"],
    [0.5, "#f8fafc"],
    [1.0, "#14b8a6"],
]

HEATMAP_MAGNITUDE = [
    [0.0, "#f8fafc"],
    [0.5, "#14b8a6"],
    [1.0, "#4f46e5"],
]


def _is_dark() -> bool:
    try:
        import streamlit as st
        return st.context.theme.type == "dark"
    except Exception:
        return False


def apply_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply the shared showcase theme, adapting to the active dark/light mode."""
    dark = _is_dark()
    font_color = "#f1f5f9" if dark else "#0f172a"
    grid_color = "#334155" if dark else "#e2e8f0"
    zeroline_color = "#475569" if dark else "#cbd5e1"
    template = "plotly_dark" if dark else "plotly_white"

    fig.update_layout(
        template=template,
        margin=dict(l=40, r=20, t=40, b=40),
        colorway=COLOR_SEQUENCE,
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13, color=font_color),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=grid_color,
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#f8fafc", family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=zeroline_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=zeroline_color)
    return fig


def interaction_site_grid(
    sites: list[tuple[int, ...]],
    beta_hat: np.ndarray,
    *,
    L: int,
) -> go.Figure:
    """Participation grid: rows = sites x1..xL, cols = interaction terms.

    Each cell is coloured by |coefficient| when the site participates in the
    term; cells where the site does not participate are transparent.
    """
    dark = _is_dark()
    row_labels = [f"x{i}" for i in range(1, L + 1)]
    col_labels: list[str] = []
    for s in sites:
        if len(s) == 1 and s[0] == 0:
            col_labels.append("1")
        else:
            col_labels.append("*".join(f"x{i}" for i in s))

    abs_beta = np.abs(np.asarray(beta_hat, dtype=float))
    n_rows = L
    n_cols = len(sites)
    z = np.full((n_rows, n_cols), float("nan"))
    for j, (s, ab) in enumerate(zip(sites, abs_beta, strict=False)):
        for site_idx in s:
            if site_idx == 0:
                continue
            row = site_idx - 1
            if row < n_rows:
                z[row, j] = ab

    colorscale = (
        [[0.0, "rgba(129,140,248,0.2)"], [1.0, "#818cf8"]]
        if dark
        else [[0.0, "rgba(99,102,241,0.15)"], [1.0, "#4338ca"]]
    )
    height = max(200, n_rows * 44 + 120)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="|coef|", thickness=14),
            hovertemplate="<b>%{y}</b> in <b>%{x}</b><br>|coef|=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(side="top", tickangle=-45, title=""),
        yaxis=dict(title="site", autorange="reversed"),
        height=height,
        margin=dict(l=60, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


PLOTLY_CONFIG: dict[str, object] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "toggleSpikelines",
    ],
}
