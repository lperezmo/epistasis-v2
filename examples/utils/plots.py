"""Plotly theming helpers for the epistasis-v2 showcase.

Tailwind teal-indigo palette. One `apply_theme` wrapper that every figure
goes through so the modebar, margins, fonts, and color sequence stay
consistent across pages.
"""

from __future__ import annotations

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


def apply_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply the shared showcase theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        colorway=COLOR_SEQUENCE,
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13, color="#0f172a"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#f8fafc", family="Inter")),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
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
