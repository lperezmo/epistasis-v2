"""Intro page: genotype-phenotype map primer with a 3D hypercube viz."""

from __future__ import annotations

import itertools as it

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.plots import INDIGO, PLOTLY_CONFIG, TEAL, TEAL_DARK, _is_dark, apply_theme


_NESTED_CORNERS = (
    (0.5, 0.5, 0.5),
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
)


def _hypercube_positions(L: int) -> np.ndarray:
    """Project vertices of an L-cube into 3D.

    L=3 places vertices at the corners of the unit cube. For L>=4, each extra
    bit k selects a distinct nested 3-cube anchored at a different corner of
    the parent cube (see ``_NESTED_CORNERS``) and shrunk by ``s``. Picking
    different corners per k guarantees all 2^L projected vertices are unique.
    """
    if not 3 <= L <= 7:
        raise ValueError(f"L must be between 3 and 7 for the hypercube viz; got {L}.")

    binary = np.array(list(it.product([0, 1], repeat=L)), dtype=float)
    positions = binary[:, :3].copy()

    s = 0.45
    for k in range(3, L):
        bit_k = binary[:, k]
        corner = np.array(_NESTED_CORNERS[k - 3], dtype=float)
        scale_col = np.where(bit_k == 1, s, 1.0)[:, None]
        offset_col = np.where(bit_k == 1, 1.0, 0.0)[:, None] * (corner * (1.0 - s))
        positions = scale_col * positions + offset_col

    return positions


def _hypercube_edges(L: int) -> list[tuple[int, int]]:
    n = 1 << L
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for bit in range(L):
            j = i ^ (1 << bit)
            if i < j:
                edges.append((i, j))
    return edges


def _phenotype_for_model(L: int, model: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    binary = np.array(list(it.product([0, 1], repeat=L)), dtype=float)
    # Encode as +/- 1: wildtype = +1, mutant = -1.
    signs = 1.0 - 2.0 * binary
    if model == "Additive":
        weights = rng.normal(0, 1, size=L)
        return signs @ weights
    if model == "With pairwise epistasis":
        weights = rng.normal(0, 1, size=L)
        pair_weights = rng.normal(0, 0.8, size=(L, L))
        pair_weights = np.triu(pair_weights, k=1)
        out = signs @ weights
        for i in range(L):
            for j in range(i + 1, L):
                out = out + pair_weights[i, j] * signs[:, i] * signs[:, j]
        return out
    # Random
    return rng.standard_normal(1 << L)


def render() -> None:
    st.markdown("### Genotype-phenotype maps, made fast")
    st.caption(
        "A visual primer on sequence space, epistatic coefficients, and why the "
        "Walsh-Hadamard transform is the right tool for biallelic full-order fits."
    )

    st.markdown("##### Sequence space as a hypercube")
    st.markdown(
        "A biallelic L-site library has `2^L` genotypes, one at each vertex of an "
        "L-dimensional cube. Edges connect genotypes that differ by a single "
        "mutation. Phenotype values live at the vertices; epistasis is the "
        "deviation from additive behavior along those edges."
    )

    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            L = st.radio("Dimension", [3, 4, 5, 6, 7], horizontal=True, key="intro_L")
        with col2:
            model = st.selectbox(
                "Phenotype structure",
                ["Additive", "With pairwise epistasis", "Random"],
                index=1,
                key="intro_model",
            )
        with col3:
            seed = st.number_input("Seed", min_value=0, value=7, step=1, key="intro_seed")

        positions = _hypercube_positions(L)
        phenotype = _phenotype_for_model(L, model, int(seed))
        labels = ["".join("A" if b == 0 else "B" for b in g) for g in it.product([0, 1], repeat=L)]

        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        for i, j in _hypercube_edges(L):
            edge_x.extend([positions[i, 0], positions[j, 0], None])
            edge_y.extend([positions[i, 1], positions[j, 1], None])
            edge_z.extend([positions[i, 2], positions[j, 2], None])

        # Tune density-sensitive visuals: fewer pixels per vertex at higher L,
        # and drop the per-vertex labels once they would overlap (L >= 5).
        edge_width = 3 if L <= 4 else 2 if L == 5 else 1
        marker_size = 14 if L <= 4 else 9 if L == 5 else 6 if L == 6 else 4
        marker_mode = "markers+text" if L <= 4 else "markers"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color="#334155" if _is_dark() else "#e2e8f0", width=edge_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode=marker_mode,
                marker=dict(
                    size=marker_size,
                    color=phenotype,
                    colorscale=[
                        [0.0, TEAL],
                        [0.5, "#1e293b" if _is_dark() else "#f8fafc"],
                        [1.0, INDIGO],
                    ],
                    cmin=-float(np.max(np.abs(phenotype))),
                    cmax=float(np.max(np.abs(phenotype))),
                    line=dict(color=TEAL_DARK, width=1),
                    colorbar=dict(
                        title="Phenotype",
                        thickness=12,
                        len=0.7,
                        x=1.02,
                    ),
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=11, color="#94a3b8" if _is_dark() else "#475569"),
                hovertemplate="<b>%{text}</b><br>phenotype=%{marker.color:.3f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
        )
        apply_theme(fig, height=520)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            f"{1 << L} genotypes, {L * (1 << (L - 1))} single-mutation edges. "
            "Color encodes phenotype magnitude; the additive model lives on a "
            "linear gradient, epistasis bends it."
        )

    st.markdown("##### What the package gives you")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(":material/grid_on: **Design matrices**")
        st.caption(
            "Hadamard or local encoding; full parallel construction in Rust. "
            "See the design-matrix page for an interactive anatomy."
        )
    with col_b:
        st.markdown(":material/bolt: **FWHT fast path**")
        st.caption(
            "Full-order biallelic OLS solves in O(n log n) without ever building "
            "the dense matrix. Thousands of times faster at L=12."
        )
    with col_c:
        st.markdown(":material/functions: **Models + validation**")
        st.caption(
            "Linear, regularized, nonlinear, logistic. k-fold and holdout CV. "
            "Bayesian sampling via emcee."
        )

    with st.expander("Quickstart", icon=":material/terminal:"):
        st.code(
            """\
# Install
uv add epistasis-v2

# Fit a full-order biallelic library (FWHT fast path engages automatically)
from epistasis.models.linear import EpistasisLinearRegression
from gpmap import GenotypePhenotypeMap

gpm = GenotypePhenotypeMap(
    wildtype="AAAA",
    genotypes=["AAAA", "AAAB", "AABA", "AABB", "ABAA", "ABAB", "ABBA", "ABBB",
               "BAAA", "BAAB", "BABA", "BABB", "BBAA", "BBAB", "BBBA", "BBBB"],
    phenotypes=[...],
)

model = EpistasisLinearRegression(order=4).add_gpm(gpm).fit()
model.epistasis.data.head()
""",
            language="python",
        )


render()
