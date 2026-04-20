"""Simulation page: build a GPM from a configurable coefficient distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.plots import INDIGO, PLOTLY_CONFIG, TEAL, apply_theme


def render() -> None:
    from epistasis.simulate import simulate_random_linear_gpm

    st.markdown("### Simulate a genotype-phenotype map")
    st.caption(
        "Draw coefficients uniformly from a user-chosen range, build the "
        "phenotype surface, optionally add Gaussian noise."
    )

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            L = st.slider("Library size L", 2, 8, 4, key="sim_L")
        with col2:
            order = st.slider("Interaction order", 1, L, min(2, L), key="sim_order")
        with col3:
            seed = st.number_input("Seed", min_value=0, value=0, step=1, key="sim_seed")

        col4, col5, col6 = st.columns(3)
        with col4:
            coef_low = st.number_input("Coefficient min", value=-1.0, step=0.1, key="sim_coef_low")
        with col5:
            coef_high = st.number_input("Coefficient max", value=1.0, step=0.1, key="sim_coef_high")
        with col6:
            noise = st.slider("Noise sigma", 0.0, 1.0, 0.0, 0.01, key="sim_noise")

        if coef_high <= coef_low:
            st.warning("Coefficient max must be greater than min.", icon=":material/warning:")
            return

        mutations = {i: ["A", "B"] for i in range(L)}
        gpm, _sites, _coefs = simulate_random_linear_gpm(
            wildtype="A" * L,
            mutations=mutations,
            order=order,
            coefficient_range=(coef_low, coef_high),
            model_type="global",
            rng=np.random.default_rng(int(seed)),
        )
        phenotypes = np.asarray(gpm.phenotypes, dtype=np.float64)
        if noise > 0:
            phenotypes = phenotypes + np.random.default_rng(int(seed) + 1).normal(
                0.0, noise, size=phenotypes.shape
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genotypes", f"{len(phenotypes)}")
        m2.metric("Phenotype mean", f"{phenotypes.mean():.3f}")
        m3.metric("Phenotype std", f"{phenotypes.std():.3f}")
        m4.metric("Range", f"{phenotypes.max() - phenotypes.min():.3f}")

    st.markdown("##### Phenotype distribution")
    with st.container(border=True):
        hist = go.Figure(
            go.Histogram(
                x=phenotypes,
                marker=dict(color=TEAL, line=dict(color="#ffffff", width=1)),
                nbinsx=min(30, max(8, len(phenotypes) // 4)),
                hovertemplate="bin=%{x}<br>count=%{y}<extra></extra>",
            )
        )
        hist.add_vline(
            x=float(phenotypes.mean()),
            line=dict(color=INDIGO, width=2, dash="dash"),
            annotation_text="mean",
            annotation_position="top right",
        )
        hist.update_layout(
            xaxis=dict(title="phenotype"),
            yaxis=dict(title="count"),
            showlegend=False,
        )
        apply_theme(hist, height=360)
        st.plotly_chart(hist, width="stretch", config=PLOTLY_CONFIG)

    st.markdown("##### Genotype preview")
    with st.container(border=True):
        preview = pd.DataFrame(
            {
                "genotype": np.asarray(gpm.genotypes),
                "phenotype": phenotypes,
            }
        )
        head_n = min(24, len(preview))
        st.dataframe(preview.head(head_n), width="stretch", hide_index=True)
        if len(preview) > head_n:
            st.caption(f"Showing {head_n} of {len(preview)} genotypes.")

    with st.expander("Code", icon=":material/code:"):
        st.code(
            f"""\
import numpy as np
from epistasis.simulate import simulate_random_linear_gpm

mutations = {{i: ["A", "B"] for i in range({L})}}
gpm, sites, coefs = simulate_random_linear_gpm(
    wildtype="A" * {L},
    mutations=mutations,
    order={order},
    coefficient_range=({coef_low}, {coef_high}),
    model_type="global",
    rng=np.random.default_rng({int(seed)}),
)
""",
            language="python",
        )


render()
