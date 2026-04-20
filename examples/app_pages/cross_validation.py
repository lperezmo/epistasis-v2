"""K-fold CV page: simulate, fit, score per-fold."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.plots import INDIGO, PLOTLY_CONFIG, TEAL, apply_theme


def render() -> None:
    from epistasis.mapping import encoding_to_sites
    from epistasis.matrix import get_model_matrix
    from epistasis.models.linear import EpistasisLinearRegression
    from epistasis.validate import k_fold
    from gpmap import GenotypePhenotypeMap

    st.markdown("### K-fold cross validation")
    st.caption(
        "Hold out disjoint fractions of the library in turn, re-fit on the "
        "remaining data, score on the held-out phenotypes. Reports per-fold "
        "R^2 and the mean +/- std."
    )

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            L = st.slider("Library size L", 4, 8, 6, key="cv_L")
        with col2:
            order = st.slider("Interaction order", 1, L, min(2, L), key="cv_order")
        with col3:
            noise = st.slider("Noise sigma", 0.0, 1.0, 0.15, 0.01, key="cv_noise")
        with col4:
            k = st.slider("k", 3, 10, 5, key="cv_k")

        seed = st.number_input("Seed", min_value=0, value=0, step=1, key="cv_seed")
        run = st.button("Run CV", type="primary", icon=":material/repeat:", key="cv_run")

    if not run:
        st.info(
            "Choose parameters above and run. Truncated orders make the system "
            "overdetermined so the held-out R^2 is meaningful.",
            icon=":material/info:",
        )
        return

    rng = np.random.default_rng(int(seed))
    genotypes = np.array(
        ["".join("A" if b == 0 else "B" for b in bits) for bits in np.ndindex(*(2,) * L)]
    )
    gpm_blank = GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=np.zeros(len(genotypes)),
    )
    sites_full = encoding_to_sites(order=order, encoding_table=gpm_blank.encoding_table)
    X = get_model_matrix(gpm_blank.binary_packed, sites_full, model_type="global").astype(
        np.float64
    )
    beta = rng.normal(0.0, 1.0, size=len(sites_full))
    y = X @ beta + rng.normal(0.0, noise, size=X.shape[0])
    gpm = GenotypePhenotypeMap(wildtype="A" * L, genotypes=genotypes, phenotypes=y)

    model = EpistasisLinearRegression(order=order)
    with st.spinner("Running k-fold CV...", show_time=True):
        scores = k_fold(gpm, model, k=int(k), rng=np.random.default_rng(int(seed) + 1))
    scores_arr = np.asarray(scores, dtype=np.float64)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean R^2", f"{scores_arr.mean():.4f}")
        c2.metric("Std", f"{scores_arr.std():.4f}")
        c3.metric("Folds", f"{int(k)}")

        bars = go.Figure(
            go.Bar(
                x=[f"Fold {i + 1}" for i in range(len(scores_arr))],
                y=scores_arr,
                marker=dict(color=TEAL, line=dict(color="#ffffff", width=1)),
                text=[f"{s:.3f}" for s in scores_arr],
                textposition="outside",
            )
        )
        bars.add_hline(
            y=float(scores_arr.mean()),
            line=dict(color=INDIGO, width=2, dash="dash"),
            annotation_text=f"mean {scores_arr.mean():.3f}",
            annotation_position="top right",
        )
        bars.update_layout(
            xaxis=dict(title=""),
            yaxis=dict(title="held-out R^2", range=[min(0.0, scores_arr.min() - 0.05), 1.05]),
            showlegend=False,
        )
        apply_theme(bars, height=360)
        st.plotly_chart(bars, width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            f"Order {order} truncation on a 2^{L}-genotype library. Higher "
            "noise or higher orders close to L will pull scores down."
        )

    with st.expander("Code", icon=":material/code:"):
        st.code(
            f"""\
import numpy as np
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.validate import k_fold

model = EpistasisLinearRegression(order={order})
scores = k_fold(gpm, model, k={int(k)}, rng=np.random.default_rng(0))
print("mean R^2:", np.mean(scores))
""",
            language="python",
        )


render()
