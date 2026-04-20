"""Linear fit page: simulate, fit, verify."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.plots import INDIGO, PLOTLY_CONFIG, ROSE, TEAL, apply_theme


def render() -> None:
    from epistasis.mapping import encoding_to_sites
    from epistasis.matrix import get_model_matrix
    from epistasis.models.linear import EpistasisLinearRegression
    from gpmap import GenotypePhenotypeMap

    st.markdown("### Fit an epistatic model")
    st.caption(
        "Simulate a phenotype surface from known coefficients, corrupt it with "
        "Gaussian noise, fit, and verify coefficients and predictions."
    )

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            L = st.slider("Library size L", 3, 8, 5, key="lf_L")
        with col2:
            order = st.slider("Interaction order", 1, L, min(2, L), key="lf_order")
        with col3:
            noise = st.slider("Noise sigma", 0.0, 1.0, 0.1, 0.01, key="lf_noise")
        with col4:
            seed = st.number_input("Seed", min_value=0, value=42, step=1, key="lf_seed")

        rng = np.random.default_rng(int(seed))
        genotypes = np.array(
            ["".join("A" if b == 0 else "B" for b in bits) for bits in np.ndindex(*(2,) * L)]
        )
        gpm_blank = GenotypePhenotypeMap(
            wildtype="A" * L,
            genotypes=genotypes,
            phenotypes=np.zeros(len(genotypes)),
        )
        sites = encoding_to_sites(order=order, encoding_table=gpm_blank.encoding_table)
        X = get_model_matrix(gpm_blank.binary_packed, sites, model_type="global").astype(np.float64)
        beta_true = rng.normal(0.0, 1.0, size=len(sites))
        y_clean = X @ beta_true
        y_obs = y_clean + rng.normal(0.0, noise, size=len(y_clean))

        gpm = GenotypePhenotypeMap(
            wildtype="A" * L,
            genotypes=genotypes,
            phenotypes=y_obs,
        )
        model = EpistasisLinearRegression(order=order).add_gpm(gpm).fit()
        beta_hat = np.asarray(model.thetas, dtype=np.float64)
        stderr = np.asarray(model.epistasis.stdeviations, dtype=np.float64)
        y_pred = model.predict()
        ss_res = float(np.sum((y_obs - y_pred) ** 2))
        ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmsd = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))

        m1, m2, m3 = st.columns(3)
        m1.metric("R^2", f"{r2:.4f}")
        m2.metric("RMSD", f"{rmsd:.4f}")
        m3.metric("Coefficients", f"{len(sites)}")

    st.markdown("##### Predicted vs observed")
    with st.container(border=True):
        lims = [float(min(y_obs.min(), y_pred.min())), float(max(y_obs.max(), y_pred.max()))]
        pv = go.Figure()
        pv.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line=dict(color="#cbd5e1", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        pv.add_trace(
            go.Scatter(
                x=y_obs,
                y=y_pred,
                mode="markers",
                marker=dict(color=TEAL, size=8, line=dict(color="#ffffff", width=1)),
                text=genotypes,
                hovertemplate="<b>%{text}</b><br>obs=%{x:.3f}<br>pred=%{y:.3f}<extra></extra>",
                name="genotypes",
            )
        )
        pv.update_layout(
            xaxis=dict(title="observed phenotype"),
            yaxis=dict(title="predicted phenotype"),
            showlegend=False,
        )
        apply_theme(pv, height=420)
        st.plotly_chart(pv, width="stretch", config=PLOTLY_CONFIG)

    st.markdown("##### Coefficients")
    st.caption(
        "Fitted coefficients with analytic OLS standard errors. Dashed markers "
        "show the ground-truth coefficients used to simulate the phenotypes."
    )
    with st.container(border=True):
        site_labels = []
        for s in sites:
            if len(s) == 1 and s[0] == 0:
                site_labels.append("1")
            else:
                site_labels.append("*".join(f"x{i}" for i in s))

        if np.all(np.isnan(stderr)):
            err_y = None
        else:
            err_y = dict(type="data", array=stderr, visible=True, color="#94a3b8")

        cf = go.Figure()
        cf.add_trace(
            go.Bar(
                x=site_labels,
                y=beta_hat,
                marker=dict(color=INDIGO, line=dict(color="#ffffff", width=1)),
                error_y=err_y,
                name="fitted",
            )
        )
        cf.add_trace(
            go.Scatter(
                x=site_labels,
                y=beta_true,
                mode="markers",
                marker=dict(
                    color=ROSE,
                    size=10,
                    symbol="diamond-open",
                    line=dict(color=ROSE, width=2),
                ),
                name="ground truth",
            )
        )
        cf.update_layout(
            xaxis=dict(title="interaction site", tickangle=-45),
            yaxis=dict(title="coefficient"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_theme(cf, height=420)
        st.plotly_chart(cf, width="stretch", config=PLOTLY_CONFIG)
        if err_y is None:
            st.caption(
                "Standard errors are undefined when the system is exactly "
                "determined (n == p). Use a truncated order to see finite bars."
            )

    with st.expander("Code", icon=":material/code:"):
        st.code(
            f"""\
from epistasis.models.linear import EpistasisLinearRegression
from gpmap import GenotypePhenotypeMap

gpm = GenotypePhenotypeMap(
    wildtype="A" * {L},
    genotypes=[...],
    phenotypes=y_obs,
)

model = EpistasisLinearRegression(order={order}).add_gpm(gpm).fit()
beta = model.thetas
stderr = model.epistasis.stdeviations
y_pred = model.predict()
""",
            language="python",
        )


render()
