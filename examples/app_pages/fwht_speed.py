"""FWHT fast path: time dense OLS against the Walsh-Hadamard path."""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.plots import INDIGO, PLOTLY_CONFIG, ROSE, TEAL, apply_theme


def _time_call(fn, *args, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def render() -> None:
    from epistasis.fast import fwht_ols_coefficients
    from epistasis.mapping import encoding_to_sites
    from epistasis.matrix import get_model_matrix
    from gpmap import GenotypePhenotypeMap

    st.markdown("### Walsh-Hadamard fast path")
    st.caption(
        "Full-order biallelic OLS has $X^{\\top} X = 2^{L} \\, I$. "
        "Solving via the Fast Walsh-Hadamard Transform turns an $O(n^{3})$ dense "
        "solve into an $O(n \\log n)$ transform."
    )

    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            L = st.slider(
                "Size exponent L",
                min_value=4,
                max_value=12,
                value=10,
                key="fw_L",
                help=(
                    "Library has 2^L genotypes at full order. Capped at 12 "
                    "because dense lstsq at L=13 takes minutes."
                ),
            )
        with col2:
            repeat = st.number_input(
                "Repeat", min_value=1, max_value=5, value=3, step=1, key="fw_repeat"
            )
        st.caption(
            f"This benchmark builds a full 2^{L} = {1 << L} genotype library, "
            "then times both solvers. Timings are best-of-N wall clock."
        )
        run = st.button("Run benchmark", type="primary", icon=":material/bolt:", key="fw_run")

    if not run:
        st.info(
            "Press the benchmark button above. At L=12 dense lstsq takes a few "
            "seconds; at L=10 it is already visibly slower than FWHT.",
            icon=":material/info:",
        )
        return

    rng = np.random.default_rng(0)
    genotypes = np.array(
        ["".join("A" if b == 0 else "B" for b in bits) for bits in np.ndindex(*(2,) * L)]
    )
    gpm = GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=rng.standard_normal(1 << L),
    )
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)

    with st.spinner("Building 2^L x 2^L design matrix...", show_time=True):
        X = get_model_matrix(gpm.binary_packed, sites, model_type="global").astype(np.float64)
        y = np.asarray(gpm.phenotypes, dtype=np.float64)

    with st.spinner("Timing dense OLS (numpy.linalg.lstsq)...", show_time=True):
        t_dense = _time_call(lambda: np.linalg.lstsq(X, y, rcond=None), repeat=int(repeat))

    with st.spinner("Timing FWHT fast path...", show_time=True):
        t_fwht = _time_call(
            lambda: fwht_ols_coefficients(gpm.binary_packed, y, sites, "global"),
            repeat=int(repeat),
        )

    speedup = t_dense / t_fwht if t_fwht > 0 else float("inf")

    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Dense lstsq", f"{t_dense * 1000:.2f} ms")
        col_b.metric("FWHT fast path", f"{t_fwht * 1000:.3f} ms")
        col_c.metric("Speedup", f"{speedup:,.1f}x")

        bars = go.Figure(
            go.Bar(
                x=["Dense OLS (numpy.linalg.lstsq)", "FWHT (epistasis.fast)"],
                y=[t_dense * 1000, t_fwht * 1000],
                marker=dict(color=[ROSE, TEAL], line=dict(color="#ffffff", width=1)),
                text=[f"{t_dense * 1000:.2f} ms", f"{t_fwht * 1000:.3f} ms"],
                textposition="outside",
            )
        )
        bars.update_layout(
            xaxis=dict(title=""),
            yaxis=dict(title="wall clock (ms)", type="log"),
            showlegend=False,
        )
        apply_theme(bars, height=360)
        st.plotly_chart(bars, width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            "Log scale. The gap widens as L grows. At L=12 the dense solve "
            "is dominated by a 4096 x 4096 matrix inversion; FWHT is a "
            "sequence of additions and subtractions."
        )

    st.markdown("##### Asymptotic curves")
    with st.container(border=True):
        Ls = np.arange(4, 14)
        ns = 2.0**Ls
        dense_ops = ns**2.4
        fwht_ops = ns * Ls
        dense_ops = dense_ops / dense_ops.min()
        fwht_ops = fwht_ops / fwht_ops.min()

        asy = go.Figure()
        asy.add_trace(
            go.Scatter(
                x=Ls,
                y=dense_ops,
                name="Dense O(n^2.4) (LAPACK lstsq)",
                mode="lines+markers",
                line=dict(color=ROSE, width=3),
                marker=dict(size=8),
            )
        )
        asy.add_trace(
            go.Scatter(
                x=Ls,
                y=fwht_ops,
                name="FWHT O(n log n)",
                mode="lines+markers",
                line=dict(color=TEAL, width=3),
                marker=dict(size=8),
            )
        )
        asy.add_vline(x=float(L), line=dict(color=INDIGO, width=2, dash="dash"))
        asy.update_layout(
            xaxis=dict(title="L (n = 2^L)"),
            yaxis=dict(title="relative cost", type="log"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_theme(asy, height=360)
        st.plotly_chart(asy, width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            "Normalized to L=4. The dashed indigo line marks the L you chose. "
            "Curves diverge because log-linear beats polynomial."
        )

    with st.expander("Code", icon=":material/code:"):
        st.code(
            """\
from epistasis.fast import fwht_ols_coefficients

# Full biallelic library at full order: FWHT is exact.
beta = fwht_ols_coefficients(
    gpm.binary_packed,
    y_observed,
    sites,
    model_type="global",
)

# Or let EpistasisLinearRegression detect the condition and use FWHT automatically:
from epistasis.models.linear import EpistasisLinearRegression
model = EpistasisLinearRegression(order=L).add_gpm(gpm).fit()
""",
            language="python",
        )


render()
