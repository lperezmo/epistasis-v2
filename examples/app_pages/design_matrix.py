"""Design matrix anatomy: pick L/order/encoding, see X and X^T X."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.plots import HEATMAP_SIGNED, PLOTLY_CONFIG, apply_theme


def _labels_for_sites(sites: list[tuple[int, ...]]) -> list[str]:
    out = []
    for s in sites:
        if len(s) == 1 and s[0] == 0:
            out.append("1")
        else:
            out.append("*".join(f"x{i}" for i in s))
    return out


def render() -> None:
    from epistasis.mapping import encoding_to_sites
    from epistasis.matrix import encode_vectors, get_model_matrix
    from gpmap import GenotypePhenotypeMap

    st.markdown("### Design matrix anatomy")
    st.caption(
        "Every column is a product of encoded mutation indicators across one "
        "interaction site. Built in Rust over rows, then exposed as a NumPy array."
    )

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            L = st.slider("Library size L", 2, 6, 4, key="dm_L")
        with col2:
            order = st.slider("Max interaction order", 1, L, L, key="dm_order")
        with col3:
            encoding = st.radio("Encoding", ["global", "local"], horizontal=True, key="dm_encoding")

        genotypes = [
            "".join("A" if b == 0 else "B" for b in bits) for bits in np.ndindex(*(2,) * L)
        ]
        gpm = GenotypePhenotypeMap(
            wildtype="A" * L,
            genotypes=np.array(genotypes),
            phenotypes=np.zeros(1 << L),
        )
        sites = encoding_to_sites(order=order, encoding_table=gpm.encoding_table)
        X = get_model_matrix(gpm.binary_packed, sites, model_type=encoding)
        site_labels = _labels_for_sites(sites)

        heat = go.Figure(
            go.Heatmap(
                z=X.astype(int),
                x=site_labels,
                y=genotypes,
                colorscale=HEATMAP_SIGNED if encoding == "global" else "Teal",
                zmid=0 if encoding == "global" else None,
                showscale=True,
                colorbar=dict(title="X[i, j]", thickness=12),
                hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z}<extra></extra>",
            )
        )
        heat.update_layout(
            xaxis=dict(title="Interaction site", side="top"),
            yaxis=dict(title="Genotype", autorange="reversed"),
        )
        apply_theme(heat, height=480)
        st.plotly_chart(heat, width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            f"X shape: {X.shape[0]} rows x {X.shape[1]} cols. "
            f"dtype: {X.dtype}. "
            f"Memory: {X.nbytes} bytes."
        )

    st.markdown("##### Orthogonality under global encoding")
    st.markdown(
        "When the library is the complete biallelic cube at full order, "
        "`X^T X` is exactly `2^L * I`. This is what the FWHT fast path "
        "exploits to avoid the dense solve."
    )
    with st.container(border=True):
        if encoding != "global" or order != L:
            st.info(
                "Set encoding to `global` and order to L to see the Hadamard "
                "orthogonality property.",
                icon=":material/info:",
            )
        gram = X.astype(np.int64).T @ X.astype(np.int64)
        gram_fig = go.Figure(
            go.Heatmap(
                z=gram,
                x=site_labels,
                y=site_labels,
                colorscale="Teal",
                showscale=True,
                colorbar=dict(title="X^T X", thickness=12),
                hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z}<extra></extra>",
            )
        )
        gram_fig.update_layout(
            xaxis=dict(title="", side="top"),
            yaxis=dict(title="", autorange="reversed"),
        )
        apply_theme(gram_fig, height=420)
        st.plotly_chart(gram_fig, width="stretch", config=PLOTLY_CONFIG)
        st.caption("Diagonal carries 2^L; all off-diagonals are 0 at full order.")

    # Encoded vectors preview.
    encoded = encode_vectors(gpm.binary_packed, model_type=encoding)
    st.markdown("##### Encoded vectors")
    st.markdown(
        "Under `global`, mutations flip a sign (`+1 -> -1`); under `local` they "
        "flip a switch (`0 -> 1`). The leading intercept column is always `+1`."
    )
    enc_fig = go.Figure(
        go.Heatmap(
            z=encoded.astype(int),
            x=["(intercept)"] + [f"x{i + 1}" for i in range(L)],
            y=genotypes,
            colorscale=HEATMAP_SIGNED if encoding == "global" else "Teal",
            zmid=0 if encoding == "global" else None,
            showscale=True,
            colorbar=dict(title="encoded", thickness=12),
        )
    )
    enc_fig.update_layout(
        xaxis=dict(title="Mutation indicator", side="top"),
        yaxis=dict(title="Genotype", autorange="reversed"),
    )
    apply_theme(enc_fig, height=360)
    st.plotly_chart(enc_fig, width="stretch", config=PLOTLY_CONFIG)

    with st.expander("Code", icon=":material/code:"):
        st.code(
            f"""\
from epistasis.mapping import encoding_to_sites
from epistasis.matrix import encode_vectors, get_model_matrix
from gpmap import GenotypePhenotypeMap

gpm = GenotypePhenotypeMap(
    wildtype="A" * {L},
    genotypes=[...],
    phenotypes=[...],
)

sites = encoding_to_sites(order={order}, encoding_table=gpm.encoding_table)
encoded = encode_vectors(gpm.binary_packed, model_type={encoding!r})
X = get_model_matrix(gpm.binary_packed, sites, model_type={encoding!r})
gram = X.astype(int).T @ X.astype(int)
""",
            language="python",
        )


render()
