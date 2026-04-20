"""About page: links, install, credits."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown("### About")
    st.caption(
        "epistasis-v2 is a clean-break rewrite of the original "
        "harmslab/epistasis package, with a Rust hot path and a Walsh-Hadamard "
        "fast solver for full-order biallelic libraries."
    )

    st.markdown("##### Install")
    with st.container(border=True):
        st.code("uv add epistasis-v2", language="bash")
        st.caption(
            "Wheels are published for Python 3.10 through 3.13 on Linux, macOS, "
            "and Windows. No Rust toolchain needed to install from PyPI."
        )

    st.markdown("##### Links")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(":material/code: **Source**")
            st.markdown("[lperezmo/epistasis-v2](https://github.com/lperezmo/epistasis-v2)")
        with col2:
            st.markdown(":material/package_2: **PyPI**")
            st.markdown("[epistasis-v2](https://pypi.org/project/epistasis-v2/)")
        with col3:
            st.markdown(":material/science: **gpmap-v2**")
            st.markdown("[GenotypePhenotypeMap container](https://github.com/lperezmo/gpmap-v2)")

    st.markdown("##### What's new in v2")
    with st.container(border=True):
        st.markdown(
            "- Rust hot-path kernels via PyO3 (`epistasis._core`): `encode_vectors`, "
            "`build_model_matrix`, and an iterative Fast Walsh-Hadamard Transform.\n"
            "- Walsh-Hadamard OLS fast path: full-order biallelic fits go from "
            "O(n^3) dense solve to O(n log n) transform. At L=12 that is over "
            "4000x.\n"
            "- Composition over `@use_sklearn` MRO injection: works with sklearn "
            ">= 1.2 out of the box.\n"
            "- Python 3.10 through 3.13, abi3 wheels, type hints on the public "
            "API, mypy strict in CI.\n"
            "- uv + maturin build, Conventional Commits + python-semantic-release "
            "+ PyPI OIDC for the release flow."
        )

    st.markdown("##### Credits")
    st.caption(
        "Lineage: harmslab/epistasis (v1) by Zach Sailer. Reference: "
        "Sailer & Harms, 'Detecting High-Order Epistasis in Nonlinear "
        "Genotype-Phenotype Maps.' Genetics 205, 1079-1088 (2017)."
    )
    st.caption(
        "License: Unlicense (public domain). This demo is part of the "
        "epistasis-v2 examples and is not packaged on PyPI."
    )


render()
