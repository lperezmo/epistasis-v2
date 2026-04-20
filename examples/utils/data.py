"""Cached synthetic GPMs used across showcase pages."""

from __future__ import annotations

import itertools as it

import numpy as np
import streamlit as st
from gpmap import GenotypePhenotypeMap


def _genotype_strings(L: int) -> np.ndarray:
    return np.array(["".join(s) for s in it.product("AB", repeat=L)])


@st.cache_resource(show_spinner=False)
def full_biallelic_gpm(L: int, seed: int = 0) -> GenotypePhenotypeMap:
    """Build the complete 2^L biallelic library with random phenotypes."""
    rng = np.random.default_rng(seed)
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=_genotype_strings(L),
        phenotypes=rng.standard_normal(1 << L),
    )


@st.cache_resource(show_spinner=False)
def gpm_from_coefficients(
    L: int,
    coefficients: tuple[float, ...],
    noise: float,
    seed: int,
    model_type: str = "global",
) -> GenotypePhenotypeMap:
    """Build a GPM with phenotypes X @ coefs + noise.

    `coefficients` is a tuple (hashable for cache). Length must equal the
    number of sites at full order for the chosen L, otherwise the caller
    must pad / truncate before calling.
    """
    from epistasis.mapping import encoding_to_sites
    from epistasis.matrix import get_model_matrix

    rng = np.random.default_rng(seed)
    genotypes = _genotype_strings(L)
    gpm = GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=np.zeros(1 << L),
    )
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    X = get_model_matrix(gpm.binary_packed, sites, model_type=model_type).astype(np.float64)
    beta = np.asarray(coefficients, dtype=np.float64)
    y = X @ beta + rng.standard_normal(1 << L) * noise
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=y,
    )
