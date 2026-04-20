"""Benchmark the FWHT fast path against dense OLS on full-order Hadamard fits."""

from __future__ import annotations

import itertools as it

import numpy as np
import pytest
from epistasis.fast import fwht_ols_coefficients
from epistasis.mapping import encoding_to_sites
from epistasis.matrix import get_model_matrix
from gpmap import GenotypePhenotypeMap


def _make_biallelic_gpm(L: int) -> GenotypePhenotypeMap:
    rng = np.random.default_rng(L)
    n = 1 << L
    genotypes = np.array(["".join(s) for s in it.product("AB", repeat=L)])
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=rng.standard_normal(n),
    )


@pytest.mark.parametrize("L", [6, 8, 10, 12])
def test_bench_fwht_full_order(benchmark, L: int) -> None:
    gpm = _make_biallelic_gpm(L)
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)
    benchmark(fwht_ols_coefficients, gpm.binary_packed, y, sites, "global")


@pytest.mark.parametrize("L", [6, 8, 10, 12])
def test_bench_dense_ols_full_order(benchmark, L: int) -> None:
    gpm = _make_biallelic_gpm(L)
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    X = get_model_matrix(gpm.binary_packed, sites, model_type="global").astype(np.float64)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)
    benchmark(np.linalg.lstsq, X, y, None)
