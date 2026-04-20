"""Tests for `epistasis.fast` and the FWHT fast path on `EpistasisLinearRegression`."""

from __future__ import annotations

import itertools as it

import numpy as np
import pytest
from epistasis.fast import fwht_ols_coefficients
from epistasis.mapping import encoding_to_sites
from epistasis.matrix import get_model_matrix
from epistasis.models.linear import EpistasisLinearRegression
from gpmap import GenotypePhenotypeMap


def _make_biallelic_gpm(L: int, seed: int = 0) -> GenotypePhenotypeMap:
    rng = np.random.default_rng(seed)
    n = 1 << L
    genotypes = np.array(["".join(s) for s in it.product("AB", repeat=L)])
    phenotypes = rng.standard_normal(n)
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@pytest.mark.parametrize("L", [2, 3, 4, 5])
def test_fwht_ols_matches_dense_ols(L: int) -> None:
    gpm = _make_biallelic_gpm(L, seed=L)
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    X = get_model_matrix(gpm.binary_packed, sites, model_type="global").astype(np.float64)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)

    dense_beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fwht_beta = fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="global")

    assert fwht_beta is not None
    np.testing.assert_allclose(fwht_beta, dense_beta, atol=1e-10)


def test_fwht_returns_none_for_local_encoding() -> None:
    gpm = _make_biallelic_gpm(3, seed=0)
    sites = encoding_to_sites(order=3, encoding_table=gpm.encoding_table)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)
    assert fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="local") is None


def test_fwht_returns_none_for_partial_library() -> None:
    L = 4
    full = _make_biallelic_gpm(L, seed=0)
    # Drop one genotype; library is no longer 2^L.
    genotypes = np.asarray(full.genotypes)[:-1]
    phenotypes = np.asarray(full.phenotypes)[:-1]
    gpm = GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)
    assert fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="global") is None


def test_fwht_returns_none_for_truncated_order() -> None:
    L = 4
    gpm = _make_biallelic_gpm(L, seed=0)
    sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)
    assert fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="global") is None


def test_fwht_returns_none_for_multiallelic() -> None:
    L = 3
    # Two mutations per position: A -> B and A -> C. Library is 3^L genotypes.
    letters = ["A", "B", "C"]
    genotypes = np.array(["".join(s) for s in it.product(letters, repeat=L)])
    phenotypes = np.random.default_rng(0).standard_normal(len(genotypes))
    gpm = GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)
    assert fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="global") is None


@pytest.mark.parametrize("L", [3, 4, 5])
def test_linear_regression_uses_fwht_and_matches_sklearn(L: int) -> None:
    gpm = _make_biallelic_gpm(L, seed=L + 42)

    fwht_model = EpistasisLinearRegression(order=L).add_gpm(gpm).fit()

    sk_model = EpistasisLinearRegression(order=L).add_gpm(gpm)
    # Force the sklearn path by passing a pre-built X so `fit` skips FWHT.
    X = get_model_matrix(gpm.binary_packed, sk_model.Xcolumns, model_type="global").astype(
        np.float64
    )
    sk_model.fit(X=X, y=np.asarray(gpm.phenotypes, dtype=np.float64))

    assert fwht_model.thetas is not None
    assert sk_model.thetas is not None
    np.testing.assert_allclose(fwht_model.thetas, sk_model.thetas, atol=1e-10)


def test_linear_regression_predict_works_after_fwht() -> None:
    L = 4
    gpm = _make_biallelic_gpm(L, seed=99)
    model = EpistasisLinearRegression(order=L).add_gpm(gpm).fit()
    pred = model.predict()
    np.testing.assert_allclose(pred, np.asarray(gpm.phenotypes, dtype=np.float64), atol=1e-10)


def test_linear_regression_score_is_one_after_fwht() -> None:
    L = 4
    gpm = _make_biallelic_gpm(L, seed=7)
    model = EpistasisLinearRegression(order=L).add_gpm(gpm).fit()
    assert model.score() == pytest.approx(1.0, abs=1e-10)


def test_linear_regression_truncated_order_falls_back_to_sklearn() -> None:
    """Order < L means FWHT is not applicable; the sklearn path must still run."""
    L = 5
    gpm = _make_biallelic_gpm(L, seed=3)
    model = EpistasisLinearRegression(order=2).add_gpm(gpm).fit()
    assert model.thetas is not None
    # The fit should still populate analytic stderrs because the sklearn path ran.
    stdev = model.epistasis.stdeviations
    assert stdev.shape == (len(model.Xcolumns),)
    # Overdetermined system => stderrs are finite for at least some coefficients.
    assert np.isfinite(stdev).any()


def test_linear_regression_fwht_stdeviations_are_nan() -> None:
    """Full-order system is exactly determined; analytic stderr is undefined."""
    L = 3
    gpm = _make_biallelic_gpm(L, seed=1)
    model = EpistasisLinearRegression(order=L).add_gpm(gpm).fit()
    stdev = model.epistasis.stdeviations
    assert np.all(np.isnan(stdev))


def test_fwht_handles_duplicate_bit_sites() -> None:
    """Sites with a repeated bit index cannot form a Hadamard basis."""
    L = 3
    gpm = _make_biallelic_gpm(L, seed=0)
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    # Replace a valid site with one that has a duplicated bit.
    sites = list(sites)
    sites[-1] = (1, 1, 2)
    y = np.asarray(gpm.phenotypes, dtype=np.float64)
    assert fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="global") is None


def test_fwht_rejects_shape_mismatch() -> None:
    """y length must equal binary_packed rows."""
    gpm = _make_biallelic_gpm(3, seed=0)
    sites = encoding_to_sites(order=3, encoding_table=gpm.encoding_table)
    y = np.zeros(4, dtype=np.float64)  # wrong length
    assert fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="global") is None


def test_fwht_fixes_known_coefficients() -> None:
    """Construct y from known beta, recover beta exactly via FWHT."""
    L = 4
    gpm = _make_biallelic_gpm(L, seed=0)
    sites = encoding_to_sites(order=L, encoding_table=gpm.encoding_table)
    X = get_model_matrix(gpm.binary_packed, sites, model_type="global").astype(np.float64)
    rng = np.random.default_rng(17)
    beta_true = rng.standard_normal(len(sites))
    y = X @ beta_true
    beta_fwht = fwht_ols_coefficients(gpm.binary_packed, y, sites, model_type="global")
    assert beta_fwht is not None
    np.testing.assert_allclose(beta_fwht, beta_true, atol=1e-10)
