"""Tests for epistasis.models.linear.ordinary."""

from __future__ import annotations

import numpy as np
import pytest
from epistasis.exceptions import FittingError
from epistasis.models.linear import EpistasisLinearRegression
from gpmap import GenotypePhenotypeMap


@pytest.fixture
def gpm_2site() -> GenotypePhenotypeMap:
    return GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.3, 0.5, 1.2]),
        stdeviations=np.array([0.1, 0.1, 0.1, 0.1]),
    )


def _all_genotypes(L: int, alphabet: tuple[str, str] = ("A", "B")) -> np.ndarray:
    from itertools import product

    return np.array(["".join(g) for g in product(alphabet, repeat=L)])


@pytest.fixture
def gpm_3site() -> GenotypePhenotypeMap:
    genotypes = _all_genotypes(3)
    rng = np.random.default_rng(42)
    phenotypes = rng.normal(size=len(genotypes))
    return GenotypePhenotypeMap(
        wildtype="AAA",
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


# Construction.


def test_defaults() -> None:
    m = EpistasisLinearRegression()
    assert m.order == 1
    assert m.model_type == "global"
    assert m.thetas is None


def test_custom_order_and_model_type() -> None:
    m = EpistasisLinearRegression(order=3, model_type="local")
    assert m.order == 3
    assert m.model_type == "local"


# Coef accessor guards.


def test_coef_accessor_errors_before_fit() -> None:
    m = EpistasisLinearRegression()
    with pytest.raises(FittingError):
        _ = m.coef_


def test_predict_errors_before_fit(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site)
    with pytest.raises(FittingError):
        m.predict()


def test_score_errors_before_fit(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site)
    with pytest.raises(FittingError):
        m.score()


def test_hypothesis_errors_without_thetas(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site)
    with pytest.raises(FittingError):
        m.hypothesis()


# Fit behavior.


def test_full_order_fit_is_exact(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    pred = m.predict()
    np.testing.assert_allclose(pred, gpm_2site.phenotypes, atol=1e-10)


def test_fit_populates_thetas(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    assert m.thetas is not None
    assert m.thetas.shape == (len(m.Xcolumns),)


def test_fit_populates_epistasis_values(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    np.testing.assert_allclose(m.epistasis.values, m.thetas)


def test_coef_alias_matches_thetas(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    np.testing.assert_array_equal(m.coef_, m.thetas)


def test_score_is_one_for_exact_fit(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    assert m.score() == pytest.approx(1.0)


def test_predict_on_subset_of_genotypes(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    pred = m.predict(X=["AA", "BB"])
    np.testing.assert_allclose(pred, [0.0, 1.2], atol=1e-10)


def test_hypothesis_with_custom_thetas(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    n_params = len(m.Xcolumns)
    thetas = np.zeros(n_params)
    thetas[0] = 2.5  # intercept only
    out = m.hypothesis(thetas=thetas)
    np.testing.assert_allclose(out, 2.5 * np.ones(4))


def test_lnlike_runs_after_fit(gpm_2site: GenotypePhenotypeMap) -> None:
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    ll = m.lnlike_of_data()
    assert ll.shape == (4,)


# Standard errors (issue #56).


def test_stderr_is_nan_when_not_overdetermined(
    gpm_2site: GenotypePhenotypeMap,
) -> None:
    """Full-order fit on 4 genotypes with 4 coefs: n == p, stderr undefined."""
    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site).fit()
    assert np.isnan(m.epistasis.stdeviations).all()


def test_stderr_is_finite_when_overdetermined(
    gpm_3site: GenotypePhenotypeMap,
) -> None:
    """Order-1 fit on 8 genotypes with 4 coefs: n > p, stderr defined."""
    m = EpistasisLinearRegression(order=1).add_gpm(gpm_3site).fit()
    assert np.isfinite(m.epistasis.stdeviations).all()
    assert (m.epistasis.stdeviations >= 0).all()


def test_stderr_shape_matches_thetas(
    gpm_3site: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLinearRegression(order=1).add_gpm(gpm_3site).fit()
    assert m.epistasis.stdeviations.shape == m.epistasis.values.shape


def test_stderr_matches_manual_formula(
    gpm_3site: GenotypePhenotypeMap,
) -> None:
    """Compare stderr to sigma_hat * sqrt(diag((X.T X)^-1))."""
    m = EpistasisLinearRegression(order=1).add_gpm(gpm_3site).fit()

    X = m._resolve_X(None).astype(np.float64)
    y = m._resolve_y(None)
    n, p = X.shape
    resid = y - X @ m.thetas  # type: ignore[operator]
    sigma2 = float(np.sum(resid**2) / (n - p))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    expected = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    np.testing.assert_allclose(m.epistasis.stdeviations, expected, rtol=1e-10)


# Design matrix passthrough.


def test_fit_accepts_raw_design_matrix(gpm_2site: GenotypePhenotypeMap) -> None:
    from epistasis.matrix import get_model_matrix

    m = EpistasisLinearRegression(order=2).add_gpm(gpm_2site)
    X = get_model_matrix(gpm_2site.binary_packed, m.Xcolumns, model_type="global")
    m.fit(X=X, y=gpm_2site.phenotypes)
    np.testing.assert_allclose(m.predict(X=X), gpm_2site.phenotypes, atol=1e-10)
