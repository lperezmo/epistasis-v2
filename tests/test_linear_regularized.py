"""Tests for Ridge, Lasso, and ElasticNet epistasis models."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from epistasis.exceptions import FittingError
from epistasis.models.linear import (
    EpistasisElasticNet,
    EpistasisLasso,
    EpistasisLinearRegression,
    EpistasisRidge,
)
from gpmap import GenotypePhenotypeMap


def _gpm_nsite(n: int, seed: int = 0) -> GenotypePhenotypeMap:
    genotypes = np.array(["".join(g) for g in product("AB", repeat=n)])
    rng = np.random.default_rng(seed)
    phenotypes = rng.normal(size=len(genotypes))
    return GenotypePhenotypeMap(
        wildtype="A" * n,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@pytest.fixture
def gpm3() -> GenotypePhenotypeMap:
    return _gpm_nsite(3, seed=1)


# Shared smoke test across all three regularized models.


@pytest.mark.parametrize(
    "cls",
    [EpistasisRidge, EpistasisLasso, EpistasisElasticNet],
)
def test_fit_predict_roundtrip(
    cls: type,
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = cls(order=2, alpha=0.01).add_gpm(gpm3).fit()
    pred = m.predict()
    assert pred.shape == (len(gpm3.genotypes),)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize(
    "cls",
    [EpistasisRidge, EpistasisLasso, EpistasisElasticNet],
)
def test_fit_populates_epistasis_values(
    cls: type,
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = cls(order=2, alpha=0.01).add_gpm(gpm3).fit()
    assert not np.any(np.isnan(m.epistasis.values))
    np.testing.assert_array_equal(m.epistasis.values, m.thetas)


@pytest.mark.parametrize(
    "cls",
    [EpistasisRidge, EpistasisLasso, EpistasisElasticNet],
)
def test_coef_alias(
    cls: type,
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = cls(order=2, alpha=0.01).add_gpm(gpm3).fit()
    np.testing.assert_array_equal(m.coef_, m.thetas)


@pytest.mark.parametrize(
    "cls",
    [EpistasisRidge, EpistasisLasso, EpistasisElasticNet],
)
def test_predict_before_fit(
    cls: type,
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = cls().add_gpm(gpm3)
    with pytest.raises(FittingError):
        m.predict()


@pytest.mark.parametrize(
    "cls",
    [EpistasisRidge, EpistasisLasso, EpistasisElasticNet],
)
def test_score_before_fit(
    cls: type,
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = cls().add_gpm(gpm3)
    with pytest.raises(FittingError):
        m.score()


@pytest.mark.parametrize(
    "cls",
    [EpistasisRidge, EpistasisLasso, EpistasisElasticNet],
)
def test_compression_ratio_before_fit(
    cls: type,
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = cls().add_gpm(gpm3)
    with pytest.raises(FittingError):
        m.compression_ratio()


@pytest.mark.parametrize(
    "cls",
    [EpistasisRidge, EpistasisLasso, EpistasisElasticNet],
)
def test_hypothesis_with_custom_thetas(
    cls: type,
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = cls(order=1).add_gpm(gpm3)
    thetas = np.zeros(len(m.Xcolumns))
    thetas[0] = 2.0
    out = m.hypothesis(thetas=thetas)
    np.testing.assert_allclose(out, np.full(len(gpm3.genotypes), 2.0))


# Ridge-specific.


def test_ridge_approaches_ols_as_alpha_shrinks(
    gpm3: GenotypePhenotypeMap,
) -> None:
    ols = EpistasisLinearRegression(order=1).add_gpm(gpm3).fit()
    ridge = EpistasisRidge(order=1, alpha=1e-10).add_gpm(gpm3).fit()
    np.testing.assert_allclose(ridge.thetas, ols.thetas, atol=1e-6)


def test_ridge_compression_ratio_is_zero(
    gpm3: GenotypePhenotypeMap,
) -> None:
    """L2 shrinkage does not produce exact zeros."""
    m = EpistasisRidge(order=2, alpha=1.0).add_gpm(gpm3).fit()
    assert m.compression_ratio() == 0.0


# Lasso-specific.


def test_lasso_produces_sparsity_at_large_alpha(
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLasso(order=2, alpha=1.0).add_gpm(gpm3).fit()
    assert m.compression_ratio() > 0.0


def test_lasso_sparsity_monotone_in_alpha(
    gpm3: GenotypePhenotypeMap,
) -> None:
    r_low = EpistasisLasso(order=2, alpha=0.01).add_gpm(gpm3).fit().compression_ratio()
    r_high = EpistasisLasso(order=2, alpha=1.0).add_gpm(gpm3).fit().compression_ratio()
    assert r_high >= r_low


def test_lasso_positive_coefficients_nonneg(
    gpm3: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLasso(order=2, alpha=0.1, positive=True).add_gpm(gpm3).fit()
    assert (m.thetas >= 0).all()  # type: ignore[operator]


# ElasticNet-specific.


def test_elastic_net_rejects_bad_l1_ratio() -> None:
    with pytest.raises(ValueError, match="l1_ratio"):
        EpistasisElasticNet(l1_ratio=-0.1)
    with pytest.raises(ValueError, match="l1_ratio"):
        EpistasisElasticNet(l1_ratio=1.5)


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_elastic_net_respects_l1_ratio_attribute(ratio: float) -> None:
    """v1 bug: l1_ratio was overwritten to 1.0 in __init__ silently. v2 must
    wire user input straight into the sklearn estimator.
    """
    en = EpistasisElasticNet(l1_ratio=ratio)
    assert en._sklearn.l1_ratio == ratio


def test_elastic_net_behavior_differs_from_lasso_at_intermediate_ratio(
    gpm3: GenotypePhenotypeMap,
) -> None:
    """At intermediate l1_ratio, ElasticNet shouldn't produce the exact Lasso
    solution. This is the observable consequence of honoring l1_ratio.
    """
    alpha = 0.05
    en = EpistasisElasticNet(order=2, alpha=alpha, l1_ratio=0.5).add_gpm(gpm3).fit()
    lasso = EpistasisLasso(order=2, alpha=alpha).add_gpm(gpm3).fit()
    assert not np.allclose(en.thetas, lasso.thetas, atol=1e-6)


def test_elastic_net_l1_ratio_one_matches_lasso(
    gpm3: GenotypePhenotypeMap,
) -> None:
    alpha = 0.1
    en = EpistasisElasticNet(order=2, alpha=alpha, l1_ratio=1.0).add_gpm(gpm3).fit()
    lasso = EpistasisLasso(order=2, alpha=alpha).add_gpm(gpm3).fit()
    np.testing.assert_allclose(en.thetas, lasso.thetas, atol=1e-6)
