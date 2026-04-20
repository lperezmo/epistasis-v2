"""Tests for epistasis.models.classifiers.logistic."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from epistasis.exceptions import FittingError
from epistasis.models.classifiers import EpistasisLogisticRegression
from gpmap import GenotypePhenotypeMap


def _viability_gpm(L: int = 4, seed: int = 0) -> GenotypePhenotypeMap:
    """GPM where genotypes with >=2 mutations become non-viable (phenotype 0)
    and others are linearly additive around the threshold.
    """
    genotypes = np.array(["".join(g) for g in product("AB", repeat=L)])
    rng = np.random.default_rng(seed)
    phenotypes = np.empty(len(genotypes), dtype=np.float64)
    for i, g in enumerate(genotypes):
        n_mut = g.count("B")
        base = 1.0 if n_mut < 2 else -1.0
        phenotypes[i] = base + rng.normal(scale=0.05)
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@pytest.fixture
def gpm_viability() -> GenotypePhenotypeMap:
    return _viability_gpm(L=4, seed=42)


# Construction guards.


def test_construction_defaults() -> None:
    m = EpistasisLogisticRegression(threshold=0.0)
    assert m.threshold == 0.0
    assert m.model_type == "global"
    assert m.order == 1


def test_coef_before_fit_raises() -> None:
    m = EpistasisLogisticRegression(threshold=0.0)
    with pytest.raises(FittingError):
        _ = m.coef_


def test_predict_before_fit_raises(gpm_viability: GenotypePhenotypeMap) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability)
    with pytest.raises(FittingError):
        m.predict()


def test_projected_X_before_fit_raises(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability)
    with pytest.raises(FittingError, match="Additive coefs"):
        m._projected_X(None)


# Fit behavior.


def test_fit_predict_roundtrip(gpm_viability: GenotypePhenotypeMap) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    preds = m.predict()
    assert preds.shape == (len(gpm_viability.genotypes),)
    assert preds.dtype == np.int64
    assert set(preds.tolist()) <= {0, 1}


def test_fit_populates_additive(gpm_viability: GenotypePhenotypeMap) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    assert m.additive.thetas is not None


def test_score_is_reasonable(gpm_viability: GenotypePhenotypeMap) -> None:
    """On the structured viability data, logistic should do better than chance."""
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    assert m.score() >= 0.7


def test_predict_proba_shape_and_rows_sum_to_one(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    proba = m.predict_proba()
    assert proba.shape == (len(gpm_viability.genotypes), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_predict_log_proba_matches_log_of_predict_proba(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    proba = m.predict_proba()
    log_proba = m.predict_log_proba()
    np.testing.assert_allclose(log_proba, np.log(proba), atol=1e-8)


def test_hypothesis_is_p1_probability(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    p1 = m.hypothesis()
    proba = m.predict_proba()
    np.testing.assert_allclose(p1, proba[:, 1], atol=1e-10)


def test_predict_on_genotype_subset(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    preds = m.predict(X=["AAAA", "BBBB"])
    assert preds.shape == (2,)
    # Wildtype should be viable, all-mutants should not be.
    assert preds[0] == 1
    assert preds[1] == 0


# Likelihood.


def test_lnlike_of_data_is_negative_finite(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    ll = m.lnlike_of_data()
    assert ll.shape == (len(gpm_viability.genotypes),)
    assert (ll <= 0).all()
    assert np.all(np.isfinite(ll))


def test_lnlike_of_data_with_custom_thetas(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    zeros = np.zeros_like(m.thetas)
    ll_zero = m.lnlike_of_data(thetas=zeros)
    # Under thetas=0, every p1 = 0.5, so per-sample lnlike = log(0.5).
    np.testing.assert_allclose(ll_zero, np.log(0.5), atol=1e-8)


def test_lnlikelihood_aggregates(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    total = m.lnlikelihood()
    per_sample = m.lnlike_of_data()
    assert total == pytest.approx(float(np.sum(per_sample)))


# Parameter validation and effects.


def test_binarize_y_respects_threshold(gpm_viability: GenotypePhenotypeMap) -> None:
    """Threshold controls the class split before handing labels to sklearn."""
    m = EpistasisLogisticRegression(threshold=0.5).add_gpm(gpm_viability)
    phen = np.asarray(gpm_viability.phenotypes, dtype=np.float64)
    y_class = m._binarize_y(phen)
    expected = (phen > 0.5).astype(np.int64)
    np.testing.assert_array_equal(y_class, expected)


def test_threshold_shifts_class_assignments(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    """Higher threshold moves at least as many samples into class 0."""
    m_low = EpistasisLogisticRegression(threshold=-0.9).add_gpm(gpm_viability)
    m_high = EpistasisLogisticRegression(threshold=0.9).add_gpm(gpm_viability)

    phen = np.asarray(gpm_viability.phenotypes, dtype=np.float64)
    low = m_low._binarize_y(phen)
    high = m_high._binarize_y(phen)
    assert high.sum() <= low.sum()


def test_coef_shape_matches_features(
    gpm_viability: GenotypePhenotypeMap,
) -> None:
    m = EpistasisLogisticRegression(threshold=0.0).add_gpm(gpm_viability).fit()
    # Additive has order 1 features; the intercept plus one coef per mutation.
    assert m.coef_.shape == (len(m.additive.Xcolumns),)
