"""Tests for the LDA, QDA, Gaussian Process, and Gaussian Mixture classifiers."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from epistasis.exceptions import FittingError
from epistasis.models.classifiers import (
    EpistasisGaussianMixture,
    EpistasisGaussianProcess,
    EpistasisLDA,
    EpistasisLogisticRegression,
    EpistasisQDA,
)
from gpmap import GenotypePhenotypeMap


def _viability_gpm(L: int = 6, seed: int = 0) -> GenotypePhenotypeMap:
    """Larger GPM than the logistic suite so QDA's per-class covariance is
    non-degenerate. Mutations stack additively; >= L/2 mutations are non-viable.
    """
    genotypes = np.array(["".join(g) for g in product("AB", repeat=L)])
    rng = np.random.default_rng(seed)
    half = L // 2
    phenotypes = np.array(
        [(-1.0 if g.count("B") >= half else 1.0) for g in genotypes],
        dtype=np.float64,
    ) + rng.normal(scale=0.05, size=len(genotypes))
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@pytest.fixture
def gpm() -> GenotypePhenotypeMap:
    return _viability_gpm(L=6, seed=7)


# Per-class smoke + shape checks across the four new classifiers.

CLASSIFIERS = [
    (EpistasisLDA, {}),
    (EpistasisQDA, {"reg_param": 0.1}),
    (EpistasisGaussianProcess, {}),
    (EpistasisGaussianMixture, {}),
]


@pytest.mark.parametrize("cls,kwargs", CLASSIFIERS)
def test_fit_predict_roundtrip(
    cls: type,
    kwargs: dict,
    gpm: GenotypePhenotypeMap,
) -> None:
    m = cls(threshold=0.0, **kwargs).add_gpm(gpm).fit()
    pred = m.predict()
    assert pred.shape == (len(gpm.genotypes),)
    assert pred.dtype == np.int64
    assert set(pred.tolist()) <= {0, 1}


@pytest.mark.parametrize("cls,kwargs", CLASSIFIERS)
def test_predict_proba_rows_sum_to_one(
    cls: type,
    kwargs: dict,
    gpm: GenotypePhenotypeMap,
) -> None:
    m = cls(threshold=0.0, **kwargs).add_gpm(gpm).fit()
    proba = m.predict_proba()
    assert proba.shape == (len(gpm.genotypes), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("cls,kwargs", CLASSIFIERS)
def test_predict_before_fit_raises(
    cls: type,
    kwargs: dict,
    gpm: GenotypePhenotypeMap,
) -> None:
    m = cls(threshold=0.0, **kwargs).add_gpm(gpm)
    with pytest.raises(FittingError):
        m.predict()


@pytest.mark.parametrize("cls,kwargs", CLASSIFIERS)
def test_score_is_finite(
    cls: type,
    kwargs: dict,
    gpm: GenotypePhenotypeMap,
) -> None:
    m = cls(threshold=0.0, **kwargs).add_gpm(gpm).fit()
    s = m.score()
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("cls,kwargs", CLASSIFIERS)
def test_predict_log_proba_matches_log_of_predict_proba(
    cls: type,
    kwargs: dict,
    gpm: GenotypePhenotypeMap,
) -> None:
    m = cls(threshold=0.0, **kwargs).add_gpm(gpm).fit()
    proba = m.predict_proba()
    log_proba = m.predict_log_proba()
    np.testing.assert_allclose(log_proba, np.log(np.clip(proba, 1e-300, None)), atol=1e-8)


# LDA-specific.


def test_lda_coef_shape(gpm: GenotypePhenotypeMap) -> None:
    m = EpistasisLDA(threshold=0.0).add_gpm(gpm).fit()
    assert m.coef_.shape == (len(m.additive.Xcolumns),)


def test_lda_thetas_match_coef(gpm: GenotypePhenotypeMap) -> None:
    m = EpistasisLDA(threshold=0.0).add_gpm(gpm).fit()
    np.testing.assert_array_equal(m.thetas, m.coef_)


def test_lda_coef_before_fit_raises() -> None:
    m = EpistasisLDA(threshold=0.0)
    with pytest.raises(FittingError):
        _ = m.coef_


# QDA-specific.


def test_qda_thetas_are_class_means_flat(gpm: GenotypePhenotypeMap) -> None:
    m = EpistasisQDA(threshold=0.0, reg_param=0.1).add_gpm(gpm).fit()
    # Two classes, len(Xcolumns) features each ⇒ thetas length = 2 * n_features.
    assert m.thetas.shape == (2 * len(m.additive.Xcolumns),)


def test_qda_hypothesis_with_thetas_raises(gpm: GenotypePhenotypeMap) -> None:
    m = EpistasisQDA(threshold=0.0, reg_param=0.1).add_gpm(gpm).fit()
    with pytest.raises(FittingError, match="no closed-form"):
        m.hypothesis(thetas=m.thetas)


# Gaussian Process-specific.


def test_gp_thetas_are_kernel_hyperparams(gpm: GenotypePhenotypeMap) -> None:
    m = EpistasisGaussianProcess(threshold=0.0).add_gpm(gpm).fit()
    # `kernel_.theta` is the log-transformed kernel hyperparameters.
    assert m.thetas is not None


def test_gp_thetas_before_fit_raises() -> None:
    m = EpistasisGaussianProcess(threshold=0.0)
    with pytest.raises(FittingError):
        _ = m.thetas


# GMM-specific.


def test_gmm_viable_component_set_after_fit(
    gpm: GenotypePhenotypeMap,
) -> None:
    m = EpistasisGaussianMixture(threshold=0.0).add_gpm(gpm).fit()
    assert m._viable_component is not None
    assert 0 <= m._viable_component < m._sklearn.n_components


def test_gmm_class_label_is_higher_phenotype_cluster(
    gpm: GenotypePhenotypeMap,
) -> None:
    """The component mapped to class 1 should genuinely be the high-mean
    component. Otherwise our cluster-to-class mapping is broken.
    """
    m = EpistasisGaussianMixture(threshold=0.0).add_gpm(gpm).fit()
    pred = m.predict()
    phen = np.asarray(gpm.phenotypes, dtype=np.float64)
    if (pred == 1).any() and (pred == 0).any():
        assert phen[pred == 1].mean() > phen[pred == 0].mean()


def test_gmm_predict_proba_columns_are_zero_one_split(
    gpm: GenotypePhenotypeMap,
) -> None:
    """Column 1 of predict_proba is the viable-component's posterior. Verify
    it equals the underlying sklearn posterior on that exact component.
    """
    m = EpistasisGaussianMixture(threshold=0.0).add_gpm(gpm).fit()
    X_proj = m._projected_X(None)
    raw = m._sklearn.predict_proba(X_proj)
    np.testing.assert_allclose(m.predict_proba()[:, 1], raw[:, m._viable_component])


def test_gmm_score_is_accuracy(gpm: GenotypePhenotypeMap) -> None:
    m = EpistasisGaussianMixture(threshold=0.0).add_gpm(gpm).fit()
    pred = m.predict()
    y_class = m._binarize_y(np.asarray(gpm.phenotypes))
    expected = float(np.mean(pred == y_class))
    assert m.score() == pytest.approx(expected)


# Cross-classifier: thresholding behaves the same for all of them via the
# base `_binarize_y`.


@pytest.mark.parametrize("cls,kwargs", CLASSIFIERS + [(EpistasisLogisticRegression, {})])
def test_threshold_is_respected_in_binarize(
    cls: type,
    kwargs: dict,
    gpm: GenotypePhenotypeMap,
) -> None:
    m = cls(threshold=0.25, **kwargs).add_gpm(gpm)
    phen = np.asarray(gpm.phenotypes, dtype=np.float64)
    y_class = m._binarize_y(phen)
    np.testing.assert_array_equal(y_class, (phen > 0.25).astype(np.int64))
