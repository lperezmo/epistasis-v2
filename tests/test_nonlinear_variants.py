"""Tests for the power-transform, spline, and monotonic-GE nonlinear models."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from epistasis.exceptions import FittingError
from epistasis.models.nonlinear import (
    EpistasisMonotonicGE,
    EpistasisPowerTransform,
    EpistasisSpline,
    MonotonicGEMinimizer,
    PowerTransformMinimizer,
    SplineMinimizer,
    monotonic_ge,
    power_transform,
)
from gpmap import GenotypePhenotypeMap


def _additive_gpm(L: int = 5, seed: int = 0, slope: float = 0.5) -> GenotypePhenotypeMap:
    genotypes = np.array(["".join(g) for g in product("AB", repeat=L)])
    rng = np.random.default_rng(seed)
    additive = np.array([slope * g.count("B") for g in genotypes], dtype=np.float64)
    additive = additive + rng.normal(scale=0.01, size=len(additive))
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=additive,
    )


def _wrap_phenotypes(gpm: GenotypePhenotypeMap, func) -> GenotypePhenotypeMap:
    y = func(np.asarray(gpm.phenotypes))
    return GenotypePhenotypeMap(
        wildtype=gpm.wildtype,
        genotypes=gpm.genotypes,
        phenotypes=y,
    )


# ======================================================================
# Power transform.
# ======================================================================


def test_power_transform_lambda_zero_log_limit() -> None:
    x = np.linspace(0.5, 2.0, 5)
    out = power_transform(x, lmbda=0.0, A=0.0, B=0.0)
    # GM(x) * log(x) + 0
    gm = np.exp(np.log(x).mean())
    np.testing.assert_allclose(out, gm * np.log(x))


def test_power_transform_lambda_one_is_affine() -> None:
    """lmbda=1 reduces to (x + A - 1) / 1 + B = x + A - 1 + B."""
    x = np.array([1.0, 2.0, 3.0])
    out = power_transform(x, lmbda=1.0, A=0.5, B=0.25)
    np.testing.assert_allclose(out, x + 0.5 - 1.0 + 0.25)


def test_power_transform_nan_when_argument_nonpositive() -> None:
    x = np.array([0.5, 1.0, 2.0])
    # A negative enough to push x + A below zero.
    out = power_transform(x, lmbda=0.5, A=-1.0, B=0.0)
    assert np.all(np.isnan(out))


def test_epistasis_power_transform_fits_known_power_shape() -> None:
    """Apply a known power transform, then check we recover near-zero residual."""
    gpm_add = _additive_gpm(L=5)
    gpm = _wrap_phenotypes(
        gpm_add,
        lambda x: power_transform(x, lmbda=0.5, A=3.0, B=1.0),
    )
    m = EpistasisPowerTransform(model_type="global", lmbda=1.0, A=3.0, B=0.0).add_gpm(gpm).fit()
    assert m.score() > 0.999
    # Additive base has noise(0.01), and the power-transform parameterization
    # has degenerate directions; tighten only to a level consistent with that.
    assert np.max(np.abs(m.predict() - gpm.phenotypes)) < 5e-2


def test_epistasis_power_transform_thetas_order() -> None:
    gpm_add = _additive_gpm(L=4)
    gpm = _wrap_phenotypes(gpm_add, lambda x: power_transform(x, lmbda=0.5, A=2.0, B=0.5))
    m = EpistasisPowerTransform(model_type="global").add_gpm(gpm).fit()
    thetas = m.thetas
    # First three are lmbda, A, B; remainder is the additive linear coefs.
    assert thetas.shape[0] == 3 + len(m.additive.Xcolumns)


def test_epistasis_power_transform_hypothesis_matches_predict() -> None:
    gpm_add = _additive_gpm(L=4)
    gpm = _wrap_phenotypes(gpm_add, lambda x: power_transform(x, lmbda=0.5, A=2.0, B=0.5))
    m = EpistasisPowerTransform(model_type="global").add_gpm(gpm).fit()
    np.testing.assert_allclose(m.hypothesis(thetas=m.thetas), m.predict(), atol=1e-12)


def test_power_transform_minimizer_param_names() -> None:
    m = PowerTransformMinimizer()
    assert m.param_names == ["lmbda", "A", "B"]


def test_power_transform_predict_before_fit() -> None:
    gpm = _additive_gpm(L=3)
    m = EpistasisPowerTransform().add_gpm(gpm)
    with pytest.raises(FittingError):
        m.predict()


# ======================================================================
# Spline.
# ======================================================================


def test_spline_minimizer_param_names_match_degree() -> None:
    m = SplineMinimizer(k=3)
    assert m.param_names == ["c0", "c1", "c2", "c3"]
    m2 = SplineMinimizer(k=1)
    assert m2.param_names == ["c0", "c1"]


def test_spline_minimizer_rejects_bad_degree() -> None:
    with pytest.raises(ValueError, match="k must be in"):
        SplineMinimizer(k=0)
    with pytest.raises(ValueError, match="k must be in"):
        SplineMinimizer(k=6)


def test_spline_fits_smooth_nonlinear_shape() -> None:
    """Saturation-style nonlinearity: 2 * x / (1 + x). Smooth and monotonic."""
    gpm_add = _additive_gpm(L=5)
    gpm = _wrap_phenotypes(gpm_add, lambda x: 2.0 * x / (1.0 + np.abs(x) + 0.5))
    m = EpistasisSpline(k=3, s=None, model_type="global").add_gpm(gpm).fit()
    assert m.score() > 0.99


def test_spline_predict_before_fit() -> None:
    gpm = _additive_gpm(L=3)
    m = EpistasisSpline().add_gpm(gpm)
    with pytest.raises(FittingError):
        m.predict()


def test_spline_hypothesis_matches_predict() -> None:
    gpm_add = _additive_gpm(L=4)
    gpm = _wrap_phenotypes(gpm_add, lambda x: 2.0 * x / (1.0 + np.abs(x) + 0.5))
    m = EpistasisSpline(k=3, model_type="global").add_gpm(gpm).fit()
    np.testing.assert_allclose(m.hypothesis(thetas=m.thetas), m.predict(), atol=1e-10)


# ======================================================================
# Monotonic GE (MAVE-NN style).
# ======================================================================


def test_monotonic_ge_function_shape() -> None:
    x = np.linspace(-2.0, 2.0, 10)
    a = 0.5
    b = np.array([1.0, 0.5])
    c = np.array([1.0, 2.0])
    d = np.array([0.0, -1.0])
    out = monotonic_ge(x, a, b, c, d)
    assert out.shape == x.shape
    # Manual reference.
    ref = a + 1.0 * np.tanh(1.0 * x + 0.0) + 0.5 * np.tanh(2.0 * x + (-1.0))
    np.testing.assert_allclose(out, ref)


def test_monotonic_ge_is_monotonic_with_nonneg_bc() -> None:
    """All b_k, c_k >= 0 ⇒ g(phi) is monotone non-decreasing in phi."""
    x = np.linspace(-3.0, 3.0, 500)
    a = 0.0
    b = np.array([0.7, 0.3, 1.5])
    c = np.array([0.5, 1.5, 2.0])
    d = np.array([-1.0, 0.5, 1.5])
    out = monotonic_ge(x, a, b, c, d)
    assert np.all(np.diff(out) >= -1e-12)


def test_monotonic_ge_minimizer_constraints_bounds() -> None:
    m = MonotonicGEMinimizer(K=3, monotonic=True)
    for k in range(3):
        assert m.parameters[f"b{k}"].min == 0.0
        assert m.parameters[f"c{k}"].min == 0.0


def test_monotonic_ge_unconstrained_drops_bounds() -> None:
    m = MonotonicGEMinimizer(K=3, monotonic=False)
    for k in range(3):
        # lmfit stores -inf when min is unset.
        assert m.parameters[f"b{k}"].min == -np.inf
        assert m.parameters[f"c{k}"].min == -np.inf


def test_monotonic_ge_fits_sigmoidal_shape() -> None:
    gpm_add = _additive_gpm(L=5, slope=0.5)
    gpm = _wrap_phenotypes(gpm_add, lambda x: 2.0 / (1.0 + np.exp(-1.5 * x)) - 1.0)
    m = EpistasisMonotonicGE(K=3, monotonic=True, model_type="global").add_gpm(gpm).fit()
    assert m.score() > 0.99


def test_monotonic_ge_hypothesis_matches_predict() -> None:
    gpm_add = _additive_gpm(L=4, slope=0.5)
    gpm = _wrap_phenotypes(gpm_add, lambda x: 2.0 / (1.0 + np.exp(-1.5 * x)) - 1.0)
    m = EpistasisMonotonicGE(K=3, monotonic=True).add_gpm(gpm).fit()
    np.testing.assert_allclose(m.hypothesis(thetas=m.thetas), m.predict(), atol=1e-10)


def test_monotonic_ge_predict_before_fit() -> None:
    gpm = _additive_gpm(L=3)
    m = EpistasisMonotonicGE().add_gpm(gpm)
    with pytest.raises(FittingError):
        m.predict()


def test_monotonic_ge_rejects_K_zero() -> None:
    with pytest.raises(ValueError, match="K must be >= 1"):
        MonotonicGEMinimizer(K=0)


def test_monotonic_ge_inferred_function_is_monotonic() -> None:
    """Fitted g(phi) from the MAVE-NN-style nonlinearity stays monotonic."""
    gpm_add = _additive_gpm(L=5, slope=0.5)
    gpm = _wrap_phenotypes(gpm_add, lambda x: 2.0 / (1.0 + np.exp(-1.5 * x)) - 1.0)
    m = EpistasisMonotonicGE(K=4, monotonic=True).add_gpm(gpm).fit()
    # Evaluate the fitted nonlinearity on a fine grid; check non-decreasing.
    add = np.asarray(gpm.phenotypes)
    phi_grid = np.linspace(add.min(), add.max(), 200)
    yg = m.minimizer.predict(phi_grid)
    assert np.all(np.diff(yg) >= -1e-9)
