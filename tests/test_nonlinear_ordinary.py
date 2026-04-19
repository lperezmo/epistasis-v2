"""Tests for epistasis.models.nonlinear."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from epistasis.exceptions import FittingError
from epistasis.models.nonlinear import (
    EpistasisNonlinearRegression,
    FunctionMinimizer,
)
from gpmap import GenotypePhenotypeMap


# Small, well-behaved nonlinear shape for tests.
def saturation(x: np.ndarray, A: float, K: float) -> np.ndarray:
    """Simple Michaelis-Menten saturation: A * x / (K + x)."""
    return A * x / (K + x)


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def _gpm_with_known_shape(
    L: int = 4,
    A: float = 1.5,
    K: float = 2.0,
    seed: int = 0,
    slope: float = 0.5,
) -> GenotypePhenotypeMap:
    """A GPM where each mutation contributes an additive amount, then the
    observed phenotype is `saturation(additive, A, K)`.
    """
    genotypes = np.array(["".join(g) for g in product("AB", repeat=L)])
    rng = np.random.default_rng(seed)
    additive = np.array([slope * g.count("B") for g in genotypes])
    additive = additive + rng.normal(scale=0.01, size=len(additive))
    phenotypes = saturation(additive, A=A, K=K)
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


# FunctionMinimizer.


def test_minimizer_rejects_function_without_x_first() -> None:
    def bad(y: np.ndarray, a: float) -> np.ndarray:  # first arg must be 'x'
        return y * a

    with pytest.raises(ValueError, match="named 'x'"):
        FunctionMinimizer(bad)


def test_minimizer_introspects_parameter_names() -> None:
    fm = FunctionMinimizer(saturation)
    assert fm.param_names == ["A", "K"]


def test_minimizer_uses_initial_guesses() -> None:
    fm = FunctionMinimizer(saturation, initial_guesses={"A": 5.0, "K": 3.0})
    assert fm.parameters["A"].value == 5.0
    assert fm.parameters["K"].value == 3.0


def test_minimizer_defaults_missing_guesses_to_one() -> None:
    fm = FunctionMinimizer(saturation, initial_guesses={"A": 5.0})
    assert fm.parameters["A"].value == 5.0
    assert fm.parameters["K"].value == 1.0


def test_minimizer_fit_recovers_known_params() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 10.0, size=200)
    y = saturation(x, A=2.5, K=1.0) + rng.normal(scale=1e-4, size=x.size)
    fm = FunctionMinimizer(saturation, initial_guesses={"A": 1.0, "K": 1.0})
    fm.fit(x, y)
    assert fm.parameters["A"].value == pytest.approx(2.5, rel=1e-3)
    assert fm.parameters["K"].value == pytest.approx(1.0, rel=1e-3)


def test_minimizer_predict_uses_current_params() -> None:
    fm = FunctionMinimizer(saturation, initial_guesses={"A": 2.0, "K": 1.0})
    x = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(fm.predict(x), saturation(x, 2.0, 1.0))


def test_minimizer_transform_formula() -> None:
    fm = FunctionMinimizer(saturation, initial_guesses={"A": 2.0, "K": 1.0})
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.5, 1.0, 1.2])
    expected = (y - fm.predict(x)) + x
    np.testing.assert_allclose(fm.transform(x, y), expected)


# EpistasisNonlinearRegression.


@pytest.fixture
def gpm_nonlinear() -> GenotypePhenotypeMap:
    return _gpm_with_known_shape(L=4, A=1.5, K=2.0, slope=0.5, seed=42)


def test_construction() -> None:
    m = EpistasisNonlinearRegression(function=saturation, initial_guesses={"A": 1.0, "K": 1.0})
    assert m.order == 1
    assert m.model_type == "global"


def test_add_gpm_wires_additive(gpm_nonlinear: GenotypePhenotypeMap) -> None:
    m = EpistasisNonlinearRegression(function=saturation, initial_guesses={"A": 1.0, "K": 1.0})
    m.add_gpm(gpm_nonlinear)
    assert m.additive.gpm is gpm_nonlinear


def test_predict_before_fit_raises(gpm_nonlinear: GenotypePhenotypeMap) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    with pytest.raises(FittingError):
        m.predict()


def test_thetas_before_fit_raises(gpm_nonlinear: GenotypePhenotypeMap) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    with pytest.raises(FittingError):
        _ = m.thetas


def test_fit_recovers_phenotypes(gpm_nonlinear: GenotypePhenotypeMap) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    m.fit()
    pred = m.predict()
    # Fit should be tight since phenotypes were generated from saturation.
    rss = float(np.sum((pred - gpm_nonlinear.phenotypes) ** 2))
    assert rss < 1.0


def test_thetas_shape(gpm_nonlinear: GenotypePhenotypeMap) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    m.fit()
    assert m.thetas.shape == (m.num_of_params,)
    assert m.num_of_params == 2 + len(m.additive.Xcolumns)


def test_hypothesis_with_custom_thetas(
    gpm_nonlinear: GenotypePhenotypeMap,
) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    m.fit()
    thetas = m.thetas.copy()
    out1 = m.hypothesis(thetas=thetas)
    out2 = m.predict()
    np.testing.assert_allclose(out1, out2, atol=1e-10)


def test_hypothesis_rejects_wrong_theta_count(
    gpm_nonlinear: GenotypePhenotypeMap,
) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    m.fit()
    with pytest.raises(FittingError, match="Expected"):
        m.hypothesis(thetas=np.zeros(3))


def test_transform_linearizes(gpm_nonlinear: GenotypePhenotypeMap) -> None:
    """After a good fit, transform(X, y) should be close to the additive
    prediction: the nonlinear residual (y - f(x_hat)) should be small, so
    (y - f(x_hat)) + x_hat ~ x_hat.
    """
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    m.fit()
    y_lin = m.transform()
    x_hat = m.additive.predict()
    np.testing.assert_allclose(y_lin, x_hat, atol=0.3)


def test_score_is_reasonable(gpm_nonlinear: GenotypePhenotypeMap) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    m.fit()
    r2 = m.score()
    assert 0.8 <= r2 <= 1.0


def test_predict_on_subset_of_genotypes(
    gpm_nonlinear: GenotypePhenotypeMap,
) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    m.fit()
    pred = m.predict(X=["AAAA", "BBBB"])
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))


# Additive-matrix slicing path.


def test_additive_X_slices_higher_order_matrix(
    gpm_nonlinear: GenotypePhenotypeMap,
) -> None:
    """Passing a full-order design matrix: _additive_X slices to the
    additive-only columns.
    """
    from epistasis.mapping import encoding_to_sites
    from epistasis.matrix import get_model_matrix

    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)

    sites_order2 = encoding_to_sites(2, gpm_nonlinear.encoding_table)
    X_order2 = get_model_matrix(gpm_nonlinear.binary_packed, sites_order2, model_type="global")
    sliced = m._additive_X(X_order2)
    assert sliced.shape[1] == len(m.additive.Xcolumns)


def test_additive_X_rejects_too_narrow_matrix(
    gpm_nonlinear: GenotypePhenotypeMap,
) -> None:
    m = EpistasisNonlinearRegression(
        function=saturation, initial_guesses={"A": 1.0, "K": 1.0}
    ).add_gpm(gpm_nonlinear)
    too_narrow = np.ones((4, 2), dtype=np.int8)
    with pytest.raises(FittingError, match="needs"):
        m._additive_X(too_narrow)
