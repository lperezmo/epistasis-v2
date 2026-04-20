"""Tests for epistasis.sampling.bayesian."""

from __future__ import annotations

import numpy as np
import pytest
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.sampling import BayesianSampler
from epistasis.simulate import simulate_random_linear_gpm


def _fitted_linear_model():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(17)
    gpm, _, _ = simulate_random_linear_gpm(
        wildtype="AA",
        mutations={0: ["A", "B"], 1: ["A", "B"]},
        order=1,
        stdeviations=0.1,
        rng=rng,
    )
    model = EpistasisLinearRegression(order=1).add_gpm(gpm).fit()
    return model


def test_sampler_init() -> None:
    m = _fitted_linear_model()
    s = BayesianSampler(m)
    assert s.ndim == m.thetas.shape[0]
    assert s.n_walkers == 2 * s.ndim


def test_sampler_rejects_too_few_walkers() -> None:
    m = _fitted_linear_model()
    with pytest.raises(ValueError, match="n_walkers"):
        BayesianSampler(m, n_walkers=1)


def test_initial_walkers_shape() -> None:
    m = _fitted_linear_model()
    s = BayesianSampler(m)
    pos = s.initial_walkers(rng=np.random.default_rng(0))
    assert pos.shape == (s.n_walkers, s.ndim)


def test_initial_walkers_are_centered_on_ml() -> None:
    m = _fitted_linear_model()
    s = BayesianSampler(m)
    pos = s.initial_walkers(relative_width=1e-6, rng=np.random.default_rng(0))
    # With tiny relative width, walker cloud clusters tightly at ml_thetas.
    np.testing.assert_allclose(pos.mean(axis=0), s.ml_thetas, atol=1e-4)


def test_sample_returns_flat_chain() -> None:
    m = _fitted_linear_model()
    s = BayesianSampler(m)
    chain = s.sample(n_steps=20, n_burn=5, rng=np.random.default_rng(0))
    assert chain.ndim == 2
    assert chain.shape[1] == s.ndim
    assert chain.shape[0] == 20 * s.n_walkers


def test_chain_values_are_finite() -> None:
    m = _fitted_linear_model()
    s = BayesianSampler(m)
    chain = s.sample(n_steps=20, n_burn=5, rng=np.random.default_rng(0))
    assert np.all(np.isfinite(chain))


def test_sample_continues_from_last_state() -> None:
    m = _fitted_linear_model()
    s = BayesianSampler(m)
    s.sample(n_steps=10, n_burn=5, rng=np.random.default_rng(0))
    chain2 = s.sample(n_steps=10, n_burn=0)
    # Second call should yield new samples; length grows.
    assert chain2.shape[0] >= 10 * s.n_walkers


def test_custom_lnprior_is_applied() -> None:
    """A Gaussian log-prior on thetas is evaluated during sampling."""
    m = _fitted_linear_model()
    calls = {"n": 0}

    def tracking_prior(thetas: np.ndarray) -> float:
        calls["n"] += 1
        return float(-0.5 * np.sum(thetas**2))

    s = BayesianSampler(m, lnprior=tracking_prior)
    s.sample(n_steps=5, n_burn=2, rng=np.random.default_rng(0))
    assert calls["n"] > 0
