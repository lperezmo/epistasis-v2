"""Tests for epistasis.simulate."""

from __future__ import annotations

import numpy as np
import pytest
from epistasis.mapping import encoding_to_sites
from epistasis.matrix import get_model_matrix
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.simulate import simulate_linear_gpm, simulate_random_linear_gpm


def test_simulate_linear_produces_predictable_phenotypes() -> None:
    """With all coefficients zero except intercept, every phenotype equals the
    intercept value."""
    wildtype = "AA"
    mutations = {0: ["A", "B"], 1: ["A", "B"]}
    # 4 sites (intercept, 2 singletons, 1 pair) for order 2 biallelic.
    coefs = np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float64)
    gpm, sites = simulate_linear_gpm(wildtype, mutations, order=2, coefficients=coefs)
    assert len(sites) == 4
    np.testing.assert_allclose(gpm.phenotypes, 2.5)


def test_simulate_linear_matches_handbuilt_design_matrix() -> None:
    wildtype = "AAA"
    mutations = {0: ["A", "B"], 1: ["A", "B"], 2: ["A", "B"]}
    coefs = np.array([1.0, 0.3, -0.2, 0.1, 0.05, -0.05, 0.02, 0.01], dtype=np.float64)
    gpm, sites = simulate_linear_gpm(wildtype, mutations, order=3, coefficients=coefs)
    X = get_model_matrix(gpm.binary_packed, sites, model_type="global")
    expected = X.astype(np.float64) @ coefs
    np.testing.assert_allclose(gpm.phenotypes, expected)


def test_simulate_linear_rejects_wrong_coefficient_length() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        simulate_linear_gpm(
            wildtype="AA",
            mutations={0: ["A", "B"], 1: ["A", "B"]},
            order=2,
            coefficients=np.zeros(99),
        )


def test_simulate_linear_stdeviations_scalar() -> None:
    gpm, _ = simulate_linear_gpm(
        wildtype="AA",
        mutations={0: ["A", "B"], 1: ["A", "B"]},
        order=2,
        coefficients=np.zeros(4),
        stdeviations=0.25,
    )
    np.testing.assert_allclose(gpm.stdeviations, 0.25)


def test_simulate_linear_stdeviations_array() -> None:
    gpm, _ = simulate_linear_gpm(
        wildtype="AA",
        mutations={0: ["A", "B"], 1: ["A", "B"]},
        order=2,
        coefficients=np.zeros(4),
        stdeviations=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    np.testing.assert_allclose(gpm.stdeviations, [0.1, 0.2, 0.3, 0.4])


def test_simulate_random_is_reproducible_with_seed() -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    gpm1, _, coefs1 = simulate_random_linear_gpm(
        wildtype="AA",
        mutations={0: ["A", "B"], 1: ["A", "B"]},
        order=2,
        rng=rng1,
    )
    gpm2, _, coefs2 = simulate_random_linear_gpm(
        wildtype="AA",
        mutations={0: ["A", "B"], 1: ["A", "B"]},
        order=2,
        rng=rng2,
    )
    np.testing.assert_allclose(coefs1, coefs2)
    np.testing.assert_allclose(gpm1.phenotypes, gpm2.phenotypes)


def test_simulate_random_respects_coefficient_range() -> None:
    rng = np.random.default_rng(0)
    _, _, coefs = simulate_random_linear_gpm(
        wildtype="AAA",
        mutations={0: ["A", "B"], 1: ["A", "B"], 2: ["A", "B"]},
        order=2,
        coefficient_range=(-0.1, 0.1),
        rng=rng,
    )
    assert (coefs >= -0.1).all() and (coefs <= 0.1).all()


def test_simulated_gpm_roundtrips_through_linear_fit() -> None:
    """End-to-end: simulate -> fit -> recover coefs (up to OLS exactness)."""
    rng = np.random.default_rng(7)
    gpm, sites, coefs = simulate_random_linear_gpm(
        wildtype="AAA",
        mutations={0: ["A", "B"], 1: ["A", "B"], 2: ["A", "B"]},
        order=3,
        rng=rng,
    )
    model = EpistasisLinearRegression(order=3).add_gpm(gpm).fit()
    np.testing.assert_allclose(model.thetas, coefs, atol=1e-10)


def test_simulate_ternary_position_works() -> None:
    """Non-biallelic positions build a sensible site list."""
    wildtype = "AA"
    mutations = {0: ["A", "B", "C"], 1: ["A", "B"]}
    # Order 1: intercept + 2 mutations at site 0 + 1 mutation at site 1 = 4.
    gpm, sites = simulate_linear_gpm(
        wildtype=wildtype,
        mutations=mutations,
        order=1,
        coefficients=np.array([1.0, 0.3, 0.5, 0.1]),
    )
    assert len(sites) == 4
    # 3 * 2 = 6 genotypes.
    assert len(gpm.genotypes) == 6


def test_sites_are_order_consistent() -> None:
    """The returned sites must match encoding_to_sites on the resulting GPM."""
    gpm, sites = simulate_linear_gpm(
        wildtype="AA",
        mutations={0: ["A", "B"], 1: ["A", "B"]},
        order=2,
        coefficients=np.zeros(4),
    )
    regen = encoding_to_sites(2, gpm.encoding_table)
    assert regen == sites
