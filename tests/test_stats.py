"""Tests for epistasis.stats."""

from __future__ import annotations

import numpy as np
import pytest
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.simulate import simulate_random_linear_gpm
from epistasis.stats import aic, pearson, r_squared, rmsd, split_gpm, ss_residuals
from gpmap import GenotypePhenotypeMap


def test_pearson_identical_is_one() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(y, y) == pytest.approx(1.0)


def test_pearson_anticorrelated_is_negative_one() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(y, -y) == pytest.approx(-1.0)


def test_pearson_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape"):
        pearson(np.zeros(3), np.zeros(4))


def test_r_squared_identical_is_one() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_constant_obs_returns_nan() -> None:
    y = np.array([1.0, 1.0, 1.0])
    assert np.isnan(r_squared(y, np.array([1.0, 2.0, 3.0])))


def test_ss_residuals_zero_on_exact() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert ss_residuals(y, y) == 0.0


def test_rmsd_matches_formula() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 4.0])
    # Residuals: 1, 0, 1 -> MSE = 2/3 -> RMSD = sqrt(2/3).
    assert rmsd(a, b) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_aic_agrees_with_manual_calc() -> None:
    rng = np.random.default_rng(0)
    gpm, _, _ = simulate_random_linear_gpm(
        wildtype="AA",
        mutations={0: ["A", "B"], 1: ["A", "B"]},
        order=2,
        stdeviations=0.1,
        rng=rng,
    )
    m = EpistasisLinearRegression(order=2).add_gpm(gpm).fit()
    expected = 2.0 * (m.num_of_params - m.lnlikelihood())
    assert aic(m) == pytest.approx(expected)


# split_gpm.


def _fixture_gpm() -> GenotypePhenotypeMap:
    rng = np.random.default_rng(1)
    gpm, _, _ = simulate_random_linear_gpm(
        wildtype="AAA",
        mutations={0: ["A", "B"], 1: ["A", "B"], 2: ["A", "B"]},
        order=2,
        stdeviations=0.05,
        rng=rng,
    )
    return gpm


def test_split_gpm_fraction() -> None:
    gpm = _fixture_gpm()
    train, test = split_gpm(gpm, fraction=0.75, rng=np.random.default_rng(42))
    assert len(train.genotypes) + len(test.genotypes) == len(gpm.genotypes)
    assert len(train.genotypes) == int(0.75 * len(gpm.genotypes))


def test_split_gpm_explicit_train_idx() -> None:
    gpm = _fixture_gpm()
    train_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    train, test = split_gpm(gpm, train_idx=train_idx)
    assert len(train.genotypes) == 4
    assert len(test.genotypes) == len(gpm.genotypes) - 4


def test_split_gpm_rejects_both_idx_and_fraction() -> None:
    gpm = _fixture_gpm()
    with pytest.raises(ValueError, match="Exactly one"):
        split_gpm(gpm, train_idx=np.array([0]), fraction=0.5)


def test_split_gpm_rejects_bad_fraction() -> None:
    gpm = _fixture_gpm()
    with pytest.raises(ValueError, match="fraction"):
        split_gpm(gpm, fraction=1.5)


def test_split_gpm_is_reproducible_with_rng() -> None:
    gpm = _fixture_gpm()
    t1, _ = split_gpm(gpm, fraction=0.6, rng=np.random.default_rng(99))
    t2, _ = split_gpm(gpm, fraction=0.6, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(t1.genotypes, t2.genotypes)
