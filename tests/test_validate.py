"""Tests for epistasis.validate."""

from __future__ import annotations

import numpy as np
import pytest
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.simulate import simulate_random_linear_gpm
from epistasis.validate import holdout, k_fold


def _gpm_for_cv():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2026)
    gpm, _, _ = simulate_random_linear_gpm(
        wildtype="AAAA",
        mutations={0: ["A", "B"], 1: ["A", "B"], 2: ["A", "B"], 3: ["A", "B"]},
        order=2,
        stdeviations=0.02,
        rng=rng,
    )
    return gpm


def test_k_fold_returns_k_scores() -> None:
    gpm = _gpm_for_cv()
    model = EpistasisLinearRegression(order=2)
    scores = k_fold(gpm, model, k=4, rng=np.random.default_rng(0))
    assert len(scores) == 4
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_k_fold_is_reproducible_with_rng() -> None:
    gpm = _gpm_for_cv()
    s1 = k_fold(gpm, EpistasisLinearRegression(order=2), k=4, rng=np.random.default_rng(0))
    s2 = k_fold(gpm, EpistasisLinearRegression(order=2), k=4, rng=np.random.default_rng(0))
    np.testing.assert_allclose(s1, s2)


def test_k_fold_rejects_bad_k() -> None:
    gpm = _gpm_for_cv()
    model = EpistasisLinearRegression(order=2)
    with pytest.raises(ValueError, match="k"):
        k_fold(gpm, model, k=1)


def test_holdout_returns_equal_length_lists() -> None:
    gpm = _gpm_for_cv()
    model = EpistasisLinearRegression(order=2)
    train, test = holdout(gpm, model, fraction=0.75, repeat=3, rng=np.random.default_rng(1))
    assert len(train) == 3
    assert len(test) == 3


def test_holdout_train_scores_are_reasonable() -> None:
    """Train scores should be near 1 since we fit on a sizeable subset of a
    linear-generated GPM."""
    gpm = _gpm_for_cv()
    model = EpistasisLinearRegression(order=2)
    train, _ = holdout(gpm, model, fraction=0.8, repeat=3, rng=np.random.default_rng(5))
    # 0.5 is a loose floor; the signal is clean, any reasonable fit clears it.
    assert all(s >= 0.5 for s in train)
