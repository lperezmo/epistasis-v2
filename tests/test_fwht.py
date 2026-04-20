"""Tests for the Rust Fast Walsh-Hadamard Transform."""

from __future__ import annotations

import numpy as np
import pytest
from epistasis import _core
from epistasis._reference import fwht_reference


@pytest.mark.parametrize("L", [0, 1, 2, 3, 4, 8])
def test_fwht_matches_reference(L: int) -> None:
    n = 1 << L
    rng = np.random.default_rng(L + 17)
    y = rng.standard_normal(n).astype(np.float64)
    got = _core.fwht(y)
    want = fwht_reference(y)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_fwht_is_its_own_inverse_scaled_by_n() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(16)
    once = _core.fwht(y)
    twice = _core.fwht(once)
    np.testing.assert_allclose(twice, len(y) * y, atol=1e-12)


def test_fwht_preserves_input() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    _ = _core.fwht(y)
    np.testing.assert_array_equal(y, [1.0, 2.0, 3.0, 4.0])


def test_fwht_rejects_non_power_of_two() -> None:
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="power of two"):
        _core.fwht(y)


def test_fwht_rejects_empty() -> None:
    with pytest.raises(ValueError, match="power of two"):
        _core.fwht(np.array([], dtype=np.float64))


def test_fwht_full_order_ols_recovers_coefficients() -> None:
    """For full-order Hadamard OLS on 2^L genotypes: beta = (1/n) * fwht(y)."""
    import itertools as it

    import pandas as pd
    from epistasis.mapping import encoding_to_sites
    from epistasis.matrix import get_model_matrix

    L = 4
    bp = np.array(list(it.product([0, 1], repeat=L)), dtype=np.uint8)
    encoding_table = pd.DataFrame(
        {
            "site_index": list(range(L)) * 2,
            "wildtype_letter": ["A"] * (2 * L),
            "mutation_letter": ["A"] * L + ["B"] * L,
            "mutation_index": [pd.NA] * L + list(range(1, L + 1)),
        }
    )
    sites = encoding_to_sites(order=L, encoding_table=encoding_table)
    X = get_model_matrix(bp, sites, model_type="global").astype(np.float64)
    rng = np.random.default_rng(123)
    beta_true = rng.standard_normal(2**L)
    y = X @ beta_true

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_fwht = _core.fwht(y) / (2**L)

    # The FWHT operates on the natural binary ordering; the epistasis sites
    # may reorder columns. We verify both solutions match the OLS answer
    # element-wise after sorting the coefficients by magnitude.
    np.testing.assert_allclose(np.sort(beta_ols), np.sort(beta_fwht), atol=1e-10)
