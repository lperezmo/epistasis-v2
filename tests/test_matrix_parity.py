"""Parity tests: Rust kernel output must match the pure-NumPy reference."""

from __future__ import annotations

import itertools as it

import numpy as np
import pytest
from epistasis._reference import build_model_matrix_reference, encode_vectors_reference
from epistasis.matrix import build_model_matrix, encode_vectors


def _all_binary_packed(n_bits: int) -> np.ndarray:
    return np.array(list(it.product([0, 1], repeat=n_bits)), dtype=np.uint8)


def _random_binary(n: int, n_bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, n_bits), dtype=np.uint8)


@pytest.mark.parametrize("model_type", ["global", "local"])
@pytest.mark.parametrize("shape", [(1, 1), (4, 3), (256, 8), (1024, 16)])
def test_encode_vectors_matches_reference(model_type: str, shape: tuple[int, int]) -> None:
    bp = _random_binary(*shape, seed=shape[0] ^ shape[1])
    got = encode_vectors(bp, model_type=model_type)  # type: ignore[arg-type]
    want = encode_vectors_reference(bp, model_type=model_type)  # type: ignore[arg-type]
    np.testing.assert_array_equal(got, want)
    assert got.dtype == np.int8


def _enumerate_sites(n_bits: int, order: int) -> list[tuple[int, ...]]:
    sites: list[tuple[int, ...]] = [(0,)]
    for k in range(1, order + 1):
        sites.extend(it.combinations(range(1, n_bits + 1), k))
    return sites


@pytest.mark.parametrize(
    ("n_bits", "order"),
    [(3, 1), (3, 2), (3, 3), (5, 2), (5, 3), (8, 2), (8, 3), (10, 2)],
)
def test_build_model_matrix_matches_reference(n_bits: int, order: int) -> None:
    bp = _all_binary_packed(n_bits)
    enc = encode_vectors(bp, model_type="global")
    sites = _enumerate_sites(n_bits, order)
    got = build_model_matrix(enc, sites)
    want = build_model_matrix_reference(enc, sites)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == np.int8


def test_build_model_matrix_large_random_matches_reference() -> None:
    n_bits = 12
    bp = _random_binary(5000, n_bits, seed=42)
    enc = encode_vectors(bp, model_type="global")
    sites = _enumerate_sites(n_bits, order=3)
    got = build_model_matrix(enc, sites)
    want = build_model_matrix_reference(enc, sites)
    np.testing.assert_array_equal(got, want)


def test_non_contiguous_binary_packed_still_matches_reference() -> None:
    """Rust kernel must accept non-contiguous inputs via the Python wrapper."""
    bp = _random_binary(64, 10, seed=7)
    view = bp[::2]  # strided view; not C-contiguous
    got = encode_vectors(view, model_type="global")
    want = encode_vectors_reference(view, model_type="global")
    np.testing.assert_array_equal(got, want)
