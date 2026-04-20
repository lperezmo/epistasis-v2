"""Benchmarks for the Rust design-matrix kernels vs the NumPy reference.

Run with:
    uv run pytest benches/ --benchmark-only

The `tests/` path is the default for pytest in this project, so these files
are only picked up when you point pytest at `benches/` explicitly.
"""

from __future__ import annotations

import itertools as it

import numpy as np
import pytest
from epistasis._reference import build_model_matrix_reference, encode_vectors_reference
from epistasis.matrix import build_model_matrix, encode_vectors


def _all_binary_packed(n_bits: int) -> np.ndarray:
    return np.array(list(it.product([0, 1], repeat=n_bits)), dtype=np.uint8)


def _enumerate_sites(n_bits: int, order: int) -> list[tuple[int, ...]]:
    sites: list[tuple[int, ...]] = [(0,)]
    for k in range(1, order + 1):
        sites.extend(it.combinations(range(1, n_bits + 1), k))
    return sites


@pytest.fixture(scope="module")
def binary_packed_16() -> np.ndarray:
    return _all_binary_packed(16)


@pytest.fixture(scope="module")
def binary_packed_20() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(200_000, 20), dtype=np.uint8)


@pytest.mark.parametrize("L", [8, 12, 16])
def test_bench_encode_vectors_rust(benchmark, L: int) -> None:
    bp = _all_binary_packed(L)
    benchmark(encode_vectors, bp, "global")


@pytest.mark.parametrize("L", [8, 12, 16])
def test_bench_encode_vectors_numpy(benchmark, L: int) -> None:
    bp = _all_binary_packed(L)
    benchmark(encode_vectors_reference, bp, "global")


@pytest.mark.parametrize(("L", "order"), [(8, 3), (12, 3), (16, 2), (16, 3)])
def test_bench_build_model_matrix_rust(benchmark, L: int, order: int) -> None:
    bp = _all_binary_packed(L)
    enc = encode_vectors(bp, model_type="global")
    sites = _enumerate_sites(L, order)
    benchmark(build_model_matrix, enc, sites)


@pytest.mark.parametrize(("L", "order"), [(8, 3), (12, 3), (16, 2), (16, 3)])
def test_bench_build_model_matrix_numpy(benchmark, L: int, order: int) -> None:
    bp = _all_binary_packed(L)
    enc = encode_vectors_reference(bp, model_type="global")
    sites = _enumerate_sites(L, order)
    benchmark(build_model_matrix_reference, enc, sites)
