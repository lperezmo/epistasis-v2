"""Tests for epistasis.matrix."""

from __future__ import annotations

import itertools as it

import numpy as np
import pandas as pd
import pytest
from epistasis.mapping import encoding_to_sites
from epistasis.matrix import (
    build_model_matrix,
    encode_vectors,
    get_model_matrix,
    model_matrix_as_dataframe,
)
from gpmap import GenotypePhenotypeMap


def _all_binary_packed(n_bits: int) -> np.ndarray:
    """Enumerate every length-n_bits binary row as a uint8 2D array."""
    rows = np.array(list(it.product([0, 1], repeat=n_bits)), dtype=np.uint8)
    return rows


# encode_vectors.


def test_encode_vectors_global_single_genotype() -> None:
    bp = np.array([[0, 0, 0]], dtype=np.uint8)
    out = encode_vectors(bp, model_type="global")
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out[0], [1, 1, 1, 1])


def test_encode_vectors_global_all_mutations() -> None:
    bp = np.array([[1, 1, 1]], dtype=np.uint8)
    out = encode_vectors(bp, model_type="global")
    np.testing.assert_array_equal(out[0], [1, -1, -1, -1])


def test_encode_vectors_global_mixed() -> None:
    bp = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    out = encode_vectors(bp, model_type="global")
    np.testing.assert_array_equal(out[0], [1, 1, -1, 1, -1])


def test_encode_vectors_local() -> None:
    bp = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    out = encode_vectors(bp, model_type="local")
    np.testing.assert_array_equal(out[0], [1, 0, 1, 1])
    np.testing.assert_array_equal(out[1], [1, 1, 0, 0])


def test_encode_vectors_rejects_unknown_model_type() -> None:
    bp = np.array([[0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="model_type"):
        encode_vectors(bp, model_type="weird")  # type: ignore[arg-type]


def test_encode_vectors_rejects_wrong_ndim() -> None:
    bp = np.array([0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="2D"):
        encode_vectors(bp)


def test_encode_vectors_rejects_wrong_dtype() -> None:
    bp = np.array([[0, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="uint8"):
        encode_vectors(bp)


def test_encode_vectors_rejects_non_binary_values() -> None:
    bp = np.array([[0, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="0 or 1"):
        encode_vectors(bp)


def test_encode_vectors_dtype_is_int8() -> None:
    bp = np.array([[0, 1]], dtype=np.uint8)
    assert encode_vectors(bp).dtype == np.int8


def test_encode_vectors_intercept_column_is_one() -> None:
    bp = np.random.default_rng(0).integers(0, 2, size=(20, 5), dtype=np.uint8)
    for model in ("global", "local"):
        enc = encode_vectors(bp, model_type=model)  # type: ignore[arg-type]
        np.testing.assert_array_equal(enc[:, 0], np.ones(20, dtype=np.int8))


# build_model_matrix.


def test_build_model_matrix_shape() -> None:
    bp = _all_binary_packed(3)
    enc = encode_vectors(bp, model_type="global")
    sites = [(0,), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    X = build_model_matrix(enc, sites)
    assert X.shape == (8, 8)


def test_build_model_matrix_intercept_column_is_one() -> None:
    bp = _all_binary_packed(3)
    enc = encode_vectors(bp, model_type="global")
    X = build_model_matrix(enc, [(0,), (1,), (2,)])
    np.testing.assert_array_equal(X[:, 0], np.ones(8, dtype=np.int8))


def test_build_model_matrix_single_index_copies_column() -> None:
    bp = _all_binary_packed(3)
    enc = encode_vectors(bp, model_type="global")
    X = build_model_matrix(enc, [(1,)])
    np.testing.assert_array_equal(X[:, 0], enc[:, 1])


def test_build_model_matrix_pair_is_product() -> None:
    bp = _all_binary_packed(3)
    enc = encode_vectors(bp, model_type="global")
    X = build_model_matrix(enc, [(1, 2)])
    expected = (enc[:, 1].astype(np.int16) * enc[:, 2].astype(np.int16)).astype(np.int8)
    np.testing.assert_array_equal(X[:, 0], expected)


def test_build_model_matrix_triple_is_product() -> None:
    bp = _all_binary_packed(3)
    enc = encode_vectors(bp, model_type="global")
    X = build_model_matrix(enc, [(1, 2, 3)])
    expected = (
        enc[:, 1].astype(np.int16) * enc[:, 2].astype(np.int16) * enc[:, 3].astype(np.int16)
    ).astype(np.int8)
    np.testing.assert_array_equal(X[:, 0], expected)


def test_build_model_matrix_rejects_empty_site() -> None:
    enc = np.ones((2, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="empty"):
        build_model_matrix(enc, [()])


def test_build_model_matrix_rejects_out_of_range_index() -> None:
    enc = np.ones((2, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="out of range"):
        build_model_matrix(enc, [(5,)])


def test_build_model_matrix_rejects_wrong_encoded_ndim() -> None:
    enc = np.ones(3, dtype=np.int8)
    with pytest.raises(ValueError, match="2D"):
        build_model_matrix(enc, [(0,)])


# get_model_matrix integration + Hadamard property.


def test_get_model_matrix_biallelic_2site_global() -> None:
    bp = _all_binary_packed(2)
    sites = [(0,), (1,), (2,), (1, 2)]
    X = get_model_matrix(bp, sites, model_type="global")
    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(X, expected)


def test_get_model_matrix_biallelic_2site_local() -> None:
    bp = _all_binary_packed(2)
    sites = [(0,), (1,), (2,), (1, 2)]
    X = get_model_matrix(bp, sites, model_type="local")
    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(X, expected)


@pytest.mark.parametrize("L", [2, 3, 4, 5])
def test_hadamard_orthogonality_full_order(L: int) -> None:
    """At full order on biallelic genotypes, X is a 2^L Hadamard matrix.

    Hadamard property: X^T @ X = 2^L * I.
    """
    bp = _all_binary_packed(L)
    encoding_table = pd.DataFrame(
        {
            "site_index": list(range(L)) * 2,
            "wildtype_letter": ["A"] * (2 * L),
            "mutation_letter": ["A"] * L + ["B"] * L,
            "mutation_index": [pd.NA] * L + list(range(1, L + 1)),
        }
    )
    sites = encoding_to_sites(order=L, encoding_table=encoding_table)
    assert len(sites) == 2**L
    X = get_model_matrix(bp, sites, model_type="global").astype(np.int64)
    gram = X.T @ X
    expected = (2**L) * np.eye(2**L, dtype=np.int64)
    np.testing.assert_array_equal(gram, expected)


# Integration with gpmap-v2.


def test_get_model_matrix_with_real_gpm() -> None:
    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
    X = get_model_matrix(gpm.binary_packed, sites, model_type="global")
    assert X.shape == (4, len(sites))
    np.testing.assert_array_equal(X[:, 0], np.ones(4, dtype=np.int8))


# model_matrix_as_dataframe.


def test_model_matrix_as_dataframe_default_index() -> None:
    bp = _all_binary_packed(2)
    sites = [(0,), (1,), (2,), (1, 2)]
    X = get_model_matrix(bp, sites)
    df = model_matrix_as_dataframe(X, sites)
    assert df.shape == (4, 4)
    assert list(df.columns) == sites


def test_model_matrix_as_dataframe_with_index() -> None:
    bp = _all_binary_packed(2)
    sites = [(0,), (1,), (2,), (1, 2)]
    X = get_model_matrix(bp, sites)
    df = model_matrix_as_dataframe(X, sites, index=["00", "01", "10", "11"])
    assert list(df.index) == ["00", "01", "10", "11"]


def test_model_matrix_as_dataframe_column_count_mismatch() -> None:
    X = np.ones((2, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="columns"):
        model_matrix_as_dataframe(X, [(0,), (1,)])
