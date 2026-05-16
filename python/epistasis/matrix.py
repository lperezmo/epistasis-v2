"""Design-matrix construction for epistasis models.

Consumes `binary_packed` uint8 2D arrays (shape `(n_genotypes, n_bits)`) from
gpmap-v2 and produces the encoded-vector matrix and the epistasis design matrix
for a given list of interaction sites. The numerical work happens in the Rust
extension `epistasis._core`; this module validates inputs, flattens the ragged
sites list, and wraps the output for pandas-consuming callers.

A pure-NumPy reference lives in `epistasis._reference` and is used only by the
test suite to cross-check the Rust kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from epistasis import _core
from epistasis.mapping import Site

__all__ = [
    "build_model_matrix",
    "build_model_matrix_sparse",
    "encode_vectors",
    "get_model_matrix",
    "get_model_matrix_sparse",
    "model_matrix_as_dataframe",
]

ModelType = Literal["global", "local"]


def encode_vectors(
    binary_packed: np.ndarray,
    model_type: ModelType = "global",
) -> np.ndarray:
    """Return the encoded vectors with a leading intercept column.

    Parameters
    ----------
    binary_packed
        uint8 array of shape `(n_genotypes, n_bits)`; values in {0, 1}. The
        `binary_packed` attribute of a gpmap-v2 `GenotypePhenotypeMap` matches
        this layout exactly.
    model_type
        `"global"` produces the Hadamard encoding: `+1` for wildtype bits and
        `-1` for mutant bits, with `+1` at the intercept column. `"local"`
        keeps `0`/`1` unchanged, with `+1` at the intercept column.

    Returns
    -------
    encoded : np.ndarray[int8]
        Shape `(n_genotypes, n_bits + 1)`. Column 0 is the intercept.
    """
    if binary_packed.ndim != 2:
        raise ValueError(
            f"binary_packed must be 2D (n_genotypes, n_bits); got ndim={binary_packed.ndim}."
        )
    if binary_packed.dtype != np.uint8:
        raise ValueError(f"binary_packed must have dtype uint8; got dtype={binary_packed.dtype}.")
    bp = np.ascontiguousarray(binary_packed)
    return _core.encode_vectors(bp, model_type)


def _flatten_sites(sites: Sequence[Site]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ragged list of sites into flat indices + offsets arrays.

    Returns `(sites_flat, sites_offsets)` with dtypes int64 and int64.
    `sites_offsets[j]..sites_offsets[j + 1]` slices `sites_flat` for site `j`.
    """
    lengths = [len(s) for s in sites]
    total = sum(lengths)
    flat = np.empty(total, dtype=np.int64)
    offsets = np.empty(len(sites) + 1, dtype=np.int64)
    offsets[0] = 0
    cursor = 0
    for j, site in enumerate(sites):
        n = len(site)
        if n:
            flat[cursor : cursor + n] = np.asarray(site, dtype=np.int64)
        cursor += n
        offsets[j + 1] = cursor
    return flat, offsets


def build_model_matrix(
    encoded: np.ndarray,
    sites: Sequence[Site],
) -> np.ndarray:
    """Build the epistasis design matrix X.

    Parameters
    ----------
    encoded
        Output of `encode_vectors`, shape `(n_genotypes, n_bits + 1)`. The
        intercept is at column 0; mutation `k` (1-based) is at column `k`.
    sites
        List of interaction sites (tuples of 1-based mutation indices, plus
        `(0,)` for the intercept). Produced by
        `epistasis.mapping.encoding_to_sites`.

    Returns
    -------
    X : np.ndarray[int8]
        Design matrix of shape `(n_genotypes, len(sites))`. Each column is the
        elementwise product of the columns in `encoded` selected by that site.
    """
    if encoded.ndim != 2:
        raise ValueError(f"encoded must be 2D; got ndim={encoded.ndim}.")
    if encoded.dtype != np.int8:
        raise ValueError(f"encoded must have dtype int8; got dtype={encoded.dtype}.")
    enc = np.ascontiguousarray(encoded)
    flat, offsets = _flatten_sites(sites)
    return _core.build_model_matrix(enc, flat, offsets)


def get_model_matrix(
    binary_packed: np.ndarray,
    sites: Sequence[Site],
    model_type: ModelType = "global",
) -> np.ndarray:
    """Convenience: `encode_vectors` followed by `build_model_matrix`."""
    encoded = encode_vectors(binary_packed, model_type=model_type)
    return build_model_matrix(encoded, sites)


def build_model_matrix_sparse(
    encoded: np.ndarray,
    sites: Sequence[Site],
    model_type: ModelType = "global",
) -> csc_matrix:
    """Build the epistasis design matrix as a `scipy.sparse.csc_matrix`.

    For `local` encoding the per-site product columns are 0/1 and only the
    nonzero rows are materialized, so the working set scales with the number
    of nonzeros rather than `n_genotypes * n_sites`. This is the memory-saving
    path enabled by Lasso/ElasticNet when `sparse=True` (or `sparse="auto"`
    on a local-encoded GPM).

    For `global` (Hadamard) encoding every entry is `+/-1`, so the design
    matrix has no exploitable sparsity; we still construct a CSC matrix from
    the dense build for API uniformity, but at higher memory cost than the
    dense form. Prefer dense for `global`.

    Parameters
    ----------
    encoded
        Output of `encode_vectors`, shape `(n_genotypes, n_bits + 1)`,
        dtype int8.
    sites
        List of interaction sites (tuples of column indices into `encoded`).
    model_type
        Must match the `model_type` used to build `encoded`. Used to pick the
        construction strategy.
    """
    if encoded.ndim != 2:
        raise ValueError(f"encoded must be 2D; got ndim={encoded.ndim}.")
    if encoded.dtype != np.int8:
        raise ValueError(f"encoded must have dtype int8; got dtype={encoded.dtype}.")

    n_rows = encoded.shape[0]
    n_cols = len(sites)

    if model_type == "local":
        indptr = np.empty(n_cols + 1, dtype=np.int64)
        indptr[0] = 0
        col_rows: list[np.ndarray] = []
        for j, site in enumerate(sites):
            if len(site) == 0:
                rows = np.arange(n_rows, dtype=np.int64)
            else:
                mask = encoded[:, site[0]] == 1
                for c in site[1:]:
                    mask &= encoded[:, c] == 1
                rows = np.flatnonzero(mask).astype(np.int64, copy=False)
            col_rows.append(rows)
            indptr[j + 1] = indptr[j] + rows.shape[0]
        total = int(indptr[-1])
        indices = np.empty(total, dtype=np.int64)
        cursor = 0
        for rows in col_rows:
            indices[cursor : cursor + rows.shape[0]] = rows
            cursor += rows.shape[0]
        data = np.ones(total, dtype=np.float64)
        return csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))

    dense = build_model_matrix(encoded, sites)
    return csc_matrix(dense.astype(np.float64, copy=False))


def get_model_matrix_sparse(
    binary_packed: np.ndarray,
    sites: Sequence[Site],
    model_type: ModelType = "global",
) -> csc_matrix:
    """Convenience: `encode_vectors` followed by `build_model_matrix_sparse`."""
    encoded = encode_vectors(binary_packed, model_type=model_type)
    return build_model_matrix_sparse(encoded, sites, model_type=model_type)


def model_matrix_as_dataframe(
    matrix: np.ndarray,
    sites: Sequence[Site],
    index: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Wrap a design matrix in a pandas DataFrame for inspection.

    Columns are keyed by site tuples; rows use `index` (e.g. genotype strings)
    if provided, otherwise a default integer index.
    """
    if matrix.shape[1] != len(sites):
        raise ValueError(f"matrix has {matrix.shape[1]} columns but got {len(sites)} sites.")
    cols = [tuple(s) for s in sites]
    return pd.DataFrame(matrix, index=pd.Index(index) if index is not None else None, columns=cols)
