"""Design-matrix construction for epistasis models.

Consumes `binary_packed` uint8 2D arrays (shape `(n_genotypes, n_bits)`) from
gpmap-v2 and produces the encoded-vector matrix and the epistasis design matrix
for a given list of interaction sites.

The NumPy implementation here is the correctness reference. The Rust kernel in
`epistasis._core` takes over in Phase 2; until then this path is what every
downstream fit uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from epistasis.mapping import Site

__all__ = [
    "build_model_matrix",
    "encode_vectors",
    "get_model_matrix",
    "model_matrix_as_dataframe",
]

ModelType = Literal["global", "local"]


def _validate_binary_packed(binary_packed: np.ndarray) -> None:
    if binary_packed.ndim != 2:
        raise ValueError(
            f"binary_packed must be 2D (n_genotypes, n_bits); got ndim={binary_packed.ndim}."
        )
    if binary_packed.dtype != np.uint8:
        raise ValueError(f"binary_packed must have dtype uint8; got dtype={binary_packed.dtype}.")
    if np.any((binary_packed != 0) & (binary_packed != 1)):
        raise ValueError("binary_packed entries must be 0 or 1.")


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
    _validate_binary_packed(binary_packed)

    n, n_bits = binary_packed.shape
    encoded = np.empty((n, n_bits + 1), dtype=np.int8)
    encoded[:, 0] = 1

    if model_type == "global":
        encoded[:, 1:] = 1 - 2 * binary_packed.astype(np.int8)
    elif model_type == "local":
        encoded[:, 1:] = binary_packed.astype(np.int8)
    else:
        raise ValueError(f"model_type must be 'global' or 'local'; got {model_type!r}.")

    return encoded


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
    n, vec_len = encoded.shape
    m = len(sites)

    X = np.empty((n, m), dtype=np.int8)
    for j, site in enumerate(sites):
        if len(site) == 0:
            raise ValueError(f"site at index {j} is empty.")
        idx = np.asarray(site, dtype=np.intp)
        if idx.min() < 0 or idx.max() >= vec_len:
            raise ValueError(f"site {site} has index out of range for encoded width {vec_len}.")
        if idx.shape[0] == 1:
            X[:, j] = encoded[:, idx[0]]
        else:
            X[:, j] = np.prod(encoded[:, idx], axis=1).astype(np.int8)

    return X


def get_model_matrix(
    binary_packed: np.ndarray,
    sites: Sequence[Site],
    model_type: ModelType = "global",
) -> np.ndarray:
    """Convenience: `encode_vectors` followed by `build_model_matrix`."""
    encoded = encode_vectors(binary_packed, model_type=model_type)
    return build_model_matrix(encoded, sites)


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
