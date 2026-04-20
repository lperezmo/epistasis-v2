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

from epistasis import _core
from epistasis.mapping import Site

__all__ = [
    "build_model_matrix",
    "encode_vectors",
    "get_model_matrix",
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
