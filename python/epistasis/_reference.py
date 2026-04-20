"""Pure-NumPy reference implementations of the design-matrix kernels.

Used as a parity oracle in the test suite. Not called at runtime: the
production path in `epistasis.matrix` routes through the Rust extension
in `epistasis._core`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from epistasis.mapping import Site

__all__ = [
    "build_model_matrix_reference",
    "encode_vectors_reference",
    "fwht_reference",
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


def encode_vectors_reference(
    binary_packed: np.ndarray,
    model_type: ModelType = "global",
) -> np.ndarray:
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


def build_model_matrix_reference(
    encoded: np.ndarray,
    sites: Sequence[Site],
) -> np.ndarray:
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


def fwht_reference(data: np.ndarray) -> np.ndarray:
    """Naive iterative FWHT in float64 for parity checks.

    Input must be a 1D float64 array whose length is a power of two.
    """
    if data.ndim != 1:
        raise ValueError(f"data must be 1D; got ndim={data.ndim}.")
    n = data.shape[0]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"fwht requires a power-of-two length; got {n}.")
    out = data.astype(np.float64, copy=True)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = out[j]
                y = out[j + h]
                out[j] = x + y
                out[j + h] = x - y
        h *= 2
    return out
