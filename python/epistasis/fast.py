"""Algorithmic fast paths for epistasis models.

Currently exposes `fwht_ols_coefficients`: a Fast Walsh-Hadamard Transform-based
closed-form OLS solver for full-order Hadamard-encoded biallelic libraries.
Returns fitted coefficients in `O(n log n)` operations without ever building
the `n x n` design matrix. At `L = 16` that is `~2e5` vs. `~10^11` ops for dense
OLS; memory drops from `~4 GB` to a handful of float64 vectors.

The function returns `None` when the fast path does not apply. Callers fall
back to sklearn (or whichever solver they prefer) in that case.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from epistasis import _core
from epistasis.mapping import Site
from epistasis.matrix import ModelType

__all__ = ["fwht_ols_coefficients"]


def _sites_to_bitmasks(sites: Sequence[Site], n_bits: int) -> np.ndarray | None:
    """Return a length-`len(sites)` int64 array of bitmasks, or None.

    None is returned when any site has a mutation index outside `[1, n_bits]`
    or when two mutations within the same site map to the same bit (only
    possible for multi-allelic encodings, which do not form a 2^L Hadamard
    basis).
    """
    out = np.zeros(len(sites), dtype=np.int64)
    for j, site in enumerate(sites):
        if len(site) == 1 and site[0] == 0:
            continue
        m = 0
        for mut_idx in site:
            bit = int(mut_idx) - 1
            if bit < 0 or bit >= n_bits:
                return None
            bit_mask = 1 << bit
            if m & bit_mask:
                return None
            m |= bit_mask
        out[j] = m
    return out


def fwht_ols_coefficients(
    binary_packed: np.ndarray,
    y: np.ndarray,
    sites: Sequence[Site],
    model_type: ModelType = "global",
) -> np.ndarray | None:
    """Fit full-order OLS via the Fast Walsh-Hadamard Transform.

    Parameters
    ----------
    binary_packed
        uint8 array of shape `(n, n_bits)` with values in {0, 1}. Must be a
        complete biallelic combinatorial library: `n == 2 ** n_bits` with every
        distinct bit pattern present exactly once.
    y
        float64 phenotypes of shape `(n,)`.
    sites
        Interaction sites in the order used by the caller's design matrix.
        Must contain exactly `2 ** n_bits` entries with distinct bitmasks
        (covering the full order).
    model_type
        Only `"global"` (Hadamard encoding) is supported; other values return
        `None`.

    Returns
    -------
    beta : np.ndarray[float64] | None
        Coefficients in the same order as `sites`, or `None` when the fast
        path does not apply.
    """
    if model_type != "global":
        return None
    if binary_packed.ndim != 2 or y.ndim != 1:
        return None
    n, n_bits = binary_packed.shape
    if n != y.shape[0]:
        return None
    if n_bits == 0 or n_bits > 62:
        return None
    n_full = 1 << n_bits
    if n != n_full or len(sites) != n_full:
        return None

    pows = np.int64(1) << np.arange(n_bits, dtype=np.int64)
    g_masks = binary_packed.astype(np.int64) @ pows
    if not np.array_equal(np.sort(g_masks), np.arange(n_full, dtype=np.int64)):
        return None

    s_masks = _sites_to_bitmasks(sites, n_bits)
    if s_masks is None:
        return None
    if not np.array_equal(np.sort(s_masks), np.arange(n_full, dtype=np.int64)):
        return None

    y_natural = np.empty(n_full, dtype=np.float64)
    y_natural[g_masks] = np.asarray(y, dtype=np.float64)
    h = _core.fwht(y_natural) / float(n_full)
    beta: np.ndarray = np.asarray(h[s_masks], dtype=np.float64)
    return beta
