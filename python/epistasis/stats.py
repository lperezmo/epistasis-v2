"""Statistics helpers for epistasis models.

Kept deliberately small. Handful of well-tested metrics and a GPM splitter;
everything else that was in v1's stats.py has been dropped as either unused
(`gmean`, `incremental_*`), redundant (`explained_variance`), or brittle
(`chi_squared`, false-rate helpers). Add back when there is a concrete use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap

__all__ = [
    "aic",
    "pearson",
    "r_squared",
    "rmsd",
    "split_gpm",
    "ss_residuals",
]


def pearson(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient between observed and predicted values."""
    a = np.asarray(y_obs, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    return float(np.corrcoef(a, b)[0, 1])


def r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination `1 - SSR / SST`."""
    a = np.asarray(y_obs, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def ss_residuals(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Residual sum of squares."""
    a = np.asarray(y_obs, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    return float(np.sum((a - b) ** 2))


def rmsd(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean square deviation."""
    a = np.asarray(y_obs, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def aic(model: Any) -> float:
    """AIC = `2 * (k - ln L)`. Model must expose `num_of_params` and
    `lnlikelihood()`.
    """
    k = int(model.num_of_params)
    lnL = float(model.lnlikelihood())
    return 2.0 * (k - lnL)


def split_gpm(
    gpm: GenotypePhenotypeMap,
    *,
    train_idx: np.ndarray | None = None,
    fraction: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[GenotypePhenotypeMap, GenotypePhenotypeMap]:
    """Split a `GenotypePhenotypeMap` into training and test sets.

    Exactly one of `train_idx` or `fraction` must be supplied. When
    `fraction` is given, a random shuffle selects the training subset; pass
    an `rng` for reproducibility.

    Returns `(train_gpm, test_gpm)`.
    """
    from gpmap import GenotypePhenotypeMap

    n = len(gpm.genotypes)
    if (train_idx is None) == (fraction is None):
        raise ValueError("Exactly one of train_idx or fraction must be provided.")

    if fraction is not None:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1); got {fraction}.")
        generator = rng if rng is not None else np.random.default_rng()
        shuffled = np.arange(n, dtype=np.int64)
        generator.shuffle(shuffled)
        cut = int(n * fraction)
        train_arr = np.sort(shuffled[:cut])
        test_arr = np.sort(shuffled[cut:])
    else:
        train_arr = np.asarray(train_idx, dtype=np.int64)
        mask = np.ones(n, dtype=bool)
        mask[train_arr] = False
        test_arr = np.nonzero(mask)[0]

    def _subset(idx: np.ndarray) -> GenotypePhenotypeMap:
        genotypes = np.asarray(gpm.genotypes)[idx]
        phenotypes = np.asarray(gpm.phenotypes)[idx]
        stds = np.asarray(gpm.stdeviations)[idx] if gpm.stdeviations is not None else None
        return GenotypePhenotypeMap(
            wildtype=gpm.wildtype,
            genotypes=genotypes,
            phenotypes=phenotypes,
            stdeviations=stds,
            mutations=gpm.mutations,
        )

    return _subset(train_arr), _subset(test_arr)
