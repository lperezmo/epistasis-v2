"""Cross-validation helpers for epistasis models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from epistasis.stats import pearson, split_gpm

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap

    from epistasis.models.base import EpistasisBaseModel

__all__ = ["holdout", "k_fold"]


def k_fold(
    gpm: GenotypePhenotypeMap,
    model: EpistasisBaseModel,
    k: int = 10,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """K-fold cross-validation. Returns the list of fold R^2 scores.

    The model is re-attached to each training fold via `add_gpm` and then
    scored against held-out phenotypes. Callers should pass a freshly
    constructed model so no fit state leaks between folds.
    """
    generator = rng if rng is not None else np.random.default_rng()
    n = len(gpm.genotypes)
    if k <= 1 or k > n:
        raise ValueError(f"k must be in [2, {n}]; got {k}.")

    order = np.arange(n, dtype=np.int64)
    generator.shuffle(order)
    folds = np.array_split(order, k)

    scores: list[float] = []
    for i in range(k):
        train_idx = np.sort(np.concatenate([f for j, f in enumerate(folds) if j != i]))

        train_gpm, test_gpm = split_gpm(gpm, train_idx=train_idx)

        model.add_gpm(train_gpm)
        model.fit()

        y_pred = model.predict(X=list(test_gpm.genotypes))
        y_obs = np.asarray(test_gpm.phenotypes, dtype=np.float64)
        scores.append(pearson(y_obs, y_pred) ** 2)

    return scores


def holdout(
    gpm: GenotypePhenotypeMap,
    model: EpistasisBaseModel,
    fraction: float = 0.8,
    repeat: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[list[float], list[float]]:
    """Repeated holdout validation.

    Splits `gpm` into train/test by `fraction`, fits on the train subset,
    scores both train and test. Repeats `repeat` times with fresh random
    splits. Returns `(train_scores, test_scores)`.
    """
    generator = rng if rng is not None else np.random.default_rng()
    train_scores: list[float] = []
    test_scores: list[float] = []

    for _ in range(repeat):
        train_gpm, test_gpm = split_gpm(gpm, fraction=fraction, rng=generator)

        model.add_gpm(train_gpm)
        model.fit()

        train_pred = model.predict(X=list(train_gpm.genotypes))
        test_pred = model.predict(X=list(test_gpm.genotypes))

        train_scores.append(
            pearson(np.asarray(train_gpm.phenotypes, dtype=np.float64), train_pred) ** 2
        )
        test_scores.append(
            pearson(np.asarray(test_gpm.phenotypes, dtype=np.float64), test_pred) ** 2
        )

    return train_scores, test_scores
