"""Shared machinery for regularized linear epistasis models.

Ridge, Lasso, and ElasticNet all follow the same composition pattern: hold a
configured sklearn estimator, forward `fit`/`predict`/`score`, copy the
resulting coefficients into `self.epistasis.values`. This module collects
that shared machinery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from epistasis.exceptions import FittingError
from epistasis.models.base import EpistasisBaseModel

__all__ = ["RegularizedLinearBase"]


class RegularizedLinearBase(EpistasisBaseModel):
    """Base for regularized linear epistasis models.

    Subclasses assign `self._sklearn` to a configured sklearn linear model in
    their `__init__`. The epistasis intercept lives in the design matrix as
    site `(0,)`, so all sklearn estimators must have `fit_intercept=False`.
    """

    _sklearn: Any

    thetas: np.ndarray | None

    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> RegularizedLinearBase:
        X_mat = np.asfortranarray(self._resolve_X(X).astype(np.float64))
        y_arr = self._resolve_y(y)
        self._sklearn.fit(X_mat, y_arr)
        self.thetas = np.reshape(np.asarray(self._sklearn.coef_), (-1,))
        if self._epistasis is not None:
            self._epistasis.values = self.thetas
        return self

    def predict(self, X: Any = None) -> np.ndarray:
        if self.thetas is None:
            raise FittingError("Call fit() before predict().")
        X_mat = np.asfortranarray(self._resolve_X(X).astype(np.float64))
        return np.asarray(self._sklearn.predict(X_mat), dtype=np.float64)

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        X_mat = self._resolve_X(X).astype(np.float64)
        th = thetas if thetas is not None else self.thetas
        if th is None:
            raise FittingError("thetas unavailable; fit() first or pass thetas=.")
        return np.asarray(X_mat @ np.asarray(th, dtype=np.float64))

    def score(self, X: Any = None, y: Any = None) -> float:
        if self.thetas is None:
            raise FittingError("Call fit() before score().")
        X_mat = np.asfortranarray(self._resolve_X(X).astype(np.float64))
        y_arr = self._resolve_y(y)
        return float(self._sklearn.score(X_mat, y_arr))

    @property
    def coef_(self) -> np.ndarray:
        if self.thetas is None:
            raise FittingError("Model has not been fit yet.")
        return self.thetas

    def compression_ratio(self) -> float:
        """Fraction of fitted coefficients that are exactly zero.

        Meaningful for L1-regularized models (Lasso, ElasticNet). For Ridge
        this is almost always 0 because L2 shrinks without zeroing out.
        """
        if self.thetas is None:
            raise FittingError("Call fit() before compression_ratio().")
        n_zero = int(np.sum(self.thetas == 0.0))
        return n_zero / len(self.thetas)
