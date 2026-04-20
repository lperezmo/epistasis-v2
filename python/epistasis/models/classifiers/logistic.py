"""Logistic-regression classifier for viable vs non-viable phenotypes."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.classifiers._base import EpistasisClassifierBase

__all__ = ["EpistasisLogisticRegression"]


class EpistasisLogisticRegression(EpistasisClassifierBase):
    """Logistic regression over the additive-scale-projected design matrix.

    Useful when some phenotypes are non-viable (below `threshold`). The
    classifier first fits an order-1 additive linear model, projects the
    design matrix through the additive coefficients, then fits a sklearn
    `LogisticRegression` to predict the binary viability class.

    Parameters
    ----------
    threshold
        Phenotype cut-off. Samples with phenotype `> threshold` are class 1
        (viable); the rest are class 0.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    C
        Inverse regularization strength (sklearn convention: smaller `C` is
        stronger regularization).
    max_iter
        Maximum solver iterations.
    solver
        sklearn LogisticRegression solver. `"lbfgs"` is a good default.
    random_state
        Seed for solvers with randomness.

    For L1 or ElasticNet penalties, construct the sklearn estimator directly
    and assign it to `self._sklearn` after construction.
    """

    def __init__(
        self,
        threshold: float,
        model_type: ModelType = "global",
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int | None = None,
    ) -> None:
        super().__init__(threshold=threshold, model_type=model_type)
        self._sklearn = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            fit_intercept=False,
        )

    @property
    def coef_(self) -> np.ndarray:
        if not hasattr(self._sklearn, "coef_"):
            raise FittingError("Model has not been fit yet.")
        return np.asarray(self._sklearn.coef_).reshape(-1)

    @property
    def thetas(self) -> np.ndarray:
        return self.coef_

    def lnlike_of_data(
        self,
        X: Any = None,
        y: Any = None,
        yerr: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-sample Bernoulli log-likelihood.

        `yerr` is accepted for signature compatibility with the base class and
        ignored. If `thetas` is passed, it temporarily replaces the fitted
        `_sklearn.coef_` for the calculation.
        """
        y_arr = self._resolve_y(y)
        y_class = self._binarize_y(y_arr)

        if thetas is None:
            p1 = self.hypothesis(X=X)
        else:
            X_proj = self._projected_X(X)
            logit = np.asarray(X_proj @ np.asarray(thetas, dtype=np.float64))
            p1 = 1.0 / (1.0 + np.exp(-logit))

        # Clip to avoid log(0).
        p1 = np.clip(p1, 1e-15, 1.0 - 1e-15)
        out: np.ndarray = np.where(y_class == 1, np.log(p1), np.log(1.0 - p1))
        return out
