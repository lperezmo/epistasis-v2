"""Shared foundation for epistasis classifiers.

Classifiers predict a discrete label (typically viable vs non-viable) from the
epistasis design matrix. v1 learned a separate additive model first and scaled
each column of the design matrix by its additive coefficient before handing
the features to the underlying sklearn classifier; v2 preserves that behavior
via composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.base import EpistasisBaseModel
from epistasis.models.linear import EpistasisLinearRegression

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap

__all__ = ["EpistasisClassifierBase"]


class EpistasisClassifierBase(EpistasisBaseModel):
    """Base for epistasis classifiers.

    Holds an `additive` (order-1 `EpistasisLinearRegression`) used to scale
    design-matrix columns onto the per-mutation contribution scale, and an
    `_sklearn` attribute that subclasses set to the concrete classifier. The
    `threshold` binarizes observed phenotypes into class labels (`y > threshold
    -> 1`, else `0`).
    """

    threshold: float
    _sklearn: Any

    def __init__(
        self,
        threshold: float,
        model_type: ModelType = "global",
    ) -> None:
        super().__init__(order=1, model_type=model_type)
        self.threshold = float(threshold)
        self.additive = EpistasisLinearRegression(order=1, model_type=model_type)

    def add_gpm(self, gpm: GenotypePhenotypeMap) -> EpistasisClassifierBase:
        super().add_gpm(gpm)
        self.additive.add_gpm(gpm)
        return self

    # ------------------------------------------------------------------
    # Feature projection: scale each column by its additive coefficient.

    def _projected_X(self, X: Any) -> np.ndarray:
        """Return the additive design matrix scaled by the fitted additive coefs.

        Column `j` of the projection is `additive_X[:, j] * additive.thetas[j]`,
        so each feature represents that mutation's contribution to the
        predicted additive phenotype.
        """
        if self.additive.thetas is None:
            raise FittingError("Additive coefs unavailable; call fit() before _projected_X().")
        Xadd = self.additive._resolve_X(X).astype(np.float64)
        out: np.ndarray = Xadd * self.additive.thetas
        return out

    def _binarize_y(self, y_arr: np.ndarray) -> np.ndarray:
        return (y_arr > self.threshold).astype(np.int64)

    # ------------------------------------------------------------------
    # Fit / predict.

    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> EpistasisClassifierBase:
        y_arr = self._resolve_y(y)

        self.additive.fit(X=X, y=y_arr)
        X_proj = self._projected_X(X)
        y_class = self._binarize_y(y_arr)

        self._sklearn.fit(X_proj, y_class)

        if self._epistasis is not None and hasattr(self._sklearn, "coef_"):
            coef = np.asarray(self._sklearn.coef_).reshape(-1)
            if coef.shape[0] == self._epistasis.n:
                self._epistasis.values = coef

        return self

    def predict(self, X: Any = None) -> np.ndarray:
        X_proj = self._projected_X(X)
        return np.asarray(self._sklearn.predict(X_proj), dtype=np.int64)

    def predict_proba(self, X: Any = None) -> np.ndarray:
        X_proj = self._projected_X(X)
        return np.asarray(self._sklearn.predict_proba(X_proj), dtype=np.float64)

    def predict_log_proba(self, X: Any = None) -> np.ndarray:
        X_proj = self._projected_X(X)
        return np.asarray(self._sklearn.predict_log_proba(X_proj), dtype=np.float64)

    def score(self, X: Any = None, y: Any = None) -> float:
        X_proj = self._projected_X(X)
        y_arr = self._resolve_y(y)
        y_class = self._binarize_y(y_arr)
        return float(self._sklearn.score(X_proj, y_class))

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predicted probability of class 1 for each row of `X`.

        Subclasses with non-logistic link functions override.
        """
        proba = self.predict_proba(X=X)
        return proba[:, 1]
