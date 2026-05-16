"""Gaussian-mixture classifier for epistasis viability prediction.

`sklearn.mixture.GaussianMixture` is unsupervised: it fits Gaussian
components to the projected design matrix without consulting any labels. To
use it as a viability classifier we apply a two-step procedure:

1. Fit `n_components` Gaussian components to the additive-projected design
   matrix `X_proj`.
2. For each component, compute the mean of the *observed* phenotype across
   the points assigned to that component. The component with the highest
   mean phenotype is mapped to class 1 ("viable"); the rest are class 0.

This recovers the v1 intent (use a mixture model to separate viable from
non-viable variants) with a deterministic and identifiable class assignment.

References
----------
McLachlan, G. J. & Peel, D. 'Finite Mixture Models.' Wiley (2000).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.classifiers._base import EpistasisClassifierBase

__all__ = ["EpistasisGaussianMixture"]


class EpistasisGaussianMixture(EpistasisClassifierBase):
    """Gaussian-mixture classifier over the additive-projected design matrix.

    Parameters
    ----------
    threshold
        Phenotype cut-off used only to seed the viable/non-viable mapping;
        see Notes.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    n_components
        Number of mixture components. Default `2` (viable vs non-viable).
    covariance_type
        sklearn `GaussianMixture` covariance form.
    max_iter
        EM iterations.
    random_state
        Seed for the EM initialization.

    Notes
    -----
    sklearn's `GaussianMixture.predict` returns the integer index of the
    most-likely component for each sample. We post-process these into binary
    labels by mapping the component whose member phenotypes have the highest
    *mean* to class 1, and all other components to class 0. This guarantees
    a deterministic viable / non-viable assignment even when the EM init
    permutes the component indices.
    """

    def __init__(
        self,
        threshold: float,
        model_type: ModelType = "global",
        n_components: int = 2,
        covariance_type: str = "full",
        max_iter: int = 100,
        random_state: int | None = None,
    ) -> None:
        super().__init__(threshold=threshold, model_type=model_type)
        self._sklearn = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._viable_component: int | None = None

    # ------------------------------------------------------------------
    # Override fit/predict because sklearn's GMM is unsupervised.

    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> EpistasisGaussianMixture:
        y_arr = self._resolve_y(y)

        self.additive.fit(X=X, y=y_arr)
        X_proj = self._projected_X(X)
        self._sklearn.fit(X_proj)

        # Map the highest-mean-phenotype component to class 1.
        comp_labels = np.asarray(self._sklearn.predict(X_proj), dtype=np.int64)
        comp_means = np.array(
            [
                float(np.mean(y_arr[comp_labels == k]))
                if np.any(comp_labels == k)
                else float("-inf")
                for k in range(self._sklearn.n_components)
            ]
        )
        self._viable_component = int(np.argmax(comp_means))
        return self

    def _components_to_class(self, comp_labels: np.ndarray) -> np.ndarray:
        if self._viable_component is None:
            raise FittingError("Call fit() before predicting.")
        out: np.ndarray = (comp_labels == self._viable_component).astype(np.int64)
        return out

    def predict(self, X: Any = None) -> np.ndarray:
        X_proj = self._projected_X(X)
        comp = np.asarray(self._sklearn.predict(X_proj), dtype=np.int64)
        return self._components_to_class(comp)

    def predict_proba(self, X: Any = None) -> np.ndarray:
        """Two-column [P(class 0), P(class 1)] viability probabilities.

        Folds the per-component posteriors into a binary distribution: class
        1 receives the posterior of the viable component, class 0 receives
        the rest.
        """
        if self._viable_component is None:
            raise FittingError("Call fit() before predicting.")
        X_proj = self._projected_X(X)
        comp_proba = np.asarray(self._sklearn.predict_proba(X_proj), dtype=np.float64)
        p1 = comp_proba[:, self._viable_component]
        out = np.column_stack([1.0 - p1, p1])
        return out

    def predict_log_proba(self, X: Any = None) -> np.ndarray:
        p = self.predict_proba(X=X)
        out: np.ndarray = np.log(np.clip(p, 1e-300, None))
        return out

    def score(self, X: Any = None, y: Any = None) -> float:
        """Classification accuracy against `y > threshold`.

        Overrides the base class which calls `self._sklearn.score`. `GaussianMixture`'s
        `score` returns the average log-likelihood rather than a classification
        metric, so we compute accuracy directly here.
        """
        y_arr = self._resolve_y(y)
        y_class = self._binarize_y(y_arr)
        y_pred = self.predict(X=X)
        return float(np.mean(y_pred == y_class))

    @property
    def thetas(self) -> np.ndarray:
        if not hasattr(self._sklearn, "means_"):
            raise FittingError("Model has not been fit yet.")
        return np.asarray(self._sklearn.means_, dtype=np.float64).reshape(-1)

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        if thetas is not None:
            raise FittingError(
                "GaussianMixture has no closed-form linear-coefficient hypothesis. "
                "Use predict_proba() instead."
            )
        return self.predict_proba(X=X)[:, 1]
