"""Linear and quadratic discriminant-analysis classifiers for epistasis.

Both wrap sklearn's `LinearDiscriminantAnalysis` / `QuadraticDiscriminantAnalysis`
through the `EpistasisClassifierBase` projection pipeline: the design matrix
is first projected through fitted additive coefficients (so each feature
represents that mutation's contribution to the additive phenotype), then the
discriminant model is fit to the binarized viability labels.

Use these when you want a generative classifier (Gaussian class-conditional
densities). LDA assumes a shared covariance across classes; QDA fits a
per-class covariance.

References
----------
Hastie, T., Tibshirani, R. & Friedman, J. 'The Elements of Statistical
Learning', 2nd ed., Springer (2009). Chapter 4 covers LDA/QDA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.classifiers._base import EpistasisClassifierBase

__all__ = ["EpistasisLDA", "EpistasisQDA"]


class EpistasisLDA(EpistasisClassifierBase):
    """Linear discriminant analysis over the additive-projected design matrix.

    Parameters
    ----------
    threshold
        Phenotype cut-off. Samples with phenotype `> threshold` are class 1.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    solver
        sklearn LDA solver: `"svd"` (default), `"lsqr"`, or `"eigen"`.
    shrinkage
        Shrinkage parameter for `"lsqr"` / `"eigen"` solvers (`None`,
        `"auto"`, or a float in `[0, 1]`).
    priors
        Prior class probabilities. If `None`, inferred from class frequencies.
    """

    def __init__(
        self,
        threshold: float,
        model_type: ModelType = "global",
        solver: str = "svd",
        shrinkage: str | float | None = None,
        priors: np.ndarray | None = None,
    ) -> None:
        super().__init__(threshold=threshold, model_type=model_type)
        self._sklearn = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            priors=priors,
        )

    @property
    def coef_(self) -> np.ndarray:
        if not hasattr(self._sklearn, "coef_"):
            raise FittingError("Model has not been fit yet.")
        return np.asarray(self._sklearn.coef_).reshape(-1)

    @property
    def thetas(self) -> np.ndarray:
        return self.coef_


class EpistasisQDA(EpistasisClassifierBase):
    """Quadratic discriminant analysis over the additive-projected design matrix.

    Unlike LDA, QDA fits a separate covariance per class, so the decision
    boundary is quadratic in feature space.

    Parameters
    ----------
    threshold
        Phenotype cut-off; class 1 is `> threshold`.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    priors
        Prior class probabilities. If `None`, inferred from class frequencies.
    reg_param
        Regularization on the per-class covariance: `Sigma_k := (1 - reg_param)
        * Sigma_k + reg_param * trace(Sigma_k)/n_features * I`. A small value
        (e.g. `1e-4`) helps when classes are nearly degenerate.
    """

    def __init__(
        self,
        threshold: float,
        model_type: ModelType = "global",
        priors: np.ndarray | None = None,
        reg_param: float = 0.0,
    ) -> None:
        super().__init__(threshold=threshold, model_type=model_type)
        self._sklearn = QuadraticDiscriminantAnalysis(
            priors=priors,
            reg_param=reg_param,
        )

    @property
    def thetas(self) -> np.ndarray:
        """QDA has no linear coefficient vector; surfaces class means stacked.

        Returns a flattened concatenation of the per-class mean vectors. Use
        `predict` / `predict_proba` for predictions; this property is mostly
        for diagnostic inspection.
        """
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
                "QDA has no closed-form linear-coefficient hypothesis. Use predict_proba() instead."
            )
        proba = self.predict_proba(X=X)
        return proba[:, 1]
