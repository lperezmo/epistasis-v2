"""Ordinary least-squares epistasis regression.

Reference:
    Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
    Genotype-Phenotype Maps.' Genetics 205, 1079-1088 (2017).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

from epistasis.exceptions import FittingError
from epistasis.fast import fwht_ols_coefficients
from epistasis.matrix import ModelType
from epistasis.models.base import EpistasisBaseModel

__all__ = ["EpistasisLinearRegression"]


class EpistasisLinearRegression(EpistasisBaseModel):
    """Ordinary least squares over the epistasis design matrix.

    Composition over sklearn: internally holds a `sklearn.linear_model
    .LinearRegression` configured for no extra intercept (the intercept lives
    in the design matrix as site `(0,)`) and forwards calls. Populates
    `self.epistasis.values` with the fitted coefficients and
    `self.epistasis.stdeviations` with the analytic OLS standard errors.

    Parameters
    ----------
    order
        Maximum interaction order to fit.
    model_type
        Encoding for the design matrix: `"global"` (Hadamard) or `"local"`
        (biochemical).
    n_jobs
        Forwarded to sklearn for parallel linear algebra.
    """

    def __init__(
        self,
        order: int = 1,
        model_type: ModelType = "global",
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(order=order, model_type=model_type)
        self._sklearn = LinearRegression(
            fit_intercept=False,
            copy_X=False,
            n_jobs=n_jobs,
            positive=False,
        )
        self.thetas: np.ndarray | None = None

    @property
    def coef_(self) -> np.ndarray:
        """Fitted coefficients. Mirror of `self.thetas`; kept for sklearn parity."""
        if self.thetas is None:
            raise FittingError("Model has not been fit yet.")
        return self.thetas

    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> EpistasisLinearRegression:
        y_arr = self._resolve_y(y)

        if X is None and self._gpm is not None:
            beta = fwht_ols_coefficients(
                self._gpm.binary_packed,
                y_arr,
                self.Xcolumns,
                model_type=self.model_type,
            )
            if beta is not None:
                self.thetas = beta
                if self._epistasis is not None:
                    self._epistasis.values = self.thetas
                    # Exactly-determined system (n == p): residuals are zero,
                    # no degrees of freedom for sigma^2, so stderr is undefined.
                    self._epistasis.stdeviations = np.full(
                        self.thetas.shape[0], np.nan, dtype=np.float64
                    )
                self._sync_sklearn_state(len(self.Xcolumns))
                return self

        X_mat = self._resolve_X(X).astype(np.float64)
        self._sklearn.fit(X_mat, y_arr)
        self.thetas = np.reshape(np.asarray(self._sklearn.coef_), (-1,))

        if self._epistasis is not None:
            self._epistasis.values = self.thetas
            self._epistasis.stdeviations = self._ols_stderr(X_mat, y_arr)

        return self

    def _sync_sklearn_state(self, n_features: int) -> None:
        """Populate sklearn's fitted-attribute contract without a full fit.

        Lets `predict` and `score` reach into `self._sklearn` after the FWHT
        fast path bypasses sklearn's own solver.
        """
        assert self.thetas is not None
        self._sklearn.coef_ = self.thetas.astype(np.float64, copy=True)
        self._sklearn.intercept_ = 0.0
        self._sklearn.n_features_in_ = n_features
        self._sklearn._residues = np.float64(0.0)
        self._sklearn.rank_ = n_features
        self._sklearn.singular_ = np.ones(n_features, dtype=np.float64)

    def predict(self, X: Any = None) -> np.ndarray:
        if self.thetas is None:
            raise FittingError("Call fit() before predict().")
        X_mat = self._resolve_X(X).astype(np.float64)
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
        X_mat = self._resolve_X(X).astype(np.float64)
        y_arr = self._resolve_y(y)
        return float(self._sklearn.score(X_mat, y_arr))

    def _ols_stderr(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Analytic OLS standard errors: `sqrt(diag(sigma_hat^2 * (X'X)^-1))`.

        Returns `np.nan` for every coefficient when the system is not
        overdetermined (n <= p) or `X.T @ X` is singular.
        """
        assert self.thetas is not None
        n, p = X.shape
        if n <= p:
            return np.full(p, np.nan, dtype=np.float64)

        resid = y - X @ self.thetas
        sigma2 = float(np.sum(resid**2) / (n - p))

        gram = X.T @ X
        try:
            cov = sigma2 * np.linalg.pinv(gram)
        except np.linalg.LinAlgError:
            return np.full(p, np.nan, dtype=np.float64)

        diag = np.clip(np.diag(cov), 0.0, None)
        return np.sqrt(diag).astype(np.float64)
