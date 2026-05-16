"""Shared machinery for regularized linear epistasis models.

Ridge, Lasso, and ElasticNet all follow the same composition pattern: hold a
configured sklearn estimator, forward `fit`/`predict`/`score`, copy the
resulting coefficients into `self.epistasis.values`. This module collects
that shared machinery.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
from scipy.sparse import csc_matrix, issparse

from epistasis.exceptions import FittingError, XMatrixError
from epistasis.matrix import get_model_matrix_sparse
from epistasis.models.base import EpistasisBaseModel
from epistasis.utils import genotypes_to_X

__all__ = ["RegularizedLinearBase", "SparseMode"]


SparseMode = Literal["auto"] | bool


class RegularizedLinearBase(EpistasisBaseModel):
    """Base for regularized linear epistasis models.

    Subclasses assign `self._sklearn` to a configured sklearn linear model in
    their `__init__`. The epistasis intercept lives in the design matrix as
    site `(0,)`, so all sklearn estimators must have `fit_intercept=False`.

    Subclasses that support sparse design matrices (Lasso, ElasticNet) set
    `self._sparse` to `True`, `False`, or `"auto"`. The base class then calls
    `_resolve_X_for_solver` instead of the dense `_resolve_X`. `Ridge` keeps
    `self._sparse = False` because `sklearn.linear_model.Ridge` densifies any
    sparse input internally anyway.
    """

    _sklearn: Any
    _sparse: SparseMode

    thetas: np.ndarray | None

    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> RegularizedLinearBase:
        X_mat = self._resolve_X_for_solver(X)
        y_arr = self._resolve_y(y)
        self._sklearn.fit(X_mat, y_arr)
        self.thetas = np.reshape(np.asarray(self._sklearn.coef_), (-1,))
        if self._epistasis is not None:
            self._epistasis.values = self.thetas
        return self

    def predict(self, X: Any = None) -> np.ndarray:
        if self.thetas is None:
            raise FittingError("Call fit() before predict().")
        X_mat = self._resolve_X_for_solver(X)
        return np.asarray(self._sklearn.predict(X_mat), dtype=np.float64)

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        X_mat = self._resolve_X_for_solver(X)
        th = thetas if thetas is not None else self.thetas
        if th is None:
            raise FittingError("thetas unavailable; fit() first or pass thetas=.")
        theta_arr = np.asarray(th, dtype=np.float64)
        out = X_mat @ theta_arr
        return np.asarray(out, dtype=np.float64)

    def score(self, X: Any = None, y: Any = None) -> float:
        if self.thetas is None:
            raise FittingError("Call fit() before score().")
        X_mat = self._resolve_X_for_solver(X)
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

    # ------------------------------------------------------------------
    # Sparse-aware design-matrix resolver.

    def _use_sparse(self) -> bool:
        """Resolve `self._sparse` against the current `model_type`.

        `"auto"` enables the sparse path only for `local` encoding, where the
        per-site product columns are 0/1 and the matrix has real exploitable
        sparsity. `global` (Hadamard) encoding has every entry equal to `+/-1`,
        so a CSC representation would be strictly larger than the dense form.
        """
        sparse = getattr(self, "_sparse", False)
        if sparse == "auto":
            return self.model_type == "local"
        return bool(sparse)

    def _resolve_X_for_solver(self, X: Any) -> Any:
        """Return a design matrix suitable for the sklearn estimator.

        Dispatches to dense `_resolve_X` when `_use_sparse()` is False; in that
        case the returned array is fortran-ordered float64 to match the path
        the regularized models used previously.

        When `_use_sparse()` is True, builds a `scipy.sparse.csc_matrix`
        directly (no dense materialization for `local` encoding). Accepts a
        pre-built sparse matrix or genotype iterable as input.
        """
        if not self._use_sparse():
            return np.asfortranarray(self._resolve_X(X).astype(np.float64))

        if X is None:
            cached = self._Xbuilt.get("sparse")
            if cached is not None:
                return cached
            mat = get_model_matrix_sparse(
                self.gpm.binary_packed,
                self.Xcolumns,
                model_type=self.model_type,
            )
            self._Xbuilt["sparse"] = mat
            return mat

        if issparse(X):
            return X.tocsc() if X.format != "csc" else X

        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                return csc_matrix(X.astype(np.float64, copy=False))
            if X.ndim == 1:
                return self._genotypes_to_sparse(list(X))
            raise XMatrixError(f"X must be 1D (genotypes) or 2D (matrix); got ndim={X.ndim}.")

        if isinstance(X, Iterable):
            return self._genotypes_to_sparse(list(X))

        raise XMatrixError(f"Unrecognized type for X: {type(X).__name__}.")

    def _genotypes_to_sparse(self, strings: list[Any]) -> csc_matrix:
        if not all(isinstance(s, str) for s in strings):
            raise XMatrixError("All entries of X must be genotype strings.")
        dense = genotypes_to_X(
            strings,
            self.gpm,
            order=self.order,
            model_type=self.model_type,
        )
        return csc_matrix(dense.astype(np.float64, copy=False))
