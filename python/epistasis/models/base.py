"""Abstract interface and concrete foundation for epistasis models.

Replaces v1's `@use_sklearn` MRO injection with plain composition. Concrete
models (linear, nonlinear, classifier) hold a sklearn/lmfit estimator as an
attribute and forward calls explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from epistasis.exceptions import XMatrixError
from epistasis.mapping import EpistasisMap, Site, encoding_to_sites
from epistasis.matrix import ModelType, get_model_matrix
from epistasis.utils import genotypes_to_X

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap

__all__ = ["AbstractEpistasisModel", "EpistasisBaseModel"]


class AbstractEpistasisModel(ABC):
    """Minimal interface every epistasis model must implement."""

    order: int
    model_type: ModelType

    @abstractmethod
    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> AbstractEpistasisModel:
        """Fit the model; return self."""

    @abstractmethod
    def predict(self, X: Any = None) -> np.ndarray:
        """Predict phenotypes for the given genotypes or design matrix."""

    @abstractmethod
    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict phenotypes for a specific parameter vector (without fitting)."""


class EpistasisBaseModel(AbstractEpistasisModel):
    """Concrete foundation shared by all epistasis model subclasses.

    Owns the GPM wiring, EpistasisMap, design-matrix cache, and argument
    resolvers. Subclasses implement `fit`, `predict`, `hypothesis`.
    """

    order: int = 1
    model_type: ModelType = "global"

    def __init__(self, order: int = 1, model_type: ModelType = "global") -> None:
        self.order = order
        self.model_type = model_type
        self._gpm: GenotypePhenotypeMap | None = None
        self._epistasis: EpistasisMap | None = None
        self._Xcolumns: list[Site] | None = None
        self._Xbuilt: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # GPM attach / accessors.

    def add_gpm(self, gpm: GenotypePhenotypeMap) -> EpistasisBaseModel:
        """Attach a GenotypePhenotypeMap and build derived structures."""
        self._gpm = gpm
        self._Xbuilt = {}
        self._Xcolumns = encoding_to_sites(self.order, gpm.encoding_table)
        self._epistasis = EpistasisMap(sites=self._Xcolumns, gpm=gpm)
        return self

    @property
    def gpm(self) -> GenotypePhenotypeMap:
        if self._gpm is None:
            raise XMatrixError("No GenotypePhenotypeMap attached; call add_gpm(gpm) first.")
        return self._gpm

    @property
    def epistasis(self) -> EpistasisMap:
        if self._epistasis is None:
            raise XMatrixError("No EpistasisMap built; call add_gpm(gpm) first.")
        return self._epistasis

    @property
    def Xcolumns(self) -> list[Site]:
        if self._Xcolumns is None:
            raise XMatrixError("No Xcolumns built; call add_gpm(gpm) first.")
        return self._Xcolumns

    @property
    def num_of_params(self) -> int:
        """Default: one parameter per site column. Subclasses override as needed."""
        return len(self.Xcolumns)

    # ------------------------------------------------------------------
    # Argument resolvers.

    def _resolve_X(self, X: Any) -> np.ndarray:
        """Coerce `X` into a 2D design matrix.

        Accepts:
        - None: cached GPM-derived design matrix (built on first access).
        - 2D numpy array: treated as a design matrix and returned as-is.
        - 1D iterable of genotype strings: built via `genotypes_to_X`.
        """
        if X is None:
            if "default" in self._Xbuilt:
                return self._Xbuilt["default"]
            mat = get_model_matrix(
                self.gpm.binary_packed, self.Xcolumns, model_type=self.model_type
            )
            self._Xbuilt["default"] = mat
            return mat

        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                return X
            if X.ndim == 1:
                return self._genotypes_to_matrix(list(X))
            raise XMatrixError(f"X must be 1D (genotypes) or 2D (matrix); got ndim={X.ndim}.")

        if isinstance(X, Iterable):
            return self._genotypes_to_matrix(list(X))

        raise XMatrixError(f"Unrecognized type for X: {type(X).__name__}.")

    def _genotypes_to_matrix(self, strings: list[Any]) -> np.ndarray:
        if not all(isinstance(s, str) for s in strings):
            raise XMatrixError("All entries of X must be genotype strings.")
        return genotypes_to_X(
            strings,
            self.gpm,
            order=self.order,
            model_type=self.model_type,
        )

    def _resolve_y(self, y: Any) -> np.ndarray:
        if y is None:
            return np.asarray(self.gpm.phenotypes, dtype=np.float64)
        return np.asarray(y, dtype=np.float64)

    def _resolve_yerr(self, yerr: Any) -> np.ndarray:
        if yerr is None:
            return np.asarray(self.gpm.stdeviations, dtype=np.float64)
        return np.asarray(yerr, dtype=np.float64)

    # ------------------------------------------------------------------
    # Likelihood helpers (Gaussian default; override for non-Gaussian noise).

    def lnlike_of_data(
        self,
        X: Any = None,
        y: Any = None,
        yerr: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-datapoint Gaussian log-likelihood.

        Uses `self.hypothesis(X, thetas)` as the mean and `yerr` as the
        standard deviation. Subclasses with non-Gaussian noise override.
        """
        X_mat = self._resolve_X(X)
        y_arr = self._resolve_y(y)
        yerr_arr = self._resolve_yerr(yerr)
        mean = self.hypothesis(X=X_mat, thetas=thetas)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = (y_arr - mean) / yerr_arr
            out: np.ndarray = -0.5 * resid**2 - 0.5 * np.log(2.0 * np.pi * yerr_arr**2)
        return out

    def lnlikelihood(
        self,
        X: Any = None,
        y: Any = None,
        yerr: Any = None,
        thetas: np.ndarray | None = None,
    ) -> float:
        """Sum of per-datapoint log-likelihoods. Non-finite collapses to `-inf`."""
        total = float(np.sum(self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas)))
        if not np.isfinite(total):
            return float("-inf")
        return total

    # ------------------------------------------------------------------
    # Prediction output helpers.

    def predict_to_df(self, X: Any = None) -> pd.DataFrame:
        """Predict and return a DataFrame with `genotypes` and `phenotypes` columns.

        `X` may be `None` (uses the attached GPM's genotypes) or an iterable of
        genotype strings. Passing a design matrix is rejected because we lose
        the genotype labels.
        """
        if X is None:
            genotypes = list(self.gpm.genotypes)
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            raise XMatrixError(
                "predict_to_df requires genotype strings; design matrices lack labels."
            )
        else:
            genotypes = list(X)
            if not all(isinstance(g, str) for g in genotypes):
                raise XMatrixError("All entries of X must be genotype strings.")

        y = self.predict(X=genotypes)
        return pd.DataFrame({"genotypes": genotypes, "phenotypes": y})

    def predict_to_csv(self, filename: str, X: Any = None) -> None:
        self.predict_to_df(X=X).to_csv(filename, index=False)

    def predict_to_excel(self, filename: str, X: Any = None) -> None:
        self.predict_to_df(X=X).to_excel(filename, index=False)
