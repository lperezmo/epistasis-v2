"""Two-stage linear + nonlinear epistasis regression.

Stage 1: fit an order-1 additive linear model to the observed phenotypes.
Stage 2: fit a user-supplied nonlinear function of the additive phenotype to
the observed phenotypes.

Reference:
    Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
    Genotype-Phenotype Maps.' Genetics 205, 1079-1088 (2017).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.base import EpistasisBaseModel
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.models.nonlinear.minimizer import FunctionMinimizer, Minimizer

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap

__all__ = ["EpistasisNonlinearRegression"]


class EpistasisNonlinearRegression(EpistasisBaseModel):
    """Linear additive model composed with a user-supplied nonlinear scale.

    Fits in two stages:

    1. `Additive` (an `EpistasisLinearRegression` at order 1) is fit to the
       observed phenotypes.
    2. The user-provided nonlinear function `f(x, *params)` is fit so that
       `f(Additive.predict(X))` approximates the observed phenotypes. Fitting
       is done by `lmfit.minimize` (Levenberg-Marquardt by default).

    Predictions compose the two stages: `y = f(Additive.predict(X), *params)`.

    Parameters
    ----------
    function
        Callable `f(x, *params)` where `x` is first. Parameter names are read
        from the signature.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    initial_guesses
        Dict mapping nonlinear-parameter name to its starting value. Parameters
        not listed default to `1.0`.
    """

    minimizer: Minimizer

    def __init__(
        self,
        function: Callable[..., np.ndarray],
        model_type: ModelType = "global",
        initial_guesses: dict[str, float] | None = None,
    ) -> None:
        super().__init__(order=1, model_type=model_type)
        self.minimizer = FunctionMinimizer(function, initial_guesses=initial_guesses)
        self.additive = EpistasisLinearRegression(order=1, model_type=model_type)

    # --------------------------------------------------------------
    # Setup.

    def add_gpm(self, gpm: GenotypePhenotypeMap) -> EpistasisNonlinearRegression:
        super().add_gpm(gpm)
        self.additive.add_gpm(gpm)
        return self

    @property
    def parameters(self) -> Any:
        """Lmfit `Parameters` object holding the fitted nonlinear parameters."""
        return self.minimizer.parameters

    @property
    def thetas(self) -> np.ndarray:
        """Concatenation: nonlinear parameters followed by linear coefficients."""
        if self.additive.thetas is None:
            raise FittingError("Call fit() before reading thetas.")
        nonlinear = np.asarray(
            [self.minimizer.parameters[p].value for p in self.minimizer.param_names],
            dtype=np.float64,
        )
        return np.concatenate([nonlinear, self.additive.thetas])

    @property
    def num_of_params(self) -> int:
        return len(self.minimizer.param_names) + len(self.additive.Xcolumns)

    # --------------------------------------------------------------
    # Fit.

    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> EpistasisNonlinearRegression:
        y_arr = self._resolve_y(y)
        Xadd = self._additive_X(X)

        self.additive.fit(X=Xadd, y=y_arr)
        x_hat = self.additive.predict(X=Xadd)

        self.minimizer.fit(x_hat, y_arr)
        return self

    # --------------------------------------------------------------
    # Predict / hypothesis.

    def predict(self, X: Any = None) -> np.ndarray:
        if self.additive.thetas is None:
            raise FittingError("Call fit() before predict().")
        Xadd = self._additive_X(X)
        x_hat = self.additive.predict(X=Xadd)
        return self.minimizer.predict(x_hat)

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        Xadd = self._additive_X(X)
        if thetas is None:
            if self.additive.thetas is None:
                raise FittingError("thetas unavailable; fit() first or pass thetas=.")
            nonlinear_vals = np.asarray(
                [self.minimizer.parameters[p].value for p in self.minimizer.param_names],
                dtype=np.float64,
            )
            linear_vals = self.additive.thetas
        else:
            thetas_arr = np.asarray(thetas, dtype=np.float64)
            n_nonlinear = len(self.minimizer.param_names)
            if thetas_arr.shape[0] != self.num_of_params:
                raise FittingError(
                    f"Expected {self.num_of_params} thetas (nonlinear + linear); "
                    f"got {thetas_arr.shape[0]}."
                )
            nonlinear_vals = thetas_arr[:n_nonlinear]
            linear_vals = thetas_arr[n_nonlinear:]

        x_hat = np.asarray(Xadd @ linear_vals, dtype=np.float64)
        return self.minimizer.function(x_hat, *nonlinear_vals)

    def transform(self, X: Any = None, y: Any = None) -> np.ndarray:
        """Linearize observed `y` onto the additive-phenotype scale.

        Returns `(y - f(x_hat)) + x_hat`, the observed phenotypes rendered on
        the same linear scale as the additive model's predictions.
        """
        y_arr = self._resolve_y(y)
        Xadd = self._additive_X(X)
        x_hat = self.additive.predict(X=Xadd)
        return self.minimizer.transform(x_hat, y_arr)

    def score(self, X: Any = None, y: Any = None) -> float:
        """Pearson R^2 between observed and predicted phenotypes."""
        if self.additive.thetas is None:
            raise FittingError("Call fit() before score().")
        y_arr = self._resolve_y(y)
        y_pred = self.predict(X=X)
        if y_arr.size < 2:
            return float("nan")
        corr = np.corrcoef(y_arr, y_pred)[0, 1]
        return float(corr**2)

    # --------------------------------------------------------------
    # Helpers.

    def _additive_X(self, X: Any) -> np.ndarray:
        """Return the order-1 design matrix whether X is None, genotypes, or a
        2D matrix that may already include higher-order columns.

        If X is 2D and has more columns than the additive model expects, the
        first `len(additive.Xcolumns)` columns are used. This matches the
        ordering produced by `encoding_to_sites` (intercept, first order,
        higher orders) so slicing off the tail is correct.
        """
        if isinstance(X, np.ndarray) and X.ndim == 2:
            width = len(self.additive.Xcolumns)
            if X.shape[1] < width:
                raise FittingError(
                    f"Design matrix has {X.shape[1]} columns but the additive model needs {width}."
                )
            return X[:, :width]
        return self.additive._resolve_X(X)
