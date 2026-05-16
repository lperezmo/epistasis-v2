"""Power-transform nonlinear epistasis regression.

Implements the global-epistasis transform from:

    Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
    Genotype-Phenotype Maps.' Genetics 205, 1079-1088 (2017).

The transform is the Box-Cox-style power function centered on the geometric
mean of the additive (latent) phenotype:

.. math::
    y = \\frac{(x + A)^{\\lambda} - 1}{\\lambda \\cdot \\mathrm{GM}(x + A)^{\\lambda - 1}} + B

with the lambda -> 0 limit reducing to `GM * log(x + A) + B`. `A` is a
horizontal translation that keeps `x + A > 0` so the power is defined; `B`
absorbs vertical offset.

This file ports the v1 implementation to the v2 composition layout.
"""

from __future__ import annotations

from typing import Any

import lmfit
import numpy as np

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.models.nonlinear.minimizer import Minimizer
from epistasis.models.nonlinear.ordinary import EpistasisNonlinearRegression

__all__ = ["EpistasisPowerTransform", "PowerTransformMinimizer", "power_transform"]


def _geometric_mean(arr: np.ndarray) -> float:
    """Geometric mean of strictly-positive values.

    Returns `nan` if any entry is non-positive. The caller is responsible for
    ensuring the input is shifted so values stay positive (the `A` parameter
    of `power_transform` is what does this in practice).
    """
    a = np.asarray(arr, dtype=np.float64)
    if (a <= 0).any():
        return float("nan")
    return float(np.exp(np.log(a).mean()))


def power_transform(
    x: np.ndarray,
    lmbda: float,
    A: float,
    B: float,
    data: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the Box-Cox-style transform of Sailer & Harms 2017.

    Parameters
    ----------
    x
        Latent (additive) phenotype.
    lmbda
        Power parameter. `lmbda == 0` triggers the log limit.
    A
        Horizontal translation. Must satisfy `A > -min(x_centering + 0)` so the
        geometric-mean argument stays positive.
    B
        Vertical translation.
    data
        Optional reference array used to compute the geometric mean. Defaults
        to `x`. Passing the *training* additive phenotypes here when predicting
        on new data keeps the scale of the fit fixed (as in v1).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    centering = x_arr if data is None else np.asarray(data, dtype=np.float64)
    gm = _geometric_mean(centering + A)
    if not np.isfinite(gm):
        return np.full_like(x_arr, np.nan)
    if lmbda == 0:
        return gm * np.log(x_arr + A) + B
    first = (x_arr + A) ** lmbda
    out: np.ndarray = (first - 1.0) / (lmbda * gm ** (lmbda - 1.0)) + B
    return out


class PowerTransformMinimizer(Minimizer):
    """Lmfit-backed minimizer for the Sailer & Harms power transform.

    Holds three named parameters (`lmbda`, `A`, `B`) and a reference to the
    additive-phenotype array used to compute the geometric mean. The reference
    is set at `fit()` time and reused at `predict()` time so the transform is
    evaluated on the same scale during training and prediction.
    """

    parameters: lmfit.Parameters

    def __init__(
        self,
        lmbda: float | None = None,
        A: float | None = None,
        B: float | None = None,
    ) -> None:
        self.parameters = lmfit.Parameters()
        self.parameters.add(name="lmbda", value=1.0 if lmbda is None else float(lmbda))
        self.parameters.add(name="A", value=0.0 if A is None else float(A))
        self.parameters.add(name="B", value=0.0 if B is None else float(B))
        self._reference: np.ndarray | None = None
        self.last_result: lmfit.minimizer.MinimizerResult | None = None

    @property
    def param_names(self) -> list[str]:
        return ["lmbda", "A", "B"]

    def function(self, x: np.ndarray, *args: float) -> np.ndarray:
        if len(args) != 3:
            raise ValueError(f"Expected (lmbda, A, B); got {len(args)} positional args.")
        lmbda, A, B = args
        return power_transform(x, lmbda=lmbda, A=A, B=B, data=self._reference)

    def predict(self, x: np.ndarray) -> np.ndarray:
        vals = tuple(self.parameters[p].value for p in self.param_names)
        return self.function(x, *vals)

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ymodel = self.predict(x)
        return np.asarray((y - ymodel) + x, dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        # Reference scale is the *training* x; held constant for predictions.
        self._reference = x_arr.copy()
        # A must stay above -min(x) so x + A > 0 for the geometric mean.
        x_min = float(x_arr.min())
        self.parameters["A"].set(min=-x_min + 1e-12)

        last_residual: tuple[lmfit.Parameters, np.ndarray] | None = None

        def residual(
            params: lmfit.Parameters,
            xval: np.ndarray,
            yval: np.ndarray,
        ) -> np.ndarray:
            nonlocal last_residual
            vals = tuple(params[p].value for p in self.param_names)
            ymodel = self.function(xval, *vals)
            last_residual = (params, ymodel)
            r: np.ndarray = yval - ymodel
            return r

        try:
            result = lmfit.minimize(
                residual,
                self.parameters,
                args=(x_arr,),
                kws={"yval": y_arr},
            )
        except Exception as exc:
            if last_residual is not None:
                raise FittingError(
                    f"Power-transform fit failed: {exc}. "
                    f"Inspect the last attempted parameters: {dict(last_residual[0].valuesdict())}."
                ) from exc
            raise FittingError(f"Power-transform fit failed: {exc}") from exc

        self.last_result = result
        self.parameters = result.params


class EpistasisPowerTransform(EpistasisNonlinearRegression):
    """Two-stage power-transform global-epistasis model.

    Stage 1: fit an order-1 additive linear model to the observed phenotypes.
    Stage 2: fit the Box-Cox-style power transform of Sailer & Harms 2017 so
    that `power_transform(additive_phenotype, lmbda, A, B)` approximates the
    observed phenotypes.

    Parameters
    ----------
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    lmbda, A, B
        Optional initial guesses for the power-transform parameters. Default
        starting values are `(1.0, 0.0, 0.0)`.
    """

    def __init__(
        self,
        model_type: ModelType = "global",
        lmbda: float | None = None,
        A: float | None = None,
        B: float | None = None,
    ) -> None:
        # Bypass EpistasisNonlinearRegression.__init__: it expects a user-supplied
        # function and wires up a FunctionMinimizer. We slot in our own minimizer.
        from epistasis.models.base import EpistasisBaseModel

        EpistasisBaseModel.__init__(self, order=1, model_type=model_type)
        self.minimizer = PowerTransformMinimizer(lmbda=lmbda, A=A, B=B)
        self.additive = EpistasisLinearRegression(order=1, model_type=model_type)

    # The fit / predict / hypothesis methods are inherited from
    # EpistasisNonlinearRegression and operate via self.minimizer.

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predicted phenotype for `X` (and optionally a custom `thetas` vector).

        Concatenates `[lmbda, A, B]` and the linear coefficients, matching the
        `thetas` ordering exposed by `EpistasisNonlinearRegression.thetas`.
        Reuses the training-set additive scale for the geometric mean.
        """
        Xadd = self._additive_X(X)
        if thetas is None:
            if self.additive.thetas is None:
                raise FittingError("thetas unavailable; fit() first or pass thetas=.")
            nonlinear_vals = tuple(self.minimizer.parameters[p].value for p in ["lmbda", "A", "B"])
            linear_vals = self.additive.thetas
        else:
            thetas_arr = np.asarray(thetas, dtype=np.float64)
            if thetas_arr.shape[0] != self.num_of_params:
                raise FittingError(
                    f"Expected {self.num_of_params} thetas (lmbda, A, B + linear); "
                    f"got {thetas_arr.shape[0]}."
                )
            nonlinear_vals = (float(thetas_arr[0]), float(thetas_arr[1]), float(thetas_arr[2]))
            linear_vals = thetas_arr[3:]

        x_hat = np.asarray(Xadd @ linear_vals, dtype=np.float64)
        return self.minimizer.function(x_hat, *nonlinear_vals)
