"""Function minimizers used by the nonlinear epistasis models."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable

import lmfit
import numpy as np

from epistasis.exceptions import FittingError

__all__ = ["FunctionMinimizer", "Minimizer"]


class Minimizer(ABC):
    """Abstract interface for function-based minimizers.

    Wraps a callable, fits it to (x, y) data, and exposes the fitted function
    for prediction and for linearizing observed phenotypes back onto the x
    scale via `transform`.
    """

    parameters: lmfit.Parameters

    @abstractmethod
    def function(self, x: np.ndarray, *args: float) -> np.ndarray:
        """Evaluate the underlying function with the supplied parameter values."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate with the currently-stored parameter values."""

    @abstractmethod
    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Linearize `y` onto the `x` scale using the current fit."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the function to `(x, y)`."""

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered names of the function's free parameters.

        Used by `EpistasisNonlinearRegression` to pack the fitted parameter
        values into the `thetas` vector.
        """


class FunctionMinimizer(Minimizer):
    """Wraps a user-provided Python function and fits it via `lmfit.minimize`.

    The callable must have signature `f(x, <param1>, <param2>, ...)`. Parameter
    names are introspected from the signature. Initial guesses can be supplied
    as a dict; parameters not listed there default to `1.0`.
    """

    def __init__(
        self,
        function: Callable[..., np.ndarray],
        initial_guesses: dict[str, float] | None = None,
    ) -> None:
        sig = inspect.signature(function)
        names = list(sig.parameters.keys())
        if not names or names[0] != "x":
            raise ValueError("First parameter of the nonlinear function must be named 'x'.")

        guesses = initial_guesses or {}
        self._function = function
        self._param_names = names[1:]

        self.parameters = lmfit.Parameters()
        for name in self._param_names:
            self.parameters.add(name=name, value=float(guesses.get(name, 1.0)))

        self.last_result: lmfit.minimizer.MinimizerResult | None = None

    @property
    def param_names(self) -> list[str]:
        return list(self._param_names)

    def function(self, x: np.ndarray, *args: float) -> np.ndarray:
        return np.asarray(self._function(x, *args), dtype=np.float64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        vals = [self.parameters[p].value for p in self._param_names]
        return np.asarray(self._function(x, *vals), dtype=np.float64)

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Linearize observed `y` onto the `x` scale: `(y - f(x)) + x`."""
        ymodel = self.predict(x)
        return np.asarray((y - ymodel) + x, dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        param_names = self._param_names
        func = self._function

        def residual(
            params: lmfit.Parameters,
            xval: np.ndarray,
            yval: np.ndarray,
        ) -> np.ndarray:
            vals = [params[p].value for p in param_names]
            ypred = np.asarray(func(xval, *vals), dtype=np.float64)
            return yval - ypred

        try:
            result = lmfit.minimize(
                residual,
                self.parameters,
                args=(x,),
                kws={"yval": y},
            )
        except Exception as exc:
            raise FittingError(f"Nonlinear fit failed: {exc}") from exc

        self.last_result = result
        self.parameters = result.params
