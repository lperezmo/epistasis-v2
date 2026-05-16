"""Spline-based nonlinear epistasis regression.

Stage 1 fits an order-1 additive linear model. Stage 2 fits a smoothing
spline mapping the additive (latent) phenotype onto the observed phenotype.
Useful when the nonlinearity is not well-approximated by the power transform
of `EpistasisPowerTransform` but is still smooth and (ideally) monotonic.

Note that `scipy.interpolate.UnivariateSpline` does not enforce monotonicity.
For strictly monotonic global-epistasis fits, consider the monotone-spline
formulation used in MAVE-NN or similar packages; this class is a faithful
port of the v1 spline fitter and matches its caveats.
"""

from __future__ import annotations

from typing import Any

import lmfit
import numpy as np
from scipy.interpolate import UnivariateSpline

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.models.nonlinear.minimizer import Minimizer
from epistasis.models.nonlinear.ordinary import EpistasisNonlinearRegression

__all__ = ["EpistasisSpline", "SplineMinimizer"]


class SplineMinimizer(Minimizer):
    """Smoothing-spline minimizer over the additive phenotype.

    Fits a `scipy.interpolate.UnivariateSpline` of order `k` (default cubic)
    with smoothness factor `s`. `parameters` exposes the spline's first
    `k + 1` coefficients as lmfit parameters for inspection; the spline
    itself is the source of truth for `predict()`.

    The legacy implementation added small random jitter to break ties in `x`
    before sorting; we keep that behavior but make it deterministic via a
    seeded RNG.
    """

    parameters: lmfit.Parameters

    def __init__(
        self,
        k: int = 3,
        s: float | None = None,
        seed: int | None = 0,
    ) -> None:
        if not 1 <= k <= 5:
            raise ValueError(f"Spline degree k must be in [1, 5]; got {k}.")
        self.k = int(k)
        self.s = s
        self._rng = np.random.default_rng(seed)
        self.parameters = lmfit.Parameters()
        for i in range(self.k + 1):
            self.parameters.add(name=f"c{i}", value=0.0)
        self._spline: UnivariateSpline | None = None

    @property
    def param_names(self) -> list[str]:
        return [f"c{i}" for i in range(self.k + 1)]

    def _sorter(
        self, x: np.ndarray, y: np.ndarray | None = None, tol: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Sort `x` ascending. Inject jitter into duplicates so the spline
        sees strictly-increasing knots.
        """
        x_arr = np.array(x, dtype=np.float64, copy=True)
        _, unique_idx = np.unique(x_arr, return_index=True)
        dup_idx = np.delete(np.arange(len(x_arr)), unique_idx)
        if dup_idx.size:
            x_arr[dup_idx] = x_arr[dup_idx] + self._rng.uniform(1.0, 9.9, size=dup_idx.size) * tol
        order = np.argsort(x_arr)
        if y is None:
            return x_arr[order]
        return x_arr[order], np.asarray(y, dtype=np.float64)[order]

    def function(self, x: np.ndarray, *coefs: float) -> np.ndarray:
        """Evaluate a polynomial-form spline from `coefs`.

        Used by likelihood / hypothesis routines that want to override the
        fitted coefficients with a candidate `thetas`. For ordinary predict
        on the fitted model, prefer `predict(x)` which uses the underlying
        scipy spline directly.
        """
        if len(coefs) != self.k + 1:
            raise ValueError(f"Expected {self.k + 1} coefficients; got {len(coefs)}.")
        x_arr = np.asarray(x, dtype=np.float64)
        n = self.k + 1
        t_arr = np.zeros(n * 2, dtype=np.float64)
        t_arr[:n] = float(x_arr.min())
        t_arr[n:] = float(x_arr.max())
        c_arr = np.zeros(n * 2, dtype=np.float64)
        c_arr[:n] = coefs
        model = UnivariateSpline._from_tck((t_arr, c_arr, self.k))
        return np.asarray(model(x_arr), dtype=np.float64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._spline is None:
            raise FittingError("Call fit() before predict().")
        return np.asarray(self._spline(np.asarray(x, dtype=np.float64)), dtype=np.float64)

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ymodel = self.predict(x)
        return np.asarray((y - ymodel) + x, dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        result = self._sorter(x, y)
        assert isinstance(result, tuple)
        x_sorted, y_sorted = result
        try:
            spline = UnivariateSpline(x=x_sorted, y=y_sorted, k=self.k, s=self.s)
        except Exception as exc:
            raise FittingError(f"Spline fit failed: {exc}") from exc
        self._spline = spline
        coefs = list(spline.get_coeffs())
        for i, coef in enumerate(coefs):
            name = f"c{i}"
            if name in self.parameters:
                self.parameters[name].value = float(coef)
            else:
                raise FittingError(
                    "UnivariateSpline fitting returned more coefficients than "
                    f"k+1 = {self.k + 1}, likely because knots were added to "
                    "fit the data closer. Increase the smoothing factor `s` "
                    "or reduce `k`."
                )


class EpistasisSpline(EpistasisNonlinearRegression):
    """Two-stage spline global-epistasis model.

    Parameters
    ----------
    k
        Spline degree (1 = linear, 3 = cubic, max 5).
    s
        Smoothing factor passed to `scipy.interpolate.UnivariateSpline`. `None`
        triggers the scipy default heuristic. Larger `s` = smoother fit.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    seed
        Seed for the small jitter applied to duplicate `x` values before the
        spline fit. Pass `None` for non-deterministic jitter.
    """

    def __init__(
        self,
        k: int = 3,
        s: float | None = None,
        model_type: ModelType = "global",
        seed: int | None = 0,
    ) -> None:
        from epistasis.models.base import EpistasisBaseModel

        EpistasisBaseModel.__init__(self, order=1, model_type=model_type)
        self.k = int(k)
        self.s = s
        self.minimizer = SplineMinimizer(k=self.k, s=self.s, seed=seed)
        self.additive = EpistasisLinearRegression(order=1, model_type=model_type)

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predicted phenotype, with optional override of spline coefficients.

        `thetas` is interpreted as `[c_0, ..., c_k] + linear_coefs`.
        """
        Xadd = self._additive_X(X)
        if thetas is None:
            if self.additive.thetas is None:
                raise FittingError("thetas unavailable; fit() first or pass thetas=.")
            x_hat = np.asarray(Xadd @ self.additive.thetas, dtype=np.float64)
            return self.minimizer.predict(x_hat)

        thetas_arr = np.asarray(thetas, dtype=np.float64)
        n_nonlinear = self.k + 1
        if thetas_arr.shape[0] != self.num_of_params:
            raise FittingError(
                f"Expected {self.num_of_params} thetas (k+1 spline coefs + linear); "
                f"got {thetas_arr.shape[0]}."
            )
        coefs = tuple(float(v) for v in thetas_arr[:n_nonlinear])
        linear_vals = thetas_arr[n_nonlinear:]
        x_hat = np.asarray(Xadd @ linear_vals, dtype=np.float64)
        return self.minimizer.function(x_hat, *coefs)
