"""Monotonic global-epistasis (GE) regression.

Modern alternative to the v1 power-transform and unconstrained spline. Models
the latent-to-observed nonlinearity as a sum of hyperbolic-tangent sigmoids
following Tareen et al. 2022 (MAVE-NN):

.. math::
    g(\\phi; \\alpha) = a + \\sum_{k=1}^{K} b_k \\tanh(c_k \\phi + d_k)

Monotonicity is enforced by constraining `b_k >= 0` and `c_k >= 0` (the
default), guaranteeing the latent phenotype scale is identifiable.

References
----------
Tareen, A., Kooshkbaghi, M., Posfai, A., Ireland, W. T., McCandlish, D. M.,
& Kinney, J. B. 'MAVE-NN: learning genotype-phenotype maps from multiplex
assays of variant effect.' Genome Biology 23, 98 (2022).
https://doi.org/10.1186/s13059-022-02661-7

Ramsay, J. O. 'Monotone Regression Splines in Action.' Statistical Science
3, 425-441 (1988). [foundational reference for monotone-spline ideas in
nonlinear regression; MAVE-NN's tanh-sum is a smooth alternative that
shares the same monotonicity-by-construction property]
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

__all__ = ["EpistasisMonotonicGE", "MonotonicGEMinimizer", "monotonic_ge"]


def monotonic_ge(
    x: np.ndarray,
    a: float,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Evaluate the MAVE-NN GE nonlinearity at `x`.

    Parameters
    ----------
    x
        Latent (additive) phenotype values.
    a
        Scalar offset.
    b, c, d
        Length-K arrays of tanh-sum parameters. Monotonicity in `x` requires
        all entries of `b` and `c` to be non-negative.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64)
    # Broadcast: x is (N,), b/c/d are (K,)
    z = c_arr[None, :] * x_arr[:, None] + d_arr[None, :]
    out: np.ndarray = float(a) + np.sum(b_arr[None, :] * np.tanh(z), axis=1)
    return out


class MonotonicGEMinimizer(Minimizer):
    """Lmfit-backed minimizer for the tanh-sum GE nonlinearity.

    Parameters
    ----------
    K
        Number of tanh hidden nodes. Larger `K` allows more curvature.
    monotonic
        If True (default), constrains `b_k, c_k >= 0` so `g` is monotonic.
    seed
        Seed for randomized initial guesses of `(b_k, c_k, d_k)`. Random
        initialization helps the local optimizer explore different sigmoid
        positions instead of all collapsing on top of each other.
    """

    parameters: lmfit.Parameters

    def __init__(self, K: int = 5, monotonic: bool = True, seed: int | None = 0) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}.")
        self.K = int(K)
        self.monotonic = bool(monotonic)
        self.parameters = lmfit.Parameters()
        rng = np.random.default_rng(seed)
        self.parameters.add(name="a", value=0.0)
        bmin = 0.0 if self.monotonic else None
        cmin = 0.0 if self.monotonic else None
        for k in range(self.K):
            self.parameters.add(name=f"b{k}", value=float(rng.uniform(0.1, 1.0)), min=bmin)
            self.parameters.add(name=f"c{k}", value=float(rng.uniform(0.1, 1.0)), min=cmin)
            self.parameters.add(name=f"d{k}", value=float(rng.uniform(-1.0, 1.0)))
        self.last_result: lmfit.minimizer.MinimizerResult | None = None

    @property
    def param_names(self) -> list[str]:
        names = ["a"]
        for k in range(self.K):
            names += [f"b{k}", f"c{k}", f"d{k}"]
        return names

    def _unpack(self, params: lmfit.Parameters) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        a = float(params["a"].value)
        b = np.fromiter((params[f"b{k}"].value for k in range(self.K)), dtype=np.float64)
        c = np.fromiter((params[f"c{k}"].value for k in range(self.K)), dtype=np.float64)
        d = np.fromiter((params[f"d{k}"].value for k in range(self.K)), dtype=np.float64)
        return a, b, c, d

    def function(self, x: np.ndarray, *args: float) -> np.ndarray:
        """Evaluate the tanh-sum nonlinearity.

        Positional args are interleaved to match `param_names` ordering:
        `(a, b0, c0, d0, b1, c1, d1, ...)`. This matches the order in which
        `EpistasisNonlinearRegression.thetas` packs and unpacks parameters.
        """
        expected = 1 + 3 * self.K
        if len(args) != expected:
            raise ValueError(
                f"Expected {expected} positional args (a, b0, c0, d0, ..., bK, cK, dK); "
                f"got {len(args)}."
            )
        a = float(args[0])
        b = np.fromiter((args[1 + 3 * k] for k in range(self.K)), dtype=np.float64)
        c = np.fromiter((args[2 + 3 * k] for k in range(self.K)), dtype=np.float64)
        d = np.fromiter((args[3 + 3 * k] for k in range(self.K)), dtype=np.float64)
        return monotonic_ge(x, a, b, c, d)

    def predict(self, x: np.ndarray) -> np.ndarray:
        a, b, c, d = self._unpack(self.parameters)
        return monotonic_ge(x, a, b, c, d)

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ymodel = self.predict(x)
        return np.asarray((y - ymodel) + x, dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # Seed `a` and `b` from the observed phenotype scale so the optimizer
        # starts in the right ballpark. This matches MAVE-NN's internal
        # rescaling of the GE nonlinearity to mean(y), std(y).
        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr))
        self.parameters["a"].set(value=y_mean)
        scale_per_node = y_std / max(self.K, 1)
        for k in range(self.K):
            self.parameters[f"b{k}"].set(value=max(scale_per_node, 1e-3))

        def residual(params: lmfit.Parameters, xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
            a, b, c, d = self._unpack(params)
            r: np.ndarray = yv - monotonic_ge(xv, a, b, c, d)
            return r

        try:
            result = lmfit.minimize(
                residual,
                self.parameters,
                args=(x_arr, y_arr),
            )
        except Exception as exc:
            raise FittingError(f"Monotonic-GE fit failed: {exc}") from exc

        self.last_result = result
        self.parameters = result.params


class EpistasisMonotonicGE(EpistasisNonlinearRegression):
    """Two-stage monotonic global-epistasis model (MAVE-NN style).

    Stage 1 fits an order-1 additive linear model. Stage 2 fits the tanh-sum
    nonlinearity `g(phi; a, b, c, d)` of Tareen et al. 2022 onto the additive
    phenotype. With `monotonic=True` (default), the inferred `g` is guaranteed
    monotonic by construction, which fixes the sign ambiguity that otherwise
    plagues GE fits.

    Parameters
    ----------
    K
        Number of tanh hidden nodes in the GE nonlinearity. `K=5` is a
        reasonable default that captures most sigmoidal shapes.
    monotonic
        Enforce `b_k, c_k >= 0`. Set False if you suspect a non-monotonic
        scale (rare in biological GE).
    model_type
        Encoding for the additive design matrix (`"global"` or `"local"`).
    seed
        Seed for the randomized initial guesses on `(b_k, c_k, d_k)`. Pass
        `None` for non-deterministic initialization.
    """

    def __init__(
        self,
        K: int = 5,
        monotonic: bool = True,
        model_type: ModelType = "global",
        seed: int | None = 0,
    ) -> None:
        from epistasis.models.base import EpistasisBaseModel

        EpistasisBaseModel.__init__(self, order=1, model_type=model_type)
        self.K = int(K)
        self.monotonic = bool(monotonic)
        self.minimizer = MonotonicGEMinimizer(K=self.K, monotonic=self.monotonic, seed=seed)
        self.additive = EpistasisLinearRegression(order=1, model_type=model_type)

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        Xadd = self._additive_X(X)
        if thetas is None:
            if self.additive.thetas is None:
                raise FittingError("thetas unavailable; fit() first or pass thetas=.")
            x_hat = np.asarray(Xadd @ self.additive.thetas, dtype=np.float64)
            return self.minimizer.predict(x_hat)

        thetas_arr = np.asarray(thetas, dtype=np.float64)
        n_nonlinear = 1 + 3 * self.K
        if thetas_arr.shape[0] != self.num_of_params:
            raise FittingError(
                f"Expected {self.num_of_params} thetas (1 + 3*K nonlinear + linear); "
                f"got {thetas_arr.shape[0]}."
            )
        nonlinear_vals = tuple(float(v) for v in thetas_arr[:n_nonlinear])
        linear_vals = thetas_arr[n_nonlinear:]
        x_hat = np.asarray(Xadd @ linear_vals, dtype=np.float64)
        return self.minimizer.function(x_hat, *nonlinear_vals)
