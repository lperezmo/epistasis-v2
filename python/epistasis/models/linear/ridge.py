"""Ridge (L2) epistasis regression."""

from __future__ import annotations

from sklearn.linear_model import Ridge

from epistasis.matrix import ModelType
from epistasis.models.linear._regularized import RegularizedLinearBase

__all__ = ["EpistasisRidge"]


class EpistasisRidge(RegularizedLinearBase):
    """Ridge regression over the epistasis design matrix.

    L2-regularized linear regression. Shrinks coefficients toward zero without
    producing exact zeros; use `EpistasisLasso` or `EpistasisElasticNet` for
    sparse solutions.

    Parameters
    ----------
    order
        Maximum interaction order to fit.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    alpha
        L2 penalty strength. `alpha = 0` is OLS.
    max_iter
        Maximum iterations for the iterative solvers.
    tol
        Convergence tolerance.
    solver
        sklearn Ridge solver: `"auto"`, `"svd"`, `"cholesky"`, `"lsqr"`,
        `"sparse_cg"`, `"sag"`, `"saga"`, `"lbfgs"`.
    random_state
        Seed for solvers that use randomness.
    """

    def __init__(
        self,
        order: int = 1,
        model_type: ModelType = "global",
        alpha: float = 1.0,
        max_iter: int | None = None,
        tol: float = 1e-4,
        solver: str = "auto",
        random_state: int | None = None,
    ) -> None:
        super().__init__(order=order, model_type=model_type)
        self._sklearn = Ridge(
            alpha=alpha,
            fit_intercept=False,
            copy_X=True,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state,
            positive=False,
        )
        self.thetas = None
