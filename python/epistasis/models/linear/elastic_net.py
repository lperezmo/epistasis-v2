"""ElasticNet (mixed L1 + L2) epistasis regression."""

from __future__ import annotations

from sklearn.linear_model import ElasticNet

from epistasis.matrix import ModelType
from epistasis.models.linear._regularized import RegularizedLinearBase, SparseMode

__all__ = ["EpistasisElasticNet"]


class EpistasisElasticNet(RegularizedLinearBase):
    """ElasticNet over the epistasis design matrix.

    Combines L1 and L2 regularization: `l1_ratio=1.0` reduces to Lasso,
    `l1_ratio=0.0` reduces to Ridge, intermediate values mix both.

    Fixes a v1 bug where `l1_ratio` was silently overwritten to `1.0` in
    `__init__`, making the model behave as pure Lasso regardless of user
    input.

    Parameters
    ----------
    order
        Maximum interaction order to fit.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    alpha
        Combined penalty strength.
    l1_ratio
        In [0, 1]. Fraction of penalty that is L1; the remainder is L2.
    precompute
        Whether to use a precomputed Gram matrix. Ignored when the sparse
        path is active.
    max_iter
        Maximum coordinate-descent iterations.
    tol
        Convergence tolerance on the dual gap.
    warm_start
        Reuse the previous solution as initialization on repeated `fit`.
    positive
        Force coefficients to be non-negative.
    selection
        `"cyclic"` or `"random"`.
    random_state
        Seed used when `selection="random"`.
    sparse
        Build the design matrix as `scipy.sparse.csc_matrix`. `"auto"`
        (default) engages sparse only for `model_type="local"`. See
        `EpistasisLasso` for the rationale.
    """

    def __init__(
        self,
        order: int = 1,
        model_type: ModelType = "global",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        precompute: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        selection: str = "cyclic",
        random_state: int | None = None,
        sparse: SparseMode = "auto",
    ) -> None:
        super().__init__(order=order, model_type=model_type)
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1]; got {l1_ratio}.")
        self._sparse = sparse
        self._sklearn = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            copy_X=True,
            precompute=precompute,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection,
        )
        self.thetas = None
