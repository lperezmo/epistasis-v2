"""Lasso (L1) epistasis regression."""

from __future__ import annotations

from sklearn.linear_model import Lasso

from epistasis.matrix import ModelType
from epistasis.models.linear._regularized import RegularizedLinearBase, SparseMode

__all__ = ["EpistasisLasso"]


class EpistasisLasso(RegularizedLinearBase):
    """Lasso regression over the epistasis design matrix.

    L1-regularized linear regression. Drives coefficients to exactly zero,
    producing sparse solutions. Use `compression_ratio()` on a fitted model
    to inspect sparsity.

    Reference:
        Poelwijk, F. J., Socolich, M. & Ranganathan, R. 'Learning the
        pattern of epistasis linking genotype and phenotype in a protein.'
        bioRxiv (2017).

    Parameters
    ----------
    order
        Maximum interaction order to fit.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    alpha
        L1 penalty strength. `alpha = 0` is OLS (and triggers a sklearn
        warning; prefer `EpistasisLinearRegression`).
    precompute
        Whether to use a precomputed Gram matrix. Ignored when the sparse
        path is active (sklearn's coordinate-descent precompute is not used
        for sparse inputs).
    max_iter
        Maximum coordinate-descent iterations.
    tol
        Convergence tolerance on the dual gap.
    warm_start
        Reuse the previous solution as initialization on repeated `fit`.
    positive
        Force coefficients to be non-negative.
    selection
        `"cyclic"` or `"random"`. Random often converges faster when `tol`
        is loose.
    random_state
        Seed used when `selection="random"`.
    sparse
        Build the design matrix as `scipy.sparse.csc_matrix` and let
        sklearn's coordinate descent use the sparse fast path. `"auto"`
        (default) enables sparse only for `model_type="local"` because the
        global Hadamard encoding has no exploitable sparsity. Pass `True` /
        `False` to force the choice. The sparse path keeps the design matrix
        out of dense float64, which is the main blocker at `L >= 20`.
    """

    def __init__(
        self,
        order: int = 1,
        model_type: ModelType = "global",
        alpha: float = 1.0,
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
        self._sparse = sparse
        self._sklearn = Lasso(
            alpha=alpha,
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
