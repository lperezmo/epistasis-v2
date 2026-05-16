"""Gaussian-process classifier for epistasis viability prediction.

Wraps `sklearn.gaussian_process.GaussianProcessClassifier`. GP classifiers
place a GP prior over a latent function `f(x)`, then squash it through a
logistic link to get class probabilities. Useful when the viability surface
is smooth but not linear in the projected design matrix.

References
----------
Rasmussen, C. E. & Williams, C. K. I. 'Gaussian Processes for Machine
Learning.' MIT Press (2006). Chapter 3 covers GP classification.
"""

from __future__ import annotations

from typing import Any

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Kernel

from epistasis.exceptions import FittingError
from epistasis.matrix import ModelType
from epistasis.models.classifiers._base import EpistasisClassifierBase

__all__ = ["EpistasisGaussianProcess"]


class EpistasisGaussianProcess(EpistasisClassifierBase):
    """Gaussian-process classifier over the additive-projected design matrix.

    Parameters
    ----------
    threshold
        Phenotype cut-off; class 1 is `> threshold`.
    model_type
        Encoding for the design matrix (`"global"` or `"local"`).
    kernel
        Covariance kernel. Defaults to `RBF()`.
    n_restarts_optimizer
        Number of kernel-hyperparameter optimizer restarts.
    max_iter_predict
        Max iterations for the Laplace approximation in `predict_proba`.
    random_state
        Seed for the kernel optimizer.
    """

    def __init__(
        self,
        threshold: float,
        model_type: ModelType = "global",
        kernel: Kernel | None = None,
        n_restarts_optimizer: int = 0,
        max_iter_predict: int = 100,
        random_state: int | None = None,
    ) -> None:
        super().__init__(threshold=threshold, model_type=model_type)
        self._sklearn = GaussianProcessClassifier(
            kernel=kernel if kernel is not None else RBF(),
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            random_state=random_state,
        )

    @property
    def thetas(self) -> Any:
        if not hasattr(self._sklearn, "base_estimator_"):
            raise FittingError("Model has not been fit yet.")
        return self._sklearn.base_estimator_.kernel_.theta
