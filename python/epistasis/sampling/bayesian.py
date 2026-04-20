"""Emcee-driven Bayesian sampling for epistasis model parameters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import emcee
import numpy as np

__all__ = ["BayesianSampler", "SamplerModel"]


class SamplerModel(Protocol):
    """Minimal interface `BayesianSampler` needs from its model."""

    thetas: np.ndarray

    def lnlikelihood(
        self,
        *,
        thetas: np.ndarray | None = ...,
    ) -> float: ...


class BayesianSampler:
    """MCMC sampler over epistatic model parameters via `emcee.EnsembleSampler`.

    Wraps a fitted epistasis model, runs affine-invariant MCMC from walker
    positions seeded around the model's ML parameters, and returns flat chains.

    Parameters
    ----------
    model
        A fitted epistasis model exposing `thetas` and `lnlikelihood(thetas=)`.
    lnprior
        Optional log-prior over `thetas`. Default is improper flat
        (`lambda thetas: 0.0`).
    n_walkers
        Number of walkers. Defaults to `2 * ndim`.
    """

    def __init__(
        self,
        model: SamplerModel,
        lnprior: Callable[[np.ndarray], float] | None = None,
        n_walkers: int | None = None,
    ) -> None:
        self.model = model
        self.ml_thetas = np.asarray(model.thetas, dtype=np.float64)
        self.ndim = self.ml_thetas.shape[0]
        self.n_walkers = n_walkers if n_walkers is not None else 2 * self.ndim
        if self.n_walkers < 2 * self.ndim:
            raise ValueError(
                f"n_walkers ({self.n_walkers}) must be at least 2 * ndim ({2 * self.ndim})."
            )
        self.lnprior = lnprior if lnprior is not None else _flat_lnprior

        def lnprob(thetas: np.ndarray) -> float:
            lp = float(self.lnprior(thetas))
            if not np.isfinite(lp):
                return float("-inf")
            ll = float(model.lnlikelihood(thetas=thetas))
            if not np.isfinite(ll):
                return float("-inf")
            return lp + ll

        self._sampler = emcee.EnsembleSampler(self.n_walkers, self.ndim, lnprob)
        self._last_state: emcee.State | None = None

    def initial_walkers(
        self,
        relative_width: float = 1e-2,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Gaussian cloud around the ML params; width scales per-coefficient."""
        generator = rng if rng is not None else np.random.default_rng()
        scale = relative_width * np.abs(self.ml_thetas)
        scale = np.where(scale == 0.0, relative_width, scale)
        noise = generator.standard_normal(size=(self.n_walkers, self.ndim))
        return self.ml_thetas + noise * scale

    def sample(
        self,
        n_steps: int = 500,
        n_burn: int = 100,
        progress: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Burn for `n_burn`, then sample `n_steps`. Returns the flat chain.

        Subsequent calls with `n_burn=0` continue from the previous state.
        """
        with np.errstate(all="ignore"):
            if self._last_state is None:
                pos = self.initial_walkers(rng=rng)
                if n_burn > 0:
                    state = self._sampler.run_mcmc(pos, n_burn, progress=progress)
                    self._sampler.reset()
                    pos = state.coords
            else:
                pos = self._last_state

            self._last_state = self._sampler.run_mcmc(pos, n_steps, progress=progress)

        return np.asarray(self._sampler.get_chain(flat=True), dtype=np.float64)

    @property
    def chain(self) -> np.ndarray:
        """Full non-flat chain of shape `(n_steps, n_walkers, ndim)`."""
        return np.asarray(self._sampler.get_chain(), dtype=np.float64)


def _flat_lnprior(thetas: np.ndarray) -> float:
    return 0.0
