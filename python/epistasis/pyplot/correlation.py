"""Observed vs predicted phenotype scatter.

``plot_correlation`` is the second classic epistasis diagnostic: a scatter of
observed against model-predicted phenotype around the 1:1 line, annotated with
the coefficient of determination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from epistasis.models.base import EpistasisBaseModel


def _r_squared(observed: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def plot_correlation(
    model: EpistasisBaseModel | None = None,
    *,
    observed: NDArray[np.float64] | None = None,
    predicted: NDArray[np.float64] | None = None,
    color: str | None = None,
    point_size: float = 36.0,
    alpha: float = 0.85,
    annotate_r2: bool = True,
    figsize: tuple[float, float] = (5.5, 5.5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Scatter observed against predicted phenotype around the 1:1 line.

    Parameters
    ----------
    model
        A fitted epistasis model. Observed values are taken from its attached
        GPM and predictions from ``model.predict()``. Mutually exclusive with
        ``observed``/``predicted``.
    observed, predicted
        Provide the two arrays directly instead of a model.
    color
        Marker color. Defaults to the active theme's first cycle color.
    point_size, alpha
        Marker size and opacity.
    annotate_r2
        Annotate the coefficient of determination computed from the observed
        and predicted values.
    figsize
        Figure size in inches (used only when ``ax`` is not supplied).
    ax
        Existing axes to draw into. A new figure is created when omitted.

    Returns
    -------
    fig, ax
        The figure and the scatter axes.
    """
    if model is not None:
        obs = np.asarray(model.gpm.phenotypes, dtype=np.float64)
        pred = np.asarray(model.predict(), dtype=np.float64)
    else:
        if observed is None or predicted is None:
            raise ValueError("Pass either model=, or both observed= and predicted=.")
        obs = np.asarray(observed, dtype=np.float64)
        pred = np.asarray(predicted, dtype=np.float64)
        if obs.shape != pred.shape:
            raise ValueError("observed and predicted must have the same shape.")
    r2 = _r_squared(obs, pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast("Figure", ax.figure)

    if color is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C9BE8"])
        color = cycle[0]

    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    span = hi - lo
    lo, hi = lo - 0.05 * span, hi + 0.05 * span

    ax.plot(
        [lo, hi],
        [lo, hi],
        color=plt.rcParams.get("grid.color", "#999999"),
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(obs, pred, s=point_size, color=color, edgecolors="none", alpha=alpha, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed phenotype")
    ax.set_ylabel("predicted phenotype")
    if annotate_r2 and np.isfinite(r2):
        ax.annotate(
            f"$R^2$ = {r2:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=12,
        )
    return fig, ax
