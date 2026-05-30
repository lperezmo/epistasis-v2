"""Coefficient plot with a site-participation grid.

``plot_coefs`` reproduces the signature figure from the original
harmslab/epistasis package: a bar chart of epistatic coefficients colored by
interaction order, with a grid underneath marking which sites participate in
each term.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from epistasis.mapping import Site

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from epistasis.models.base import EpistasisBaseModel

# Index 0 is reserved for the intercept / insignificant terms (grey). Indices
# 1.. are interaction orders. These read on both light and dark backgrounds.
DEFAULT_ORDER_COLORS: list[str] = [
    "#7d8590",  # 0: intercept / insignificant
    "#4C9BE8",  # 1
    "#F39C4B",  # 2
    "#56B870",  # 3
    "#E5564E",  # 4
    "#B07CD6",  # 5
    "#46C7C7",  # 6
]


def _order_of(site: Site) -> int:
    """Interaction order of a term. The intercept is encoded as ``(0,)``."""
    if len(site) >= 1 and site[0] == 0:
        return 0
    return len(site)


def _extract(
    model: EpistasisBaseModel | None,
    sites: Sequence[Site] | None,
    values: NDArray[np.float64] | None,
    stdeviations: NDArray[np.float64] | None,
) -> tuple[list[Site], NDArray[np.float64], NDArray[np.float64]]:
    """Resolve (sites, values, stdeviations) from a model or explicit arrays."""
    if model is not None:
        em = model.epistasis
        site_list = [tuple(int(x) for x in s) for s in em.sites]
        vals = np.asarray(em.values, dtype=np.float64)
        std = np.asarray(em.stdeviations, dtype=np.float64)
    else:
        if sites is None or values is None:
            raise ValueError("Pass either model=, or both sites= and values=.")
        site_list = [tuple(int(x) for x in s) for s in sites]
        vals = np.asarray(values, dtype=np.float64)
        if vals.shape[0] != len(site_list):
            raise ValueError("sites and values must be the same length.")
        std = (
            np.asarray(stdeviations, dtype=np.float64)
            if stdeviations is not None
            else np.full(vals.shape[0], np.nan, dtype=np.float64)
        )
    return site_list, vals, std


def _significance_mask(
    values: NDArray[np.float64],
    stdeviations: NDArray[np.float64],
    sigmas: float,
    significance: str | None,
    cutoff: float,
) -> NDArray[np.bool_]:
    """Boolean mask of significant terms. All True when significance is off."""
    n = values.shape[0]
    if sigmas == 0 or significance is None:
        return np.ones(n, dtype=bool)
    if not np.isfinite(stdeviations).all() or np.any(stdeviations <= 0):
        # Without usable standard errors we cannot judge significance.
        return np.ones(n, dtype=bool)

    from scipy.stats import norm

    z = np.abs(values / stdeviations)
    z = np.clip(z, None, 8.2)  # keep p within floating-point range
    p = 2.0 * (1.0 - norm.cdf(z))
    if significance == "bon":
        p = p * n
    elif significance != "p":
        raise ValueError(f"significance must be 'bon', 'p', or None; got {significance!r}.")
    return np.asarray(p < cutoff, dtype=bool)


def plot_coefs(
    model: EpistasisBaseModel | None = None,
    *,
    sites: Sequence[Site] | None = None,
    values: NDArray[np.float64] | None = None,
    stdeviations: NDArray[np.float64] | None = None,
    order_colors: Sequence[str] | None = None,
    sigmas: float = 0.0,
    significance: str | None = "bon",
    significance_cutoff: float = 0.05,
    star_cutoffs: tuple[float, ...] = (0.05, 0.01, 0.001),
    y_axis_name: str = "coefficient value",
    figsize: tuple[float, float] = (8.0, 5.0),
    height_ratio: float = 3.0,
    xgrid: bool = True,
    gridlines: float = 1.0,
    ax: list[Axes] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot epistatic coefficients as bars with a site-participation grid.

    Each bar is one coefficient, colored by interaction order. When ``xgrid``
    is True a grid is drawn underneath: one row per site, one column per
    coefficient, with a cell filled (in that term's order color) when the site
    participates in the term. Vertical dotted lines separate interaction
    orders. The intercept term is dropped.

    Parameters
    ----------
    model
        A fitted epistasis model. Its ``.epistasis`` map supplies the
        coefficients. Mutually exclusive with ``sites``/``values``.
    sites, values, stdeviations
        Provide coefficients directly instead of a model. ``sites`` is a
        sequence of 1-indexed site tuples (intercept ``(0,)``).
    order_colors
        Colors indexed by order; index 0 is the intercept / insignificant
        color. Defaults to :data:`DEFAULT_ORDER_COLORS`.
    sigmas
        Number of standard deviations for error bars. ``0`` (default) draws no
        error bars and disables significance shading.
    significance
        ``"bon"`` (Bonferroni-corrected p-values), ``"p"`` (raw), or ``None``.
        Only used when ``sigmas > 0`` and standard errors are available.
    significance_cutoff
        P-value threshold below which a term is considered significant.
    star_cutoffs
        Descending p-value thresholds; one ``*`` is stacked per threshold
        crossed (only when ``sigmas > 0``).
    y_axis_name
        Label for the bar-panel y-axis.
    figsize
        Figure size in inches (used only when ``ax`` is not supplied).
    height_ratio
        Bar-panel height relative to the grid panel.
    xgrid
        Draw the site-participation grid panel.
    gridlines
        Line width of the grid cell borders.
    ax
        Existing axes to draw into: ``[bar_axis, grid_axis]`` when ``xgrid``,
        else ``[bar_axis]``. A new figure is created when omitted.

    Returns
    -------
    fig, axes
        The figure and a list of axes: ``[bar_axis, grid_axis]`` when
        ``xgrid`` is True, otherwise ``[bar_axis]``.
    """
    site_list, vals, std = _extract(model, sites, values, stdeviations)

    # Drop the intercept; keep genuine interaction terms only.
    orders = [_order_of(s) for s in site_list]
    keep = [i for i, o in enumerate(orders) if o > 0]
    if not keep:
        raise ValueError("No interaction terms to plot (only an intercept was found).")
    site_list = [site_list[i] for i in keep]
    vals = vals[keep]
    std = std[keep]
    orders = [orders[i] for i in keep]
    ncoef = len(site_list)
    n_sites = max(max(s) for s in site_list)

    colors = list(order_colors) if order_colors is not None else list(DEFAULT_ORDER_COLORS)
    if len(colors) <= max(orders):
        raise ValueError(
            f"order_colors needs at least {max(orders) + 1} entries "
            f"(index 0 is the insignificant color); got {len(colors)}."
        )

    significant = _significance_mask(vals, std, sigmas, significance, significance_cutoff)
    bar_colors = [
        colors[o] if sig else colors[0] for o, sig in zip(orders, significant, strict=True)
    ]

    # --- figure / axes -----------------------------------------------------
    if ax is not None:
        axes = list(ax)
        fig = cast("Figure", axes[0].figure)
        bar_axis = axes[0]
        grid_axis = axes[1] if xgrid and len(axes) > 1 else None
    elif xgrid:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[height_ratio, 1.0], hspace=0.05)
        bar_axis = fig.add_subplot(gs[0])
        grid_axis = fig.add_subplot(gs[1], sharex=bar_axis)
    else:
        fig, bar_axis = plt.subplots(figsize=figsize)
        grid_axis = None

    x = np.arange(ncoef)

    # --- bars --------------------------------------------------------------
    yerr = sigmas * std if sigmas > 0 and np.isfinite(std).all() else None
    bar_axis.bar(
        x,
        vals,
        width=0.9,
        color=bar_colors,
        edgecolor="none",
        yerr=yerr,
        error_kw={"ecolor": plt.rcParams.get("text.color", "black"), "lw": 0.8, "capsize": 1.5},
    )
    bar_axis.axhline(0, color=plt.rcParams.get("grid.color", "#cccccc"), linewidth=0.8, zorder=0)
    bar_axis.set_ylabel(y_axis_name)
    bar_axis.set_xlim(-0.7, ncoef - 0.3)
    bar_axis.set_xticks([])

    # Order-break dividers and per-order labels.
    boundaries = [0]
    for i in range(1, ncoef):
        if orders[i] != orders[i - 1]:
            boundaries.append(i)
    ymax = bar_axis.get_ylim()[1]
    seg = boundaries + [ncoef]
    for a, start in enumerate(boundaries):
        end = seg[a + 1]
        o = orders[start]
        if a > 0:
            for axis in (bar_axis, grid_axis):
                if axis is not None:
                    axis.axvline(
                        start - 0.5,
                        color=plt.rcParams.get("text.color", "black"),
                        linestyle=":",
                        linewidth=1.0,
                    )
        bar_axis.text(
            (start + end - 1) / 2,
            ymax * 0.94,
            f"order {o}",
            ha="center",
            va="top",
            color=colors[o],
            fontsize=11,
            fontweight="bold",
        )

    # Significance stars.
    if sigmas > 0 and significance is not None and np.isfinite(std).all() and np.any(std > 0):
        from scipy.stats import norm

        z = np.clip(np.abs(vals / std), None, 8.2)
        p = 2.0 * (1.0 - norm.cdf(z))
        if significance == "bon":
            p = p * ncoef
        ymin = bar_axis.get_ylim()[0]
        offset = 0.03 * (ymax - ymin)
        for i in range(ncoef):
            n_stars = sum(1 for c in star_cutoffs if p[i] < c)
            for k in range(n_stars):
                bar_axis.text(x[i], ymin + k * offset, "*", ha="center", fontsize=12)

    # --- participation grid ------------------------------------------------
    if grid_axis is not None:
        grid = np.zeros((n_sites, ncoef, 4), dtype=float)  # transparent by default
        for j, (s, o) in enumerate(zip(site_list, orders, strict=True)):
            col = to_rgba(colors[o])
            for site_val in s:
                grid[site_val - 1, j] = col
        grid_axis.imshow(
            grid,
            aspect="auto",
            interpolation="nearest",
            extent=(-0.5, ncoef - 0.5, n_sites - 0.5, -0.5),
        )
        grid_axis.set_yticks(range(n_sites))
        grid_axis.set_yticklabels([f"site {i}" for i in range(n_sites)], fontsize=9)
        grid_axis.set_xticks([])
        grid_axis.set_ylabel("sites in term")
        grid_axis.set_xlim(-0.5, ncoef - 0.5)
        border = plt.rcParams.get("grid.color", "#cccccc")
        for r in range(n_sites + 1):
            grid_axis.axhline(r - 0.5, color=border, linewidth=gridlines)
        for spine in grid_axis.spines.values():
            spine.set_visible(False)

    out_axes = [bar_axis] if grid_axis is None else [bar_axis, grid_axis]
    return fig, out_axes
