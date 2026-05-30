---
title: "pyplot reference"
description: "plot_coefs: coefficient bar chart with a site-participation grid."
---

# `epistasis.pyplot`

Optional matplotlib-backed plotting. Requires `epistasis-v2[plot]`. See
[Plotting epistatic coefficients](../guides/plotting.md) for a guide-style walkthrough.

```python
from epistasis.pyplot import plot_coefs, DEFAULT_ORDER_COLORS
```

## `plot_coefs`

```python
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
) -> tuple[Figure, list[Axes]]
```

Plot epistatic coefficients as bars with a site-participation grid. Each bar is one
coefficient, colored by interaction order; the grid underneath marks which sites
participate in each term. The intercept term is dropped.

### Coefficient source

- Pass `model` (a fitted epistasis model) to read coefficients from its `.epistasis`
  map (`sites`, `values`, `stdeviations`).
- Or pass `sites` and `values` directly. `sites` is a sequence of 1-indexed site
  tuples; the intercept is `(0,)`.

These are mutually exclusive; supplying neither raises `ValueError`.

### Key behaviors

- Bars are colored by interaction order via `order_colors` (index 0 is the intercept /
  insignificant color, defaulting to `DEFAULT_ORDER_COLORS`). Vertical dotted lines
  separate orders.
- The grid (`xgrid=True`) is one row per site, one column per coefficient; a cell is
  filled in that term's order color when the site participates.
- When `sigmas > 0` and standard errors are present, error bars are drawn,
  non-significant terms are greyed, and `*` stars are stacked per `star_cutoffs`
  threshold crossed. `significance` is `"bon"` (Bonferroni), `"p"` (raw), or `None`.

### Returns

`(fig, axes)` where `axes` is `[bar_axis, grid_axis]` when `xgrid` is True, otherwise
`[bar_axis]`. Pass `ax` to draw into existing axes instead of creating a new figure.

## `DEFAULT_ORDER_COLORS`

A list of hex colors indexed by interaction order. Index 0 is reserved for the
intercept and insignificant terms (grey); indices 1.. are the per-order colors. The
defaults read on both light and dark backgrounds.
