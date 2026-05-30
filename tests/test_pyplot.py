"""Smoke tests for epistasis.pyplot. Run with the Agg backend (headless)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.pyplot import DEFAULT_ORDER_COLORS, plot_coefs
from epistasis.simulate import simulate_random_linear_gpm


def _fitted_model(order: int = 3, length: int = 4):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    gpm, _, _ = simulate_random_linear_gpm(
        wildtype="A" * length,
        mutations={i: ["A", "T"] for i in range(length)},
        order=order,
        coefficient_range=(-1.0, 1.0),
        stdeviations=0.05,
        rng=rng,
    )
    return EpistasisLinearRegression(order=order).add_gpm(gpm).fit()


def test_plot_coefs_returns_two_axes() -> None:
    model = _fitted_model()
    fig, axes = plot_coefs(model)
    assert len(axes) == 2
    assert fig is not None
    plt.close(fig)


def test_plot_coefs_no_grid_returns_one_axis() -> None:
    model = _fitted_model()
    fig, axes = plot_coefs(model, xgrid=False)
    assert len(axes) == 1
    plt.close(fig)


def test_plot_coefs_from_arrays() -> None:
    # No model: supply sites and values directly. (1,) and (2,) are order-1
    # terms; (1, 2) is an order-2 term.
    sites = [(0,), (1,), (2,), (1, 2)]
    values = np.array([0.5, -0.3, 0.8, 0.2])
    fig, axes = plot_coefs(sites=sites, values=values)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_coefs_significance_stars() -> None:
    model = _fitted_model()
    fig, axes = plot_coefs(model, sigmas=1.0, significance="bon")
    assert len(axes) == 2
    plt.close(fig)


def test_plot_coefs_custom_axes() -> None:
    model = _fitted_model()
    fig, (ax_bar, ax_grid) = plt.subplots(2, 1)
    out_fig, axes = plot_coefs(model, ax=[ax_bar, ax_grid])
    assert out_fig is fig
    assert axes[0] is ax_bar
    plt.close(fig)


def test_plot_coefs_requires_model_or_arrays() -> None:
    with pytest.raises(ValueError):
        plot_coefs()


def test_plot_coefs_rejects_intercept_only() -> None:
    with pytest.raises(ValueError):
        plot_coefs(sites=[(0,)], values=np.array([1.0]))


def test_default_order_colors_has_intercept_slot() -> None:
    # Index 0 is reserved for intercept / insignificant terms.
    assert len(DEFAULT_ORDER_COLORS) >= 5
