"""Matplotlib-backed plotting for epistasis models.

This subpackage is optional. Install via ``pip install epistasis-v2[plot]``
or ``uv add --extra plot epistasis-v2``. Importing :mod:`epistasis.pyplot`
without matplotlib available raises a clear ImportError.
"""

from __future__ import annotations

try:
    import matplotlib  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("epistasis.pyplot requires matplotlib; install epistasis-v2[plot]") from exc

from epistasis.pyplot.coefs import DEFAULT_ORDER_COLORS, plot_coefs
from epistasis.pyplot.correlation import plot_correlation

__all__ = [
    "DEFAULT_ORDER_COLORS",
    "plot_coefs",
    "plot_correlation",
]
