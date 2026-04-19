"""Custom exceptions for epistasis-v2."""

from __future__ import annotations


class EpistasisError(Exception):
    """Base class for all epistasis-v2 errors."""


class XMatrixError(EpistasisError):
    """Invalid or missing epistasis design matrix."""


class FittingError(EpistasisError):
    """Model fit failed."""


__all__ = ["EpistasisError", "FittingError", "XMatrixError"]
