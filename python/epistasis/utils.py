"""Shared helpers used across epistasis-v2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from epistasis.mapping import encoding_to_sites
from epistasis.matrix import ModelType, get_model_matrix

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap

__all__ = ["genotypes_to_X"]


def genotypes_to_X(
    genotypes: Sequence[str],
    gpm: GenotypePhenotypeMap,
    order: int = 1,
    model_type: ModelType = "global",
) -> np.ndarray:
    """Build the epistasis design matrix for an arbitrary list of genotype strings.

    Looks up each genotype in the gpm encoding table, produces the uint8 2D
    binary representation via gpmap-v2, then forms the design matrix with the
    site list implied by `order`.
    """
    from gpmap import genotypes_to_binary_packed

    sites = encoding_to_sites(order, gpm.encoding_table)
    binary_packed = genotypes_to_binary_packed(list(genotypes), gpm.encoding_table)
    return get_model_matrix(binary_packed, sites, model_type=model_type)
