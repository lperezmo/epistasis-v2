"""Factory functions for synthetic genotype-phenotype maps driven by linear
epistatic coefficients.

v1 exposed this as subclasses of `GenotypePhenotypeMap` with a mutable
`build()` method. v2 collapses it into stateless factory functions that
return a plain `GenotypePhenotypeMap` plus the coefficients used to build
it. If you want to iterate on coefficients, call the function again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from gpmap import GenotypePhenotypeMap, enumerate_genotypes_str

from epistasis.mapping import Site, encoding_to_sites
from epistasis.matrix import ModelType, get_model_matrix

__all__ = ["simulate_linear_gpm", "simulate_random_linear_gpm"]


def simulate_linear_gpm(
    wildtype: str,
    mutations: Mapping[int, list[str] | None],
    order: int,
    coefficients: Sequence[float] | np.ndarray,
    model_type: ModelType = "global",
    stdeviations: float | np.ndarray | None = None,
) -> tuple[GenotypePhenotypeMap, list[Site]]:
    """Build a synthetic GPM whose phenotypes are `X @ coefficients`.

    Parameters
    ----------
    wildtype
        Wildtype genotype string.
    mutations
        Mapping from each position (0-indexed) to the list of alternative
        letters allowed at that position. A value of `None` means the
        position is fixed to the wildtype letter. This is the same shape
        gpmap-v2's `enumerate_genotypes_str` expects.
    order
        Epistatic order for the design matrix.
    coefficients
        Epistatic coefficient values. Must match the site list produced by
        `encoding_to_sites(order, encoding_table)`, so you typically want to
        build this GPM once to inspect the site list, then re-simulate.
    model_type
        `"global"` (Hadamard) or `"local"` (biochemical).
    stdeviations
        Optional phenotype uncertainties. If a float, used for every
        genotype.

    Returns
    -------
    gpm
        The new `GenotypePhenotypeMap`.
    sites
        The list of sites used to build the design matrix; same ordering as
        `coefficients`.
    """
    genotypes = np.array(enumerate_genotypes_str(wildtype=wildtype, mutations=mutations))

    # Build a provisional GPM with zero phenotypes to get the encoding table.
    provisional = GenotypePhenotypeMap(
        wildtype=wildtype,
        genotypes=genotypes,
        phenotypes=np.zeros(len(genotypes), dtype=np.float64),
        mutations=mutations,
    )

    sites = encoding_to_sites(order, provisional.encoding_table)
    coefs = np.asarray(coefficients, dtype=np.float64)
    if coefs.shape != (len(sites),):
        raise ValueError(
            f"coefficients has length {coefs.shape[0]}, expected {len(sites)} "
            f"(order={order}, wildtype length={len(wildtype)})."
        )

    X = get_model_matrix(provisional.binary_packed, sites, model_type=model_type)
    phenotypes = (X.astype(np.float64) @ coefs).astype(np.float64)

    if stdeviations is None:
        stds: np.ndarray | None = None
    elif isinstance(stdeviations, int | float):
        stds = np.full(len(genotypes), float(stdeviations), dtype=np.float64)
    else:
        stds = np.asarray(stdeviations, dtype=np.float64)

    gpm = GenotypePhenotypeMap(
        wildtype=wildtype,
        genotypes=genotypes,
        phenotypes=phenotypes,
        stdeviations=stds,
        mutations=mutations,
    )
    return gpm, sites


def simulate_random_linear_gpm(
    wildtype: str,
    mutations: Mapping[int, list[str] | None],
    order: int,
    coefficient_range: tuple[float, float] = (-1.0, 1.0),
    model_type: ModelType = "global",
    stdeviations: float | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[GenotypePhenotypeMap, list[Site], np.ndarray]:
    """Build a synthetic GPM with coefficients drawn uniformly from `coefficient_range`.

    Returns (gpm, sites, coefficients_used). Pass a seeded `rng` for
    reproducibility.
    """
    generator = rng if rng is not None else np.random.default_rng()
    provisional = GenotypePhenotypeMap(
        wildtype=wildtype,
        genotypes=np.array(enumerate_genotypes_str(wildtype=wildtype, mutations=mutations)),
        phenotypes=np.zeros(
            len(enumerate_genotypes_str(wildtype=wildtype, mutations=mutations)),
            dtype=np.float64,
        ),
        mutations=mutations,
    )
    sites = encoding_to_sites(order, provisional.encoding_table)

    low, high = coefficient_range
    coefs = generator.uniform(low, high, size=len(sites))

    gpm, _ = simulate_linear_gpm(
        wildtype=wildtype,
        mutations=mutations,
        order=order,
        coefficients=coefs,
        model_type=model_type,
        stdeviations=stdeviations,
    )
    return gpm, sites, coefs
