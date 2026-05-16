---
title: "Simulate genotype-phenotype maps with known epistasis"
description: "Use simulate_linear_gpm and simulate_random_linear_gpm to create synthetic GPMs with known epistatic coefficients for testing and benchmarking your models."
---

# Simulate genotype-phenotype maps with known epistasis

Synthetic genotype-phenotype maps let you verify that your fitting pipeline recovers epistatic coefficients you set, benchmark runtime at different sequence lengths and interaction orders, and build controlled datasets for unit tests. The `epistasis.simulate` module provides two factory functions that return plain `GenotypePhenotypeMap` objects alongside the coefficient metadata you need to compare fitted results with ground truth.

## Why simulate?

Before applying an epistasis model to experimental data, it is useful to confirm that the model can recover known signal from a clean dataset. Simulation gives you that control: you set the coefficients, generate phenotypes deterministically, fit the model, and measure how close the recovered coefficients are to the originals. This workflow catches implementation bugs, reveals overfitting at high order, and gives you a concrete baseline for benchmarking.

## `simulate_linear_gpm`

`simulate_linear_gpm` builds a synthetic GPM whose phenotypes equal `X @ coefficients`, where `X` is the epistasis design matrix at the requested order and encoding.

```python
from epistasis.simulate import simulate_linear_gpm

gpm, sites = simulate_linear_gpm(
    wildtype=wildtype,
    mutations=mutations,
    order=order,
    coefficients=coefficients,
    model_type="global",
    stdeviations=None,
)
```

The function returns a `(gpm, sites)` tuple. `gpm` is a `GenotypePhenotypeMap` with phenotypes computed from your coefficients. `sites` is the ordered list of epistatic sites used to build the design matrix, the same ordering as `coefficients`.

!!! warning

    `coefficients` must have exactly as many elements as there are sites at the requested order. The number of sites depends on `wildtype`, `mutations`, and `order`. Call the function once to inspect the returned `sites` list, then construct your coefficient array to match it.

### Parameters

`wildtype` (`str`, required)
:   The wildtype genotype string (e.g., `"AAAA"`). Defines the sequence length and the reference state at each position.

`mutations` (`Mapping[int, list[str] | None]`, required)
:   A mapping from each position (0-indexed) to the list of alternative letters allowed at that position. Pass `None` for a position to fix it to the wildtype letter. This is the same format `gpmap.enumerate_genotypes_str` expects.

`order` (`int`, required)
:   Epistatic order for the design matrix. Order 1 is additive; order 2 includes pairwise terms; and so on up to the sequence length.

`coefficients` (`Sequence[float] | np.ndarray`, required)
:   Epistatic coefficient values, one per site in the site list returned by `encoding_to_sites(order, encoding_table)`. Length must match exactly.

`model_type` (`str`, default `"global"`)
:   Design matrix encoding: `"global"` (Hadamard / Walsh basis) or `"local"` (biochemical / indicator basis).

`stdeviations` (`float | np.ndarray | None`, default `None`)
:   Optional phenotype measurement uncertainties. A single `float` is broadcast to all genotypes. An array must have one value per genotype. Pass `None` for no uncertainty.

## `simulate_random_linear_gpm`

`simulate_random_linear_gpm` works like `simulate_linear_gpm` but draws coefficients uniformly at random from `coefficient_range`, so you do not need to supply explicit values.

```python
from epistasis.simulate import simulate_random_linear_gpm
import numpy as np

rng = np.random.default_rng(42)

gpm, sites, coefs = simulate_random_linear_gpm(
    wildtype=wildtype,
    mutations=mutations,
    order=order,
    coefficient_range=(-1.0, 1.0),
    model_type="global",
    stdeviations=None,
    rng=rng,
)
```

The function returns a `(gpm, sites, coefficients_used)` tuple. `coefficients_used` is the array of randomly drawn values, which you can use later to compare against fitted coefficients.

### Parameters

`wildtype` (`str`, required)
:   The wildtype genotype string.

`mutations` (`Mapping[int, list[str] | None]`, required)
:   Mapping from position to allowed alternative letters, as described above.

`order` (`int`, required)
:   Epistatic order for the design matrix.

`coefficient_range` (`tuple[float, float]`, default `(-1.0, 1.0)`)
:   `(low, high)` bounds for the uniform distribution from which coefficients are drawn.

`model_type` (`str`, default `"global"`)
:   Design matrix encoding: `"global"` or `"local"`.

`stdeviations` (`float | np.ndarray | None`, default `None`)
:   Optional phenotype uncertainties, as described above.

`rng` (`np.random.Generator | None`, default `None`)
:   A NumPy random generator. Pass a seeded generator (e.g., `np.random.default_rng(42)`) to make the simulation reproducible. If `None`, a fresh unseeded generator is created.

## Complete example

The following example simulates a length-4 biallelic GPM at order 2, fits `EpistasisLinearRegression` to recover the coefficients, and prints the comparison.

```python
import numpy as np
from epistasis.simulate import simulate_random_linear_gpm
from epistasis.models.linear import EpistasisLinearRegression

# --- 1. Define the sequence space ---
wildtype = "AAAA"
mutations = {0: ["T"], 1: ["T"], 2: ["T"], 3: ["T"]}

# --- 2. Simulate with a seeded RNG so results are reproducible ---
rng = np.random.default_rng(42)

gpm, sites, true_coefs = simulate_random_linear_gpm(
    wildtype=wildtype,
    mutations=mutations,
    order=2,
    coefficient_range=(-1.0, 1.0),
    model_type="global",
    rng=rng,
)

print(f"Sequence space: {len(gpm.genotypes)} genotypes")
print(f"Number of epistatic sites (order <= 2): {len(sites)}")

# --- 3. Fit the model ---
model = EpistasisLinearRegression(order=2, model_type="global")
model.add_gpm(gpm)
model.fit()

fitted_coefs = model.epistasis.values

# --- 4. Compare true vs fitted coefficients ---
print("\nSite | True coef | Fitted coef")
print("-" * 40)
for site, true, fitted in zip(sites, true_coefs, fitted_coefs):
    print(f"{str(site):<20} {true:+.4f}   {fitted:+.4f}")

residuals = true_coefs - fitted_coefs
print(f"\nMax absolute error: {np.abs(residuals).max():.2e}")
```

!!! tip

    On a full biallelic library at full order, `EpistasisLinearRegression` uses the Walsh-Hadamard fast path and recovers coefficients exactly (up to floating-point precision). Errors on the order of `1e-12` are normal and indicate a correct fit, not numerical issues.

## Reproducibility with `rng`

Passing a seeded `np.random.Generator` makes the random coefficient draw deterministic across runs and platforms:

```python
# Both calls produce identical gpm, sites, and coefs
rng_a = np.random.default_rng(42)
rng_b = np.random.default_rng(42)

_, _, coefs_a = simulate_random_linear_gpm("AAAA", {0: ["T"], 1: ["T"], 2: ["T"], 3: ["T"]}, 2, rng=rng_a)
_, _, coefs_b = simulate_random_linear_gpm("AAAA", {0: ["T"], 1: ["T"], 2: ["T"], 3: ["T"]}, 2, rng=rng_b)

assert np.allclose(coefs_a, coefs_b)
```

If you omit `rng`, a fresh unseeded generator is used each time, so coefficients will differ between runs.
