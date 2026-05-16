---
title: "Get started with epistasis-v2"
description: "Learn how to install epistasis-v2, construct a GenotypePhenotypeMap, and fit your first EpistasisLinearRegression model in under five minutes."
---

# Get started with epistasis-v2

epistasis-v2 is a Python library for fitting high-order epistatic interactions in genotype-phenotype maps. This page walks you through installing the package, constructing a `GenotypePhenotypeMap`, and running your first fit with `EpistasisLinearRegression`. Everything you need to get a result from a minimal dataset.

## 1. Install the package

Install epistasis-v2 from PyPI. The wheel includes pre-compiled Rust extensions, so you do not need a Rust toolchain on your machine.

```bash
pip install epistasis-v2
```

Python 3.10 or later is required. See [Installation](installation.md) for build-from-source instructions.

## 2. Build a genotype-phenotype map

epistasis-v2 reads experimental data through `GenotypePhenotypeMap` objects provided by the companion library `gpmap-v2` (installed automatically as a dependency).

You need four things to construct a map:

- **`wildtype`**: the reference sequence
- **`genotypes`**: all sequences in your library (including the wildtype)
- **`phenotypes`**: the measured phenotype for each genotype, in the same order
- **`mutations`**: a dict mapping each position to its possible characters

```python
from gpmap import GenotypePhenotypeMap

gpm = GenotypePhenotypeMap(
    wildtype="AAA",
    genotypes=["AAA", "AAT", "ATA", "ATT", "TAA", "TAT", "TTA", "TTT"],
    phenotypes=[0.0, 0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 1.0],
    mutations={0: "AT", 1: "AT", 2: "AT"},
)
```

!!! note

    The `mutations` dict keys are zero-based position indices. Each value is a string of the characters that appear at that position, with the wildtype character listed first.

## 3. Attach the map and fit the model

Create an `EpistasisLinearRegression`, attach the map with `add_gpm()`, then call `fit()`. For a complete biallelic library encoded with global (Hadamard) encoding, `fit()` automatically uses the Walsh-Hadamard fast path: no dense design matrix is built.

```python
from epistasis.models.linear.ordinary import EpistasisLinearRegression

model = EpistasisLinearRegression(order=3, model_type="global")
model.add_gpm(gpm)
model.fit()
```

- `order` sets the maximum interaction order to fit. `order=3` captures all pairwise and three-way interactions for a three-site map.
- `model_type` selects the encoding. `"global"` uses Hadamard (Walsh) encoding; `"local"` uses biochemical (reference-free) encoding.

## 4. Inspect the epistatic coefficients

After fitting, the epistatic coefficients are stored in `model.epistasis.values` and the analytic OLS standard errors in `model.epistasis.stdeviations`.

```python
import numpy as np

print("Epistatic coefficients:")
print(model.epistasis.values)

print("\nStandard deviations:")
print(model.epistasis.stdeviations)
```

`model.epistasis` is an `EpistasisMap` object. You can also access the full DataFrame:

```python
print(model.epistasis.data)
```

The DataFrame has columns `labels`, `orders`, `sites`, `values`, and `stdeviations`, one row per fitted coefficient.

## Complete runnable example

The snippet below combines all four steps into a single script you can copy and run directly.

```python
from gpmap import GenotypePhenotypeMap
from epistasis.models.linear.ordinary import EpistasisLinearRegression

# 1. Build a complete 3-site biallelic genotype-phenotype map.
gpm = GenotypePhenotypeMap(
    wildtype="AAA",
    genotypes=["AAA", "AAT", "ATA", "ATT", "TAA", "TAT", "TTA", "TTT"],
    phenotypes=[0.0, 0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 1.0],
    mutations={0: "AT", 1: "AT", 2: "AT"},
)

# 2. Create the model, attach the GPM, and fit.
model = EpistasisLinearRegression(order=3, model_type="global")
model.add_gpm(gpm)
model.fit()

# 3. Inspect results.
print(model.epistasis.data.to_string(index=False))
```

## Next steps

- Read [Installation](installation.md) for Python version support, optional dependencies, and building from source.
- See [Core Concepts: Genotype-Phenotype Maps](concepts/genotype-phenotype-maps.md) for a deeper explanation of the map structure.
- Explore [Models: Linear](models/linear.md) for `EpistasisRidge`, `EpistasisLasso`, and `EpistasisElasticNet`.
