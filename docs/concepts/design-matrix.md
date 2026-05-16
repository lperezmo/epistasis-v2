---
title: "The epistasis design matrix"
description: "The design matrix encodes each genotype as a vector of epistatic site products. Learn how epistasis-v2 builds and uses design matrices for regression."
---

# The epistasis design matrix

Regression-based epistasis models work by expressing each genotype as a row in a design matrix X, where each column corresponds to one epistatic coefficient. Building X correctly, choosing the right columns, encoding genotypes in the right way, and computing site products efficiently, is the core numerical step that sits between your raw genotype-phenotype data and a fitted model. This page explains how that process works in epistasis-v2.

## Structure of the design matrix

The design matrix X has shape `(n_genotypes, n_sites)`:

- Each **row** represents one genotype from your library.
- Each **column** represents one epistatic site: a tuple of 1-based mutation indices whose encoded values are multiplied together to form that column's entries.

The sites are ordered by interaction order. The intercept always occupies the first column.

### Site tuples

Sites are represented as tuples of integers. The convention is:

| Site tuple | Meaning |
|------------|---------|
| `(0,)` | Intercept: all entries are `+1` (global) or `+1` (local) |
| `(1,)` | First-order term for mutation index 1 |
| `(2,)` | First-order term for mutation index 2 |
| `(1, 2)` | Pairwise interaction between mutations 1 and 2 |
| `(1, 2, 3)` | Three-way interaction among mutations 1, 2, and 3 |

For a given site `(i, j)`, the column value for a genotype is the product of the encoded values at columns `i` and `j` of the encoded-vector matrix.

!!! note

    Mutation indices are 1-based and match the `mutation_index` column of the gpmap-v2 `encoding_table`. Column 0 of the encoded-vector matrix is always the intercept, which is why `(0,)` maps to a constant column.

## Building the site list with `encoding_to_sites`

Before you can assemble X, you need the ordered list of sites. `encoding_to_sites` reads the `encoding_table` from your GPM and returns that list:

```python
from epistasis.mapping import encoding_to_sites

sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
# Returns a list of Site tuples, e.g.:
# [(0,), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `order` | Highest interaction order to include. |
| `encoding_table` | The `encoding_table` DataFrame from a `GenotypePhenotypeMap`. |
| `start_order` | If `0` (default), prepend the intercept `(0,)`. If `>= 1`, omit the intercept and start from that order. |

Wildtype rows (where `mutation_index` is NaN) are automatically dropped before building the combinations.

## Assembling X with `get_model_matrix`

`get_model_matrix` is the single-call convenience function that encodes genotypes and computes all site products in one step:

```python
from epistasis.matrix import get_model_matrix

X = get_model_matrix(
    binary_packed=gpm.binary_packed,
    sites=sites,
    model_type="global",
)
# X is a numpy int8 array of shape (n_genotypes, len(sites))
```

Internally it calls two lower-level functions in sequence:

1. **`encode_vectors(binary_packed, model_type)`**: converts the uint8 `binary_packed` array to an int8 encoded-vector matrix. Under `"global"` encoding wildtype bits become `+1` and mutant bits become `-1`. Under `"local"` encoding the values stay `0` and `1`. Both encodings prepend a constant intercept column.

2. **`build_model_matrix(encoded, sites)`**: takes the encoded-vector matrix and the site list, and computes the elementwise column products for each site across all genotype rows.

## The Rust-accelerated `build_model_matrix` kernel

The site-product computation in `build_model_matrix` is implemented as a Rust kernel in `epistasis._core`. It uses a ragged flat layout: all site indices are concatenated into one `int64` array (`sites_flat`) with a companion `int64` offsets array (`sites_offsets`) that records where each site's indices begin and end. This avoids Python-level iteration over genotype rows and runs the products in parallel across threads.

```python
# Lower-level usage. You rarely need this directly.
from epistasis.matrix import encode_vectors, build_model_matrix
from epistasis.mapping import encoding_to_sites

sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
encoded = encode_vectors(gpm.binary_packed, model_type="global")
X = build_model_matrix(encoded, sites)
```

!!! tip

    You only need to call `encode_vectors` and `build_model_matrix` separately if you want to reuse the encoded-vector matrix for multiple site lists (for example, when experimenting with different interaction orders on the same dataset).

## Building X manually: a complete example

```python
from gpmap import GenotypePhenotypeMap
from epistasis.mapping import encoding_to_sites
from epistasis.matrix import get_model_matrix, model_matrix_as_dataframe

gpm = GenotypePhenotypeMap(
    wildtype="AAA",
    genotypes=["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"],
    phenotypes=[0.1, 0.4, 0.3, 0.9, 0.2, 0.5, 0.7, 1.2],
    mutations={0: ["A", "B"], 1: ["A", "B"], 2: ["A", "B"]},
)

sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
X = get_model_matrix(gpm.binary_packed, sites, model_type="global")

# Wrap in a DataFrame for inspection; columns are keyed by site tuples
df = model_matrix_as_dataframe(X, sites, index=gpm.genotypes)
print(df)
```

`model_matrix_as_dataframe` wraps X in a pandas DataFrame with site tuples as column labels and (optionally) genotype strings as the row index, which makes it easy to inspect which column corresponds to which interaction.

## How models use X

In normal usage you never call these functions yourself. When you call `model.fit()` without passing an explicit `X`, the model calls `get_model_matrix` internally (caching the result) and passes it to the underlying sklearn estimator. When you call `model.predict(X=["ABB", "BAA"])`, the model builds a fresh X for just those two genotypes using `genotypes_to_X` and returns predictions.

You can also pass a prebuilt design matrix directly:

```python
X = get_model_matrix(gpm.binary_packed, sites, model_type="global")
model.fit(X=X, y=gpm.phenotypes)
```

This skips the internal build step and is useful when you want precise control over the site list or encoding.
