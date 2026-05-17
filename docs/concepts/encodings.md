---
title: "Global and local encodings explained"
description: "epistasis-v2 supports global (Hadamard) and local (biochemical) encodings. Learn which to use and how each affects coefficient interpretation."
---

# Global and local encodings explained

Every epistasis model in epistasis-v2 encodes genotypes as numeric vectors before building the design matrix. The choice of encoding controls how binary genotype values (`0`/`1`) are mapped to numeric entries in those vectors, and that choice directly determines what the fitted coefficients mean. Two encodings are supported: global (Hadamard/Walsh) and local (biochemical). Choosing the right one depends on whether you prioritize mathematical properties like orthogonality or mechanistic interpretability.

## The `ModelType` parameter

The encoding is selected through the `model_type` parameter, typed as `Literal["global", "local"]`. You pass it when constructing a model or when calling matrix functions directly:

```python
from epistasis.models.linear import EpistasisLinearRegression

model = EpistasisLinearRegression(order=2, model_type="global")
```

The same parameter is accepted by `encode_vectors` and `get_model_matrix` when you build design matrices manually.

## Global encoding (Hadamard/Walsh)

Under global encoding, wildtype bits are mapped to $+1$ and mutant bits to $-1$. The intercept column is always $+1$.

$$
\text{wildtype letter} \to +1, \quad
\text{mutant letter} \to -1, \quad
\text{intercept} \to +1
$$

This is the Hadamard (Walsh) encoding. It has two important mathematical properties:

- **Orthogonality**: for a complete biallelic library, the columns of the design matrix are orthogonal. Each coefficient can be estimated independently of all others.
- **Intercept = mean phenotype**: the intercept coefficient equals the mean phenotype across all genotypes in the library.

Global encoding also enables the Walsh-Hadamard fast path. When you fit an `EpistasisLinearRegression` at full order on a complete biallelic library with `model_type="global"`, the library automatically uses `fwht_ols_coefficients`, a closed-form `O(N log N)` solve via the Fast Walsh-Hadamard Transform, instead of assembling a dense design matrix and solving a least-squares system. This fast path delivers the speedups shown in the benchmark table in the README.

!!! info

    The FWHT fast path is only available for global encoding on full-order, complete biallelic libraries. For any other configuration (partial libraries, multiallelic sequences, order < L, or local encoding) the model falls back to the sklearn lstsq path automatically.

## Local encoding (biochemical)

Under local encoding, the binary values are kept as-is: wildtype bits remain $0$ and mutant bits remain $1$. The intercept column is $+1$.

$$
\text{wildtype letter} \to 0, \quad
\text{mutant letter} \to 1, \quad
\text{intercept} \to +1
$$

This encoding has a direct mechanistic interpretation:

- The **intercept** is the wildtype phenotype (the predicted value when all binary inputs are 0, i.e., the pure wildtype).
- Each **first-order coefficient** measures the additive effect of introducing that mutation into the wildtype background.
- Each **higher-order coefficient** measures the deviation from the additive prediction in the wildtype background.

Local encoding is preferred when you want coefficients that correspond to effects observable in a biochemical experiment: the additive effect of a single mutation measured against the wildtype, without reference to a library mean.

!!! note

    Local encoding does not produce orthogonal columns for a complete library, so coefficient estimates will be correlated. This rarely matters for interpretation, but it means the FWHT fast path is not available.

## Comparison at a glance

=== "Global encoding"

    ```python
    from epistasis.models.linear import EpistasisLinearRegression

    model = EpistasisLinearRegression(order=2, model_type="global")
    model.add_gpm(gpm)
    model.fit()

    # Intercept ~ mean phenotype across the full library
    # First-order coefficients: deviation from the mean due to each mutation
    # FWHT fast path engaged for full-order biallelic libraries
    print(model.epistasis.values)
    ```

=== "Local encoding"

    ```python
    from epistasis.models.linear import EpistasisLinearRegression

    model = EpistasisLinearRegression(order=2, model_type="local")
    model.add_gpm(gpm)
    model.fit()

    # Intercept ~ wildtype phenotype
    # First-order coefficients: effect of each mutation in the wildtype background
    # sklearn lstsq path always used
    print(model.epistasis.values)
    ```

## How the Rust kernel handles encoding

Both encodings are implemented in the `encode_vectors` Rust kernel (exposed as `epistasis._core.encode_vectors`). It accepts a uint8 `binary_packed` array of shape `(n_genotypes, n_bits)` and returns an int8 encoded-vector matrix of shape `(n_genotypes, n_bits + 1)`:

```python
from epistasis.matrix import encode_vectors
import numpy as np

# gpm.binary_packed is uint8 with values in {0, 1}
encoded = encode_vectors(gpm.binary_packed, model_type="global")
# encoded is int8, shape (n_genotypes, n_bits + 1)
# Column 0 is the intercept (+1 for all rows)
# Columns 1..n_bits hold +1/-1 (global) or 0/1 (local)
```

The int8 output dtype keeps memory usage low for large libraries. The `build_model_matrix` kernel consumes this int8 matrix to compute site products and returns another int8 matrix: the full design matrix X.

## Which encoding should you use?

Choose **global** when:

- You want orthogonal coefficients and the mathematical properties of Walsh-Hadamard analysis.
- You are working with a complete biallelic library and want the FWHT fast path.
- You are comparing coefficients across datasets with different wildtype phenotypes.

Choose **local** when:

- You want coefficients that are directly interpretable as mutational effects in the wildtype background.
- You are communicating results to an audience familiar with biochemical conventions.
- You are working with a partial library or a multiallelic alphabet where global orthogonality does not hold anyway.
