---
title: "EpistasisLinearRegression: OLS fitting"
description: "Fit epistatic coefficients with ordinary least squares, analytic standard errors, and a Walsh-Hadamard fast path for large combinatorial libraries."
---

# EpistasisLinearRegression: OLS fitting

`EpistasisLinearRegression` is the simplest and fastest way to decompose a genotype-phenotype map into epistatic coefficients. It performs ordinary least squares over the epistasis design matrix and writes analytic coefficient standard errors directly into `model.epistasis.stdeviations`. For complete biallelic libraries encoded with the global (Hadamard) scheme, the model automatically engages a Walsh-Hadamard fast path that scales as O(n log n) instead of O(n^2), up to 6000x faster than the full matrix solve.

## Constructor parameters

`order` (`int`, default `1`)
:   Maximum interaction order to fit. `order=1` gives the additive model; `order=2` adds pairwise interactions; and so on up to the length of the sequence.

`model_type` (`str`, default `"global"`)
:   Encoding for the design matrix. `"global"` uses the Walsh-Hadamard (Hadamard) encoding, which supports the fast path and is the standard choice for most analyses. `"local"` uses the biochemical (local) encoding.

`n_jobs` (`int | None`, default `None`)
:   Number of parallel jobs forwarded to the underlying sklearn `LinearRegression` solver. `None` uses one job; `-1` uses all available cores.

## Workflow

1. **Construct the model**

    Pass the desired interaction order and encoding type to the constructor.

    ```python
    from epistasis.models.linear import EpistasisLinearRegression

    model = EpistasisLinearRegression(order=2, model_type="global")
    ```

2. **Attach a genotype-phenotype map**

    Call `add_gpm` with a `GenotypePhenotypeMap` object. This builds the `EpistasisMap` and caches the column layout for the design matrix.

    ```python
    model.add_gpm(gpm)
    ```

3. **Fit**

    Call `fit()` with no arguments to use the phenotypes stored in the attached GPM. Pass explicit `X` and `y` arrays to override.

    ```python
    model.fit()
    ```

4. **Inspect coefficients**

    Read fitted epistatic coefficients and their standard errors from the `EpistasisMap`.

    ```python
    print(model.epistasis.values)       # fitted coefficients
    print(model.epistasis.stdeviations) # analytic OLS standard errors
    ```

## Key methods

### `fit(X=None, y=None)`

Fit the model to the data. When both `X` and `y` are `None`, the method pulls phenotypes from the attached GPM and, where eligible, uses the Walsh-Hadamard fast path. You can pass:

- `None`: use the GPM's genotypes and phenotypes (recommended)
- A 2D NumPy array: treated as a pre-built design matrix
- A 1D array or list of genotype strings: converted to a design matrix automatically

Returns `self` so calls can be chained.

### `predict(X=None)`

Predict phenotypes. Accepts the same `X` forms as `fit`. Returns a 1D `np.ndarray` of predicted phenotype values.

### `score(X=None, y=None)`

Return the R^2 coefficient of determination between predicted and observed phenotypes.

### `hypothesis(X=None, thetas=None)`

Compute `X @ thetas` without modifying any stored state. Useful for exploring how predicted phenotypes change under alternative coefficient vectors.

## Key attributes

| Attribute | Type | Description |
|---|---|---|
| `model.thetas` | `np.ndarray \| None` | Fitted coefficients array, set after `fit()`. |
| `model.coef_` | `np.ndarray` | Alias for `model.thetas`; provided for sklearn API parity. |
| `model.epistasis` | `EpistasisMap` | Map of epistatic sites to fitted values and standard errors. |
| `model.epistasis.values` | `np.ndarray` | Fitted coefficient per site, written by `fit()`. |
| `model.epistasis.stdeviations` | `np.ndarray` | Analytic OLS standard error per coefficient. |

## Complete example

```python
import gpmap
from epistasis.models.linear import EpistasisLinearRegression

# Build a genotype-phenotype map (replace with your own data).
gpm = gpmap.GenotypePhenotypeMap(
    wildtype="AA",
    genotypes=["AA", "AB", "BA", "BB"],
    phenotypes=[1.0, 1.5, 1.8, 2.6],
)

# Fit a second-order (pairwise) epistasis model.
model = EpistasisLinearRegression(order=2, model_type="global")
model.add_gpm(gpm)
model.fit()

# Read results.
print("Epistatic coefficients:", model.epistasis.values)
print("Standard errors:       ", model.epistasis.stdeviations)

# Predict phenotypes for all genotypes in the GPM.
y_pred = model.predict()
print("Predictions:", y_pred)

# Evaluate goodness-of-fit.
r2 = model.score()
print(f"R^2: {r2:.4f}")
```

## Walsh-Hadamard fast path

When you call `fit()` without supplying an explicit `X`, the model checks whether the library is eligible for the Walsh-Hadamard transform:

- The GPM must be **complete** (all 2^L genotypes present for a length-L binary sequence).
- `model_type` must be `"global"`.
- `order` must equal the full sequence length (all interactions included).

When all three conditions hold, the fast path computes coefficients in O(n log n) time using `fwht_ols_coefficients` instead of solving the normal equations. For a fully connected 20-site landscape (2^20, about 1 million genotypes) this translates to a wall-time reduction from hours to seconds.

!!! note

    When the fast path is engaged, the system is exactly determined (n observations, n parameters). Residuals are identically zero, so there are no degrees of freedom available to estimate σ^2. In this case `model.epistasis.stdeviations` is filled with `NaN`. Standard errors are only finite for overdetermined systems (more genotypes than parameters).
