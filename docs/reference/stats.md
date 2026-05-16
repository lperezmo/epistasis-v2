---
title: "Statistics functions reference"
description: "Reference for epistasis.stats: pearson, r_squared, rmsd, ss_residuals, aic, and split_gpm. Use these to evaluate and compare fitted epistasis models."
---

# Statistics functions reference

The `epistasis.stats` module provides a focused set of model evaluation metrics and a genotype-phenotype map splitter. You can use these functions to assess fit quality, compare competing models with information criteria, and partition your data into training and test sets for cross-validation.

```python
from epistasis.stats import pearson, r_squared, rmsd, aic, split_gpm, ss_residuals
```

!!! note

    All metric functions accept any array-like input and internally cast to `float64`. Pass NumPy arrays, lists, or any other sequence that NumPy can convert.

## Metric functions

### `pearson`

Computes the Pearson correlation coefficient between observed and predicted phenotypes.

`y_obs` (`np.ndarray`, required)
:   Observed phenotype values. Shape must match `y_pred`.

`y_pred` (`np.ndarray`, required)
:   Model-predicted phenotype values. Shape must match `y_obs`.

**Returns** (`float`)
:   Pearson correlation coefficient in the range `[-1, 1]`. Raises `ValueError` if `y_obs` and `y_pred` have different shapes.

```python
from epistasis.stats import pearson

r = pearson(y_obs, model.predict())
print(f"Pearson r = {r:.4f}")
```

### `r_squared`

Computes the coefficient of determination `1 - SSR / SST`.

`y_obs` (`np.ndarray`, required)
:   Observed phenotype values.

`y_pred` (`np.ndarray`, required)
:   Model-predicted phenotype values.

**Returns** (`float`)
:   R^2 value. Returns `NaN` when `ss_tot == 0` (all observed values are identical), because the ratio is undefined.

!!! warning

    `r_squared` returns `float("nan")` when all observed values are the same (zero variance). Check for this case before comparing models.

```python
from epistasis.stats import r_squared

r2 = r_squared(y_obs, model.predict())
print(f"R^2 = {r2:.4f}")
```

### `rmsd`

Computes the root mean square deviation between observed and predicted phenotypes.

`y_obs` (`np.ndarray`, required)
:   Observed phenotype values.

`y_pred` (`np.ndarray`, required)
:   Model-predicted phenotype values.

**Returns** (`float`)
:   RMSD in the same units as the phenotype. Lower is better.

```python
from epistasis.stats import rmsd

error = rmsd(y_obs, model.predict())
print(f"RMSD = {error:.4f}")
```

### `ss_residuals`

Computes the residual sum of squares (SSR).

`y_obs` (`np.ndarray`, required)
:   Observed phenotype values.

`y_pred` (`np.ndarray`, required)
:   Model-predicted phenotype values.

**Returns** (`float`)
:   Sum of squared residuals `Σ(y_obs - y_pred)^2`. Used internally by `r_squared`.

```python
from epistasis.stats import ss_residuals

ssr = ss_residuals(y_obs, model.predict())
print(f"SSR = {ssr:.4f}")
```

### `aic`

Computes the Akaike Information Criterion: `AIC = 2 * (k - ln L)`.

`model` (`Any`, required)
:   A fitted epistasis model. Must expose two attributes:

    - `num_of_params` (int): number of free parameters `k`
    - `lnlikelihood()` (callable -> float): log-likelihood of the fitted model

**Returns** (`float`)
:   AIC score. Lower values indicate a better trade-off between fit quality and model complexity.

!!! tip

    Use AIC to compare models fit to the same dataset. The model with the lowest AIC is preferred. The absolute value of AIC is not meaningful on its own.

```python
from epistasis.stats import aic

aic_linear = aic(linear_model)
aic_nonlinear = aic(nonlinear_model)

if aic_linear < aic_nonlinear:
    print("Linear model preferred by AIC")
else:
    print("Nonlinear model preferred by AIC")
```

## Data splitting

### `split_gpm`

Splits a `GenotypePhenotypeMap` into training and test subsets.

!!! warning

    Exactly one of `train_idx` or `fraction` must be provided. Passing both, or neither, raises `ValueError`.

`gpm` (`GenotypePhenotypeMap`, required)
:   The full genotype-phenotype map to split.

`train_idx` (`np.ndarray | None`, default `None`)
:   Integer array of indices to include in the training set. The remaining indices become the test set. Keyword-only.

`fraction` (`float | None`, default `None`)
:   Fraction of genotypes to assign to the training set. Must be in `(0, 1)`. Genotypes are sampled without replacement using a random shuffle. Keyword-only.

`rng` (`np.random.Generator | None`, default `None`)
:   NumPy random generator for reproducibility. Only used when `fraction` is provided. Keyword-only.

**Returns** (`tuple[GenotypePhenotypeMap, GenotypePhenotypeMap]`)
:   A `(train_gpm, test_gpm)` pair. Each is a fresh `GenotypePhenotypeMap` built from the selected subset, preserving `wildtype`, `mutations`, and `stdeviations`.

```python
from epistasis.stats import split_gpm
import numpy as np

# Reproducible 80/20 split by fraction
rng = np.random.default_rng(42)
train_gpm, test_gpm = split_gpm(gpm, fraction=0.8, rng=rng)

# Explicit index split
train_idx = np.array([0, 1, 2, 5, 7])
train_gpm, test_gpm = split_gpm(gpm, train_idx=train_idx)
```

## Typical usage

The example below shows a full evaluation workflow: fit a model, compute multiple metrics, and compare two models with AIC.

```python
import numpy as np
from epistasis.stats import pearson, r_squared, rmsd, aic, split_gpm
from epistasis.models.linear import EpistasisLinearRegression
from epistasis.models.nonlinear import EpistasisNonlinearRegression

# Split data for out-of-sample evaluation
rng = np.random.default_rng(0)
train_gpm, test_gpm = split_gpm(gpm, fraction=0.8, rng=rng)

# Fit a linear model
linear = EpistasisLinearRegression(order=1)
linear.add_gpm(train_gpm)
linear.fit()

# Evaluate on the held-out test set
y_obs = test_gpm.phenotypes
y_pred = linear.predict(test_gpm.genotypes)

print(f"Pearson r  : {pearson(y_obs, y_pred):.4f}")
print(f"R^2        : {r_squared(y_obs, y_pred):.4f}")
print(f"RMSD       : {rmsd(y_obs, y_pred):.4f}")

# Compare models on the training set using AIC
def saturation(x, A, K):
    return A * x / (K + x)

nonlinear = EpistasisNonlinearRegression(saturation, order=1)
nonlinear.add_gpm(train_gpm)
nonlinear.fit()

print(f"Linear AIC    : {aic(linear):.2f}")
print(f"Nonlinear AIC : {aic(nonlinear):.2f}")
```
