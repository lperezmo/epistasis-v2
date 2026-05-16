---
title: "Ridge, Lasso, and ElasticNet models"
description: "Prevent overfitting at high interaction order with L2 (Ridge), L1 (Lasso), or mixed L1+L2 (ElasticNet) regularization over the epistasis design matrix."
---

# Ridge, Lasso, and ElasticNet models

When you fit a high-order epistasis model, the number of interaction terms grows combinatorially with sequence length. Without regularization, the model can overfit the data, capturing noise rather than genuine epistatic signal. The three regularized models in `epistasis.models.linear` address this by adding a penalty to the loss function that shrinks coefficients toward zero. All three share the same `add_gpm` -> `fit` -> `predict` -> `score` workflow as `EpistasisLinearRegression`, with no changes to how you attach data or read results.

!!! tip

    Choose your regularizer based on what you expect about the underlying biology. If you expect most interactions to be small but non-zero, use Ridge. If you expect most high-order interactions to be absent (a sparse landscape), use Lasso. If you are unsure, ElasticNet gives you a tunable blend of both behaviors.

## Choosing alpha

All three models expose an `alpha` parameter that controls regularization strength. Larger values of `alpha` apply stronger shrinkage, producing simpler models with smaller (or exactly zero) coefficients. Smaller values approach the unregularized OLS solution.

!!! note

    The right value of `alpha` depends on your data. Use k-fold cross-validation via `epistasis.validate.k_fold` to select `alpha` empirically rather than guessing.

## EpistasisRidge

`EpistasisRidge` applies an L2 penalty: it minimizes the residual sum of squares plus `alpha * sum(coef^2)`. L2 regularization shrinks all coefficients smoothly toward zero but rarely drives any coefficient to exactly zero. It is a good default when you believe most interactions are real but small.

### Constructor parameters

`order` (`int`, default `1`)
:   Maximum interaction order to fit.

`model_type` (`str`, default `"global"`)
:   Design-matrix encoding: `"global"` (Hadamard) or `"local"` (biochemical).

`alpha` (`float`, default `1.0`)
:   L2 penalty strength. Setting `alpha=0` recovers OLS; prefer `EpistasisLinearRegression` in that case.

`max_iter` (`int | None`, default `None`)
:   Maximum iterations for iterative solvers. `None` lets sklearn choose a default based on the solver.

`tol` (`float`, default `1e-4`)
:   Convergence tolerance.

`solver` (`str`, default `"auto"`)
:   sklearn Ridge solver. One of `"auto"`, `"svd"`, `"cholesky"`, `"lsqr"`, `"sparse_cg"`, `"sag"`, `"saga"`, or `"lbfgs"`. `"auto"` picks a solver based on the data type and size.

`random_state` (`int | None`, default `None`)
:   Random seed for solvers that use randomness (`"sag"`, `"saga"`).

### Ridge example

```python
from epistasis.models.linear import EpistasisRidge

model = EpistasisRidge(order=3, alpha=0.1)
model.add_gpm(gpm)
model.fit()

y_pred = model.predict()
r2 = model.score()
print(f"Ridge R^2: {r2:.4f}")
print("Coefficients:", model.epistasis.values)
```

## EpistasisLasso

`EpistasisLasso` applies an L1 penalty: it minimizes the residual sum of squares plus `alpha * sum(|coef|)`. Unlike L2, the L1 penalty drives many coefficients to exactly zero, producing a sparse set of epistatic interactions. After fitting, call `model.compression_ratio()` to see what fraction of coefficients were zeroed out.

### Constructor parameters

`order` (`int`, default `1`)
:   Maximum interaction order to fit.

`model_type` (`str`, default `"global"`)
:   Design-matrix encoding: `"global"` or `"local"`.

`alpha` (`float`, default `1.0`)
:   L1 penalty strength. Setting `alpha=0` is OLS and triggers a sklearn warning; prefer `EpistasisLinearRegression` for the unregularized case.

`max_iter` (`int`, default `1000`)
:   Maximum coordinate-descent iterations. Increase if the solver reports a convergence warning.

`tol` (`float`, default `1e-4`)
:   Convergence tolerance on the dual gap.

`warm_start` (`bool`, default `False`)
:   When `True`, reuse the previous solution as the starting point on repeated calls to `fit`. Useful when scanning a path of `alpha` values.

`positive` (`bool`, default `False`)
:   Force all fitted coefficients to be non-negative.

`selection` (`str`, default `"cyclic"`)
:   Coordinate update strategy: `"cyclic"` or `"random"`. Random selection often converges faster when `tol` is loose.

`random_state` (`int | None`, default `None`)
:   Seed used when `selection="random"`.

`precompute` (`bool`, default `False`)
:   Whether to use a precomputed Gram matrix to speed up coordinate descent for dense problems.

### Lasso example

```python
from epistasis.models.linear import EpistasisLasso

model = EpistasisLasso(order=4, alpha=0.05)
model.add_gpm(gpm)
model.fit()

y_pred = model.predict()
print(f"Compression ratio: {model.compression_ratio():.1%}")  # fraction zeroed
print("Non-zero coefficients:", model.epistasis.values[model.epistasis.values != 0])
```

## EpistasisElasticNet

`EpistasisElasticNet` blends L1 and L2 regularization. The `l1_ratio` parameter sets the balance: `l1_ratio=1.0` is pure Lasso, `l1_ratio=0.0` is pure Ridge, and values in between give a mix of sparsity-inducing and coefficient-shrinking behavior.

!!! warning

    A bug in epistasis-v1 silently overwrote `l1_ratio` to `1.0` during `__init__`, making the model behave as pure Lasso regardless of user input. This is fixed in v2. If you are migrating from v1, re-check any `ElasticNet` fits that relied on `l1_ratio < 1.0`.

### Constructor parameters

`order` (`int`, default `1`)
:   Maximum interaction order to fit.

`model_type` (`str`, default `"global"`)
:   Design-matrix encoding: `"global"` or `"local"`.

`alpha` (`float`, default `1.0`)
:   Combined penalty strength applied to the L1 + L2 blend.

`l1_ratio` (`float`, default `0.5`)
:   Fraction of the penalty that is L1. Must be in `[0, 1]`. Values outside this range raise a `ValueError`.

`max_iter` (`int`, default `1000`)
:   Maximum coordinate-descent iterations.

`tol` (`float`, default `1e-4`)
:   Convergence tolerance on the dual gap.

`warm_start` (`bool`, default `False`)
:   Reuse the previous solution on repeated `fit` calls.

`positive` (`bool`, default `False`)
:   Force all fitted coefficients to be non-negative.

`selection` (`str`, default `"cyclic"`)
:   Coordinate update strategy: `"cyclic"` or `"random"`.

`random_state` (`int | None`, default `None`)
:   Seed for `selection="random"`.

`precompute` (`bool`, default `False`)
:   Use a precomputed Gram matrix.

### ElasticNet example

```python
from epistasis.models.linear import EpistasisElasticNet

# 70% L1, 30% L2 mix.
model = EpistasisElasticNet(order=3, alpha=0.1, l1_ratio=0.7)
model.add_gpm(gpm)
model.fit()

y_pred = model.predict()
r2 = model.score()
print(f"ElasticNet R^2: {r2:.4f}")
print(f"Compression ratio: {model.compression_ratio():.1%}")
```

## Shared methods

All three regularized models inherit from `RegularizedLinearBase` and expose the same interface:

| Method | Description |
|---|---|
| `fit(X=None, y=None)` | Fit the model. Returns `self`. |
| `predict(X=None)` | Predict phenotypes; returns a 1D `np.ndarray`. |
| `score(X=None, y=None)` | Return R^2 between predicted and observed phenotypes. |
| `hypothesis(X=None, thetas=None)` | Evaluate `X @ thetas` without modifying stored state. |
| `compression_ratio()` | Fraction of fitted coefficients that are exactly zero. Most meaningful for Lasso and ElasticNet. |

The `coef_` property mirrors `thetas` for sklearn API parity. `epistasis.values` is populated after every `fit` call.

!!! note

    Regularized models do not compute analytic standard errors. `model.epistasis.stdeviations` is not set by `fit`. If you need uncertainty estimates for regularized coefficients, use bootstrap resampling or Bayesian methods.
