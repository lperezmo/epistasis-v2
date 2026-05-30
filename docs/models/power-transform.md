---
title: "EpistasisPowerTransform: Box-Cox global epistasis"
description: "Fit a Box-Cox-style nonlinear scale between an additive genetic score and the observed phenotype, the Sailer and Harms (2017) global-epistasis model."
---

# `EpistasisPowerTransform`

`EpistasisPowerTransform` is a ready-made nonlinear epistasis model: it fits a
Box-Cox-style power transform as the scale between the additive genetic score and
the observed phenotype. It is the Sailer and Harms (2017) global-epistasis model,
and a concrete subclass of [`EpistasisNonlinearRegression`](nonlinear.md) with the
nonlinear function already supplied, so you do not pass a `function` of your own.

![Observed phenotype against the additive score, with the fitted power-transform scale](../assets/model-power-light.png#only-light)
![Observed phenotype against the additive score, with the fitted power-transform scale](../assets/model-power-dark.png#only-dark)

The fitted red curve is the learned scale: it maps the stage-1 additive prediction
(x-axis) to the observed measurement scale (y-axis), absorbing the global
nonlinearity so the additive coefficients underneath stay interpretable.

## When to use this model

Use `EpistasisPowerTransform` when you believe the measurement scale is a smooth,
monotonic power-law-like distortion of an underlying additive landscape, and you
want a parametric, identifiable transform rather than hand-writing one for
[`EpistasisNonlinearRegression`](nonlinear.md). For non-monotonic or otherwise
arbitrary scales, use [`EpistasisSpline`](spline.md); for a monotone neural-style
scale, use [`EpistasisMonotonicGE`](monotonic-ge.md).

## How it works

It runs the same two-stage fit as `EpistasisNonlinearRegression`: an order-1
additive model produces a per-genotype additive score, then the power transform
`f(x; lmbda, A, B)` is fit so that `f(additive.predict())` matches the observed
phenotypes. The training-set geometric-mean reference is locked at fit time, so
predictions on new genotypes reuse the same scale.

## Constructor parameters

`model_type` (`str`, default `"global"`)
:   Encoding for the additive design matrix: `"global"` (Hadamard) or `"local"`.

`lmbda`, `A`, `B` (`float | None`, default `None`)
:   Optional fixed values for the power-transform parameters. Leave as `None` to
    fit them; set one to hold it constant.

## Workflow

```python
from epistasis.models.nonlinear import EpistasisPowerTransform

model = EpistasisPowerTransform(model_type="global")
model.add_gpm(gpm)
model.fit()

y_pred = model.predict()         # phenotype on the observed (nonlinear) scale
y_linear = model.transform()     # phenotypes linearized onto the additive scale
r2 = model.score()
additive_coefs = model.additive.epistasis.values
```

## Key methods and attributes

| Member | Description |
|---|---|
| `fit(X=None, y=None)` | Two-stage fit: additive model, then the power transform. |
| `predict(X=None)` | Predicted phenotype on the observed scale. |
| `transform(X=None, y=None)` | Observed phenotypes projected back onto the additive scale. |
| `score(X=None, y=None)` | Pearson R^2 between observed and predicted. |
| `model.additive` | The internal order-1 `EpistasisLinearRegression`. |
| `model.minimizer` | `PowerTransformMinimizer`; holds the fitted scale parameters. |

See the [two-stage fitting guide](nonlinear.md) for the shared mechanics,
[`EpistasisSpline`](spline.md), and [`EpistasisMonotonicGE`](monotonic-ge.md).
