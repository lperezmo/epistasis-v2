---
title: "EpistasisSpline: smoothing-spline global epistasis"
description: "Fit an arbitrary smooth nonlinear scale between an additive genetic score and the observed phenotype with a smoothing spline."
---

# `EpistasisSpline`

`EpistasisSpline` fits a smoothing spline as the nonlinear scale between the
additive genetic score and the observed phenotype. Where
[`EpistasisPowerTransform`](power-transform.md) assumes a specific parametric shape,
the spline can follow an **arbitrary smooth scale**, including non-monotonic ones.
It is a concrete subclass of [`EpistasisNonlinearRegression`](nonlinear.md) backed
by `scipy.interpolate.UnivariateSpline`.

![Observed phenotype against the additive score, with the fitted smoothing-spline scale following a non-monotonic shape](../assets/model-spline-light.png#only-light)
![Observed phenotype against the additive score, with the fitted smoothing-spline scale following a non-monotonic shape](../assets/model-spline-dark.png#only-dark)

The example scale dips before rising again. A power transform could not capture
that shape, but the spline tracks it. The trade-off is that the spline is not
monotone by construction, so it can chase noise if under-smoothed.

## When to use this model

Use `EpistasisSpline` when the measurement scale is smooth but its functional form
is unknown or non-monotonic, and you would rather let the data choose the shape
than commit to a parametric transform. If you know the scale is monotone, prefer
[`EpistasisMonotonicGE`](monotonic-ge.md) or
[`EpistasisPowerTransform`](power-transform.md), which cannot produce a spurious
wiggle.

## How it works

The same two-stage fit as `EpistasisNonlinearRegression`: an order-1 additive model
produces a per-genotype additive score, then a univariate smoothing spline is fit
mapping the additive score to the observed phenotypes. Duplicate x-values get
deterministic jitter so the spline solver stays well-posed.

## Constructor parameters

`k` (`int`, default `3`)
:   Spline degree. `3` is a cubic smoothing spline.

`s` (`float | None`, default `None`)
:   Smoothing factor passed to `UnivariateSpline`. Smaller values follow the data
    more closely (and risk overfitting); larger values smooth more. `None` lets
    scipy choose from the data.

`model_type` (`str`, default `"global"`)
:   Encoding for the additive design matrix: `"global"` (Hadamard) or `"local"`.

`seed` (`int | None`, default `0`)
:   Seed for the deterministic jitter applied to duplicate x-values.

## Workflow

```python
from epistasis.models.nonlinear import EpistasisSpline

model = EpistasisSpline(k=3, model_type="global")
model.add_gpm(gpm)
model.fit()

y_pred = model.predict()
y_linear = model.transform()
r2 = model.score()
```

## Key methods and attributes

| Member | Description |
|---|---|
| `fit(X=None, y=None)` | Two-stage fit: additive model, then the smoothing spline. |
| `predict(X=None)` | Predicted phenotype on the observed scale. |
| `transform(X=None, y=None)` | Observed phenotypes projected back onto the additive scale. |
| `score(X=None, y=None)` | Pearson R^2 between observed and predicted. |
| `model.minimizer` | `SplineMinimizer`; wraps the fitted `UnivariateSpline`. |

See the [two-stage fitting guide](nonlinear.md),
[`EpistasisPowerTransform`](power-transform.md), and
[`EpistasisMonotonicGE`](monotonic-ge.md).
