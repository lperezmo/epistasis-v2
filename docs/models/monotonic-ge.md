---
title: "EpistasisMonotonicGE: monotone global epistasis"
description: "Fit a monotone global-epistasis scale as a sum of tanh sigmoids, the identifiable MAVE-NN-style alternative to the power transform."
---

# `EpistasisMonotonicGE`

`EpistasisMonotonicGE` fits the nonlinear scale as a sum of `K` tanh sigmoids with
non-negative coefficients, which makes the scale **monotone by construction**. It
follows Tareen et al. 2022 (MAVE-NN, Genome Biology 23:98) and is the modern,
identifiable alternative to [`EpistasisPowerTransform`](power-transform.md): the
monotonicity constraint removes the sign ambiguity that hits unconstrained
global-epistasis fits. It is a concrete subclass of
[`EpistasisNonlinearRegression`](nonlinear.md).

![Observed phenotype against the additive score, with the fitted monotone tanh-sum scale](../assets/model-monotonic-ge-light.png#only-light)
![Observed phenotype against the additive score, with the fitted monotone tanh-sum scale](../assets/model-monotonic-ge-dark.png#only-dark)

The fitted curve is guaranteed non-decreasing, so a higher additive score never
maps to a lower predicted phenotype. With enough sigmoids (`K`) it can still
capture sharp saturation, but it can never produce the spurious dips that an
unconstrained or spline fit might.

## When to use this model

Use `EpistasisMonotonicGE` when you believe the measurement scale is monotone (more
additive signal never reduces the measured phenotype) and you want an identifiable,
flexible fit. It is the recommended default for global-epistasis modeling when the
power transform's specific shape is too rigid but a [spline](spline.md) is too
unconstrained.

## How it works

The same two-stage fit as `EpistasisNonlinearRegression`: an order-1 additive model
produces a per-genotype additive score, then a sum of `K` tanh sigmoids,
`f(x) = sum_k a_k * tanh(b_k * x + c_k)` with `b_k, c_k >= 0`, is fit mapping the
additive score to the observed phenotypes. The non-negativity constraints are what
guarantee monotonicity and identifiability.

## Constructor parameters

`K` (`int`, default `5`)
:   Number of tanh sigmoids in the sum. More sigmoids can represent sharper
    saturation at the cost of more parameters.

`monotonic` (`bool`, default `True`)
:   Enforce the non-negativity constraints that make the scale monotone. Set
    `False` to relax them (rarely needed).

`model_type` (`str`, default `"global"`)
:   Encoding for the additive design matrix: `"global"` (Hadamard) or `"local"`.

`seed` (`int | None`, default `0`)
:   Seed for the optimizer's parameter initialization.

## Workflow

```python
from epistasis.models.nonlinear import EpistasisMonotonicGE

model = EpistasisMonotonicGE(K=5, monotonic=True, model_type="global")
model.add_gpm(gpm)
model.fit()

y_pred = model.predict()
y_linear = model.transform()
r2 = model.score()
```

## Key methods and attributes

| Member | Description |
|---|---|
| `fit(X=None, y=None)` | Two-stage fit: additive model, then the monotone tanh-sum scale. |
| `predict(X=None)` | Predicted phenotype on the observed scale. |
| `transform(X=None, y=None)` | Observed phenotypes projected back onto the additive scale. |
| `score(X=None, y=None)` | Pearson R^2 between observed and predicted. |
| `model.minimizer` | `MonotonicGEMinimizer`; holds the fitted sigmoid parameters. |

See the [two-stage fitting guide](nonlinear.md),
[`EpistasisPowerTransform`](power-transform.md), and [`EpistasisSpline`](spline.md).
