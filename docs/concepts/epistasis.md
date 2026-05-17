---
title: "Understanding epistatic interactions"
description: "Epistasis describes how mutation combinations deviate from additive predictions. Learn how epistasis-v2 quantifies these interactions."
---

# Understanding epistatic interactions

When multiple mutations are combined in the same sequence, their combined effect on phenotype often differs from the sum of their individual effects. This deviation from independent, additive behavior is called epistasis, and it is the central quantity that epistasis-v2 measures. Understanding what epistatic coefficients represent, and how the library organizes them, will help you interpret model output and choose the right interaction order for your experiment.

## What is epistasis?

The simplest model of how mutations affect phenotype assumes independence: each mutation contributes a fixed additive effect regardless of the genetic background. Epistasis captures everything that this additive model fails to predict.

Concretely, consider two mutations at positions 1 and 2. The additive prediction for the double mutant is:

$$
\hat{y}_{\text{double}} \;=\; y_{\text{wt}} \;+\; \Delta_1 \;+\; \Delta_2
$$

where $\Delta_k$ is the additive effect of mutation $k$ alone. If the measured phenotype deviates from this prediction, the deviation is the pairwise (order-2) epistatic coefficient for that pair. Higher-order terms capture deviations that remain even after accounting for all lower-order combinations.

!!! info

    Order-1 terms are the familiar additive (main) effects. Order-2 terms are pairwise interactions. Order-k terms capture interactions among k mutations simultaneously. A full-order model includes all terms up to order L (the sequence length).

## The epistatic coefficient model

epistasis-v2 fits a linear model in epistatic coefficient space:

$$
y \;=\; \beta_0 \;+\; \sum_{i} \beta_i \, x_i \;+\; \sum_{i < j} \beta_{ij} \, x_i x_j \;+\; \sum_{i < j < k} \beta_{ijk} \, x_i x_j x_k \;+\; \cdots
$$

where each $x_i$ is $\pm 1$ (global encoding) or $0/1$ (local encoding), and each $\beta$ coefficient is an epistatic term. The model is fit by assembling a design matrix X and solving the regression with an sklearn estimator.

The two supported encodings, global (Hadamard) and local (biochemical), produce numerically different coefficients that carry different interpretations. See [Global and local encodings explained](encodings.md) for details.

## Interaction order

The `order` parameter you pass to a model constructor controls the highest interaction order included in the fit:

| `order` | Terms included |
|---------|---------------|
| `1` | Intercept + all additive (first-order) effects |
| `2` | All order-1 terms + all pairwise interactions |
| `k` | All terms up to and including k-way interactions |

Higher order captures more of the phenotypic variance but increases the number of parameters exponentially: a length-L biallelic library has `C(L, k)` sites at order k.

!!! note

    For a full-order biallelic library under global encoding, epistasis-v2 engages a Walsh-Hadamard fast path (`O(N log N)`) that avoids building the design matrix entirely. See [The epistasis design matrix](design-matrix.md) for when this applies.

## `EpistasisMap`: the coefficient container

After you call `model.add_gpm(gpm)`, the model builds an `EpistasisMap` at `model.epistasis`. This is a DataFrame-backed container that holds one row per epistatic coefficient. It has five columns:

| Column | Description |
|--------|-------------|
| `labels` | Human-readable coefficient name (e.g. `"A1B"` or `["A1B", "A2B"]`) |
| `orders` | Interaction order: `0` for the intercept, `k` for a k-way interaction |
| `sites` | Tuple of 1-based mutation indices involved in this coefficient |
| `values` | Fitted coefficient values (NaN until after `fit()`) |
| `stdeviations` | Coefficient standard errors (NaN until after `fit()`) |

### Reading fitted coefficients

```python
from gpmap import GenotypePhenotypeMap
from epistasis.models.linear import EpistasisLinearRegression

gpm = GenotypePhenotypeMap(
    wildtype="AAA",
    genotypes=["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"],
    phenotypes=[0.1, 0.4, 0.3, 0.9, 0.2, 0.5, 0.7, 1.2],
    mutations={0: ["A", "B"], 1: ["A", "B"], 2: ["A", "B"]},
)

model = EpistasisLinearRegression(order=2, model_type="global")
model.add_gpm(gpm)
model.fit()

# All fitted coefficient values
print(model.epistasis.values)

# Interaction orders for each coefficient
print(model.epistasis.orders)

# Filter to only first- and second-order terms
first_and_second = model.epistasis.get_orders(1, 2)
print(first_and_second.data)
```

`get_orders(*orders)` returns a new `EpistasisMap` containing only the rows whose `orders` value matches one of the supplied integers.

!!! tip

    To export fitted coefficients for downstream analysis, use `model.epistasis.to_csv("coefficients.csv")` or `model.epistasis.to_dict()`.

## Why high-order interactions matter

In protein engineering, even small high-order epistatic terms can redirect the accessible evolutionary paths through sequence space. A model fit at order 1 or 2 will predict the mean phenotype accurately on average, but may fail badly for specific multi-mutation combinations. When you need to rank variants that carry many simultaneous mutations, as in deep mutational scanning or combinatorial library design, fitting at higher order can meaningfully improve prediction accuracy.
