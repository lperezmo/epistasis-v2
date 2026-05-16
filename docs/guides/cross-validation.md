---
title: "Cross-validate epistasis models to prevent overfitting"
description: "Use k_fold and holdout from epistasis.validate to evaluate how well your epistasis model generalizes to unseen genotypes and avoid overfitting at high order."
---

# Cross-validate epistasis models to prevent overfitting

Epistasis models with many free parameters, high interaction order or weak regularization, can fit the training phenotypes closely while predicting unseen genotypes poorly. The `epistasis.validate` module provides two cross-validation helpers, `k_fold` and `holdout`, that let you measure generalization performance and choose the interaction order and regularization strength that work best for your dataset.

## Why validate?

A model's training R^2 always increases as you raise the interaction order, because more parameters fit the observed phenotypes more tightly. Cross-validation exposes whether that improvement represents real signal or memorization: you fit the model on one part of the data and score it on another part the model has never seen. When test R^2 levels off or drops while training R^2 keeps rising, you have found the point of overfitting.

## `k_fold`

`k_fold` partitions the GPM into `k` roughly equal folds, trains on `k - 1` of them, and scores on the held-out fold. It repeats this for every fold and returns the list of per-fold R^2 values (Pearson r^2).

```python
from epistasis.validate import k_fold
from epistasis.models.linear import EpistasisLinearRegression

scores = k_fold(gpm, model, k=10, rng=None)
```

### Parameters

`gpm` (`GenotypePhenotypeMap`, required)
:   The full genotype-phenotype map to cross-validate. All genotypes are used; no pre-split is needed.

`model` (`EpistasisBaseModel`, required)
:   A freshly constructed (unfitted) epistasis model. Pass a new instance each time you call `k_fold` so that no fitted state from a previous run leaks into the evaluation.

`k` (`int`, default `10`)
:   Number of folds. Must be in `[2, n]` where `n` is the number of genotypes. Smaller `k` (e.g., 5) gives faster evaluation; larger `k` (e.g., leave-one-out at `k = n`) gives lower-bias estimates but is slower.

`rng` (`np.random.Generator | None`, default `None`)
:   A NumPy random generator used to shuffle genotype indices before splitting into folds. Pass a seeded generator for reproducible fold assignments.

`k_fold` returns a `list[float]`, one R^2 value per fold. Each value is the squared Pearson correlation between the held-out phenotypes and the model's predictions for those genotypes.

!!! note

    R^2 here is **Pearson r^2** (correlation-based), not the coefficient of determination `1 - SSR/SST`. Values close to 1.0 indicate strong predictive performance on held-out genotypes.

## `holdout`

`holdout` performs repeated random train/test splits. On each repeat it fits the model on `fraction` of the genotypes and scores both the training set and the test set. It returns both lists so you can compare train and test R^2 directly.

```python
from epistasis.validate import holdout
from epistasis.models.linear import EpistasisLinearRegression

train_scores, test_scores = holdout(gpm, model, fraction=0.8, repeat=5, rng=None)
```

### Parameters

`gpm` (`GenotypePhenotypeMap`, required)
:   The full genotype-phenotype map.

`model` (`EpistasisBaseModel`, required)
:   A freshly constructed epistasis model.

`fraction` (`float`, default `0.8`)
:   Proportion of genotypes to include in the training set. Must be strictly between 0 and 1. The remainder forms the test set.

`repeat` (`int`, default `5`)
:   Number of independent random splits to perform. More repeats reduce variance in the score estimates.

`rng` (`np.random.Generator | None`, default `None`)
:   A NumPy random generator for reproducible splits.

`holdout` returns `(train_scores, test_scores)`, each a `list[float]` of length `repeat`.

## `split_gpm`

Both `k_fold` and `holdout` use `split_gpm` internally, but you can call it directly when you need manual control over how the data is partitioned.

```python
from epistasis.stats import split_gpm
import numpy as np

# Split by explicit indices
train_gpm, test_gpm = split_gpm(gpm, train_idx=np.array([0, 1, 4, 7]))

# Split by fraction (random)
train_gpm, test_gpm = split_gpm(gpm, fraction=0.8, rng=np.random.default_rng(0))
```

### Parameters

`gpm` (`GenotypePhenotypeMap`, required)
:   The GPM to split.

`train_idx` (`np.ndarray | None`, default `None`)
:   Integer indices (into `gpm.genotypes`) to include in the training set. Exactly one of `train_idx` or `fraction` must be supplied.

`fraction` (`float | None`, default `None`)
:   Proportion of genotypes to include in the training set. Exactly one of `train_idx` or `fraction` must be supplied.

`rng` (`np.random.Generator | None`, default `None`)
:   A NumPy random generator, used only when `fraction` is supplied.

`split_gpm` returns `(train_gpm, test_gpm)`, each a `GenotypePhenotypeMap` containing the appropriate subset of genotypes, phenotypes, and (if present) standard deviations.

## Complete example: choosing interaction order with k-fold CV

The following example runs 5-fold cross-validation across interaction orders 1 through 4, then picks the order with the highest mean test R^2.

```python
import numpy as np
from epistasis.validate import k_fold
from epistasis.models.linear import EpistasisLinearRegression

# Assume `gpm` is your loaded GenotypePhenotypeMap
rng = np.random.default_rng(0)

results = {}
for order in range(1, 5):
    model = EpistasisLinearRegression(order=order, model_type="global")
    fold_scores = k_fold(gpm, model, k=5, rng=np.random.default_rng(0))
    mean_r2 = float(np.mean(fold_scores))
    std_r2 = float(np.std(fold_scores))
    results[order] = (mean_r2, std_r2)
    print(f"Order {order}: mean R^2 = {mean_r2:.3f} +/- {std_r2:.3f}")

best_order = max(results, key=lambda o: results[o][0])
print(f"\nBest order by CV: {best_order}")
```

!!! warning

    Pass a **fresh** model instance to each `k_fold` call. Reusing the same fitted model object will cause the fold evaluation to start from a previously fitted state, which can silently inflate your scores.

!!! tip

    Look for test R^2 that **flattens or drops** as you increase order. That is the signature of overfitting. A small gap between train R^2 (from `holdout`) and test R^2 is normal; a large gap at the same order is a sign to apply regularization with `EpistasisLasso` or `EpistasisRidge` instead.

## Comparing train and test R^2 with holdout

To visualize the train-test gap across orders, combine `holdout` with a simple mean summary:

```python
import numpy as np
from epistasis.validate import holdout
from epistasis.models.linear import EpistasisLinearRegression

for order in range(1, 5):
    model = EpistasisLinearRegression(order=order, model_type="global")
    train_scores, test_scores = holdout(gpm, model, fraction=0.8, repeat=10, rng=np.random.default_rng(1))
    print(
        f"Order {order}: "
        f"train R^2 = {np.mean(train_scores):.3f}, "
        f"test R^2  = {np.mean(test_scores):.3f}"
    )
```

When the train R^2 is high and the test R^2 is considerably lower, the model is memorizing training phenotypes rather than learning transferable epistatic structure.
