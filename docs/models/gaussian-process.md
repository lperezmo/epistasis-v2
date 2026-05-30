---
title: "EpistasisGaussianProcess: probabilistic viability classification"
description: "Classify genotype viability with a Gaussian process classifier over the additive-projected epistasis design matrix, with calibrated class probabilities."
---

# `EpistasisGaussianProcess`

`EpistasisGaussianProcess` classifies genotypes as viable or nonviable with a
Gaussian process classifier. Its draw over the other classifiers is **calibrated
probability**: rather than a hard boundary, it returns a smoothly varying
`P(viable)` with sensible uncertainty away from the training data. It uses the
same `add_gpm` -> `fit` -> `predict` workflow as every other epistasis model.

![Predicted P(viable) on the genotype graph; node size encodes the classifier's certainty and misclassified genotypes are ringed in red](../assets/model-gaussian-process-light.png#only-light)
![Predicted P(viable) on the genotype graph; node size encodes the classifier's certainty and misclassified genotypes are ringed in red](../assets/model-gaussian-process-dark.png#only-dark)

In the figure each node is a genotype colored by `P(viable)` and sized by how
certain the model is (`|p - 0.5|`); genotypes near the decision boundary stay
small and pale, which is exactly the calibrated-uncertainty behavior a GP buys you.

## When to use this model

Use `EpistasisGaussianProcess` when you want probability estimates you can trust as
probabilities (for ranking, thresholding at a chosen risk level, or propagating
uncertainty) rather than just a class label. It is the most expensive classifier
here because fitting optimizes kernel hyperparameters, so prefer it on small to
medium libraries.

## How it works

1. An order-1 `EpistasisLinearRegression` (`model.additive`) learns each mutation's
   additive contribution from the continuous phenotypes.
2. The design-matrix columns are scaled by those additive coefficients.
3. sklearn's `GaussianProcessClassifier` (RBF kernel by default) is fit on the
   projected matrix with binarized labels (`y > threshold -> 1`).

## Constructor parameters

`threshold` (`float`, required)
:   Phenotype cut-off. Genotypes above it are class 1 (viable).

`model_type` (`str`, default `"global"`)
:   Design-matrix encoding: `"global"` (Hadamard) or `"local"` (biochemical).

`kernel` (`sklearn.gaussian_process.kernels.Kernel | None`, default `None`)
:   Covariance kernel. Defaults to an `RBF()` kernel. Pass your own to encode
    different smoothness assumptions.

`n_restarts_optimizer` (`int`, default `0`)
:   Number of restarts for the kernel-hyperparameter optimizer. Raise it for a more
    thorough (slower) fit.

`max_iter_predict` (`int`, default `100`)
:   Iterations for the Laplace approximation used at prediction time.

`random_state` (`int | None`, default `None`)
:   Seed for reproducible optimization.

## Workflow

```python
import numpy as np
from epistasis.models.classifiers import EpistasisGaussianProcess

model = EpistasisGaussianProcess(
    threshold=float(np.median(gpm.phenotypes)),
    random_state=0,
)
model.add_gpm(gpm)
model.fit()

p_viable = model.predict_proba()[:, 1]  # calibrated P(viable)
labels = model.predict()                # 0 / 1
accuracy = model.score()
```

## Key methods

| Method | Returns | Description |
|---|---|---|
| `fit(X=None, y=None)` | `self` | Fit the additive model then the GP classifier. |
| `predict(X=None)` | `np.ndarray[int]` | Predicted class labels (0 or 1). |
| `predict_proba(X=None)` | `np.ndarray[float]` | Calibrated class probabilities, shape `(n_genotypes, 2)`. |
| `predict_log_proba(X=None)` | `np.ndarray[float]` | Log class probabilities (falls back to `log(predict_proba)`). |
| `score(X=None, y=None)` | `float` | Classification accuracy. |

See also [`EpistasisLogisticRegression`](classifiers.md),
[discriminant analysis](discriminant-analysis.md), and
[`EpistasisGaussianMixture`](gaussian-mixture.md).
