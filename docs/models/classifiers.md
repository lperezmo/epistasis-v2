---
title: "EpistasisLogisticRegression for binary phenotypes"
description: "Predict genotype viability with logistic regression over the epistasis design matrix. Uses the same add_gpm, fit, and predict workflow as linear models."
---

# EpistasisLogisticRegression for binary phenotypes

Some genotype-phenotype experiments produce binary outcomes rather than continuous measurements: a protein either folds or it does not, a variant is viable or lethal, a sequence binds or fails to bind. `EpistasisLogisticRegression` handles these datasets by fitting a logistic regression classifier over the epistasis design matrix, predicting whether each genotype falls above or below a user-defined phenotype threshold.

## When to use this model

Use `EpistasisLogisticRegression` when:

- Your phenotype is binary (viable/nonviable, functional/nonfunctional, binding/non-binding).
- You want to predict class membership, not a continuous phenotype value, from the genotype.
- You have a natural cut-off threshold that separates the two classes.

If your phenotype is continuous, use `EpistasisLinearRegression` or one of the regularized linear models instead.

## How it works

The classifier uses an internal two-step procedure adapted from the base classifier class:

1. An order-1 `EpistasisLinearRegression` (`model.additive`) is fit to the **continuous** phenotype values to learn each mutation's additive contribution.
2. The design matrix columns are scaled by the fitted additive coefficients. This projects the design matrix onto a per-mutation contribution scale, giving the logistic classifier interpretable, comparable features.
3. sklearn's `LogisticRegression` is fit on the projected matrix, using binarized labels (`y > threshold -> 1`, else `0`).

Predictions and probabilities are made on this same projected feature space.

!!! note

    The design matrix is the same Walsh-Hadamard or local-encoded matrix used by all other epistasis models, just scaled by the additive coefficients before being handed to the logistic solver. You do not need to build or transform the matrix yourself.

## Constructor parameters

`threshold` (`float`, required)
:   Phenotype cut-off. Genotypes with a continuous phenotype value strictly greater than `threshold` are assigned class 1 (viable); all others are class 0.

`model_type` (`str`, default `"global"`)
:   Encoding for the design matrix: `"global"` (Hadamard) or `"local"` (biochemical).

`C` (`float`, default `1.0`)
:   Inverse regularization strength, following sklearn's convention. Smaller values apply stronger regularization. Equivalent to `1 / alpha` in the linear regularized models.

`max_iter` (`int`, default `1000`)
:   Maximum solver iterations. Increase if the solver raises a convergence warning.

`solver` (`str`, default `"lbfgs"`)
:   sklearn `LogisticRegression` solver. `"lbfgs"` is a good default for most problems. Other options include `"liblinear"`, `"newton-cg"`, `"newton-cholesky"`, `"sag"`, and `"saga"`.

`random_state` (`int | None`, default `None`)
:   Random seed for solvers that use randomness.

## Workflow

The workflow is identical to the linear models: attach a GPM, fit, then predict.

1. **Construct the classifier**

    ```python
    from epistasis.models.classifiers import EpistasisLogisticRegression

    model = EpistasisLogisticRegression(threshold=0.5, model_type="global")
    ```

2. **Attach a genotype-phenotype map**

    ```python
    model.add_gpm(gpm)
    ```

3. **Fit**

    ```python
    model.fit()
    ```

    This fits the internal additive model and then the logistic classifier in sequence.

4. **Predict class labels or probabilities**

    ```python
    # Predicted class labels (0 or 1) for each genotype.
    labels = model.predict()

    # Class probabilities: shape (n_genotypes, 2).
    # Column 0 = P(class 0), column 1 = P(class 1 / viable).
    proba = model.predict_proba()

    # Accuracy on the training data.
    accuracy = model.score()
    print(f"Training accuracy: {accuracy:.4f}")
    ```

## Key methods

| Method | Returns | Description |
|---|---|---|
| `fit(X=None, y=None)` | `self` | Fit the additive model then the logistic classifier. |
| `predict(X=None)` | `np.ndarray[int]` | Predicted class labels (0 or 1) for each genotype. |
| `predict_proba(X=None)` | `np.ndarray[float]` | Class probabilities, shape `(n_genotypes, 2)`. |
| `predict_log_proba(X=None)` | `np.ndarray[float]` | Log class probabilities, shape `(n_genotypes, 2)`. |
| `score(X=None, y=None)` | `float` | Fraction of correctly classified genotypes (accuracy). |
| `hypothesis(X=None, thetas=None)` | `np.ndarray[float]` | Probability of class 1 for each row of `X`. |

## Complete example

```python
import gpmap
from epistasis.models.classifiers import EpistasisLogisticRegression

# Genotypes with continuous phenotypes; some above threshold, some below.
gpm = gpmap.GenotypePhenotypeMap(
    wildtype="AA",
    genotypes=["AA", "AB", "BA", "BB"],
    phenotypes=[0.1, 0.4, 0.6, 0.9],
)

# Classify as viable (1) if phenotype > 0.5, nonviable (0) otherwise.
model = EpistasisLogisticRegression(threshold=0.5, model_type="global")
model.add_gpm(gpm)
model.fit()

# Predicted labels for all genotypes in the GPM.
labels = model.predict()
print("Class labels:", labels)  # e.g. [0, 0, 1, 1]

# Probability of being viable (class 1).
proba = model.predict_proba()
print("P(viable):", proba[:, 1])

# Training accuracy.
print(f"Accuracy: {model.score():.4f}")

# Predict for new genotypes not in the original GPM.
new_labels = model.predict(X=["AB", "BB"])
new_proba = model.predict_proba(X=["AB", "BB"])
```

## Alpha classifiers: deferred models

LDA (Linear Discriminant Analysis), QDA (Quadratic Discriminant Analysis), Gaussian Process, and Gaussian Mixture Model classifiers were part of the v1 roadmap. They are not included in v2 and are deferred pending user demand. If your use case requires one of these classifiers, open an issue in the repository describing your application.
