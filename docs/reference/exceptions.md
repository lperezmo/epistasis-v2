---
title: "epistasis-v2 exceptions reference"
description: "Reference for EpistasisError, FittingError, and XMatrixError. Learn when each exception is raised and how to handle them in your code."
---

# epistasis-v2 exceptions reference

All exceptions raised by `epistasis-v2` inherit from a single base class, `EpistasisError`. You can catch the base class to handle any library error, or catch a specific subclass to handle a particular failure mode.

```python
from epistasis.exceptions import EpistasisError, FittingError, XMatrixError
```

## Exception hierarchy

```
Exception
└── EpistasisError
    ├── FittingError
    └── XMatrixError
```

## `EpistasisError`

The base class for all `epistasis-v2` errors. Inherits directly from the built-in `Exception`.

Catch `EpistasisError` when you want a single handler for any problem the library raises, regardless of its specific cause.

```python
from epistasis.exceptions import EpistasisError

try:
    model.fit()
    predictions = model.predict()
except EpistasisError as exc:
    print(f"epistasis error: {exc}")
```

## `FittingError`

Raised when a model fit has failed or when model methods that require a completed fit are called before `fit()` has been invoked.

**Common triggers:**

- Calling `predict()` or `score()` before `fit()`
- Reading `model.epistasis.thetas` before `fit()`
- A nonlinear minimizer failing to converge (raises `FittingError` rather than surfacing the raw `lmfit` error)

```python
from epistasis.exceptions import FittingError

try:
    predictions = model.predict()
except FittingError:
    # Model has not been fitted yet, fit it first
    model.fit()
    predictions = model.predict()
```

!!! tip

    When catching `FittingError` to trigger a deferred fit, make sure the GPM has already been attached via `model.add_gpm(gpm)` before calling `fit()`. Missing GPM data raises `XMatrixError`, not `FittingError`.

## `XMatrixError`

Raised for an invalid or missing epistasis design matrix. This typically means a model property that depends on the design matrix was accessed before `add_gpm()` was called.

**Common triggers:**

- Accessing `model.gpm`, `model.epistasis`, or `model.Xcolumns` before calling `model.add_gpm(gpm)`
- Passing a design matrix with incompatible shape or type

```python
from epistasis.exceptions import XMatrixError

try:
    cols = model.Xcolumns
except XMatrixError:
    # GPM not yet attached, wire it up
    model.add_gpm(gpm)
    cols = model.Xcolumns
```

## Handling both errors

In practice, you may encounter either error in sequence: `XMatrixError` if the GPM is missing, then `FittingError` if the model has not been fitted.

```python
from epistasis.exceptions import FittingError, XMatrixError

try:
    predictions = model.predict()
except XMatrixError:
    model.add_gpm(gpm)
    model.fit()
    predictions = model.predict()
except FittingError:
    model.fit()
    predictions = model.predict()
```

!!! note

    `XMatrixError` and `FittingError` are both subclasses of `EpistasisError`. If your code catches `EpistasisError` first, the more specific `except` blocks below it will never be reached. Order your handlers from most specific to least specific.
