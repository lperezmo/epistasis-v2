---
title: "Walsh-Hadamard fast path reference"
description: "fwht_ols_coefficients solves full-order OLS in O(n log n) via the Fast Walsh-Hadamard Transform, eliminating the need to build the dense design matrix."
---

# Walsh-Hadamard fast path reference

The `epistasis.fast` module exposes `fwht_ols_coefficients`, a closed-form OLS solver for full-order biallelic libraries encoded under the global (Hadamard) scheme. Instead of constructing the `n x n` design matrix and calling a dense least-squares solver, it exploits the Hadamard structure to solve for all `2^L` coefficients in `O(n log n)` operations.

```python
from epistasis.fast import fwht_ols_coefficients
```

!!! note

    You rarely need to call this function directly. `EpistasisLinearRegression.fit()` automatically engages the fast path when the attached GPM satisfies the conditions below. The function is exposed as a public utility for benchmarking and custom pipelines.

## When the fast path engages

`EpistasisLinearRegression.fit()` checks four conditions before invoking `fwht_ols_coefficients`. If any condition fails the function returns `None` and the standard sklearn least-squares path runs unchanged.

??? note "model_type must be 'global'"

    The Hadamard transform is only valid for global (Hadamard) encoding where `0 -> +1` and `1 -> -1`. Passing any other `model_type` string returns `None` immediately.

??? note "Complete biallelic library: n == 2^L"

    Every distinct bit pattern across `L` positions must be present exactly once. The check computes genotype bitmasks and verifies they cover `[0, 2^L)` bijectively. Partial libraries, duplicate genotypes, and multi-allelic positions all return `None`.

??? note "Full order: exactly 2^L interaction sites"

    The `sites` list must contain exactly `2^L` entries with distinct bitmasks that also cover `[0, 2^L)` bijectively. Any truncated order (e.g., fitting only up to order 3 on an `L = 12` library) returns `None`.

??? note "n_bits in [1, 62]"

    Libraries larger than `2^62` are not supported (and do not fit in memory). Zero-bit inputs are rejected as degenerate.

## Performance

The table below shows measured fit times from `benchmarks/vs_v1.py` on a Windows 11 workstation comparing v2 FWHT against dense `numpy.linalg.lstsq`.

| L  | Genotypes | Dense lstsq | FWHT fast path | Speedup   |
|----|-----------|-------------|----------------|-----------|
| 8  | 256       | 195 ms      | 1.75 ms        | ~111x     |
| 10 | 1,024     | 3,005 ms    | 3.10 ms        | ~969x     |
| 12 | 4,096     | 59,344 ms   | 8.97 ms        | >6,000x   |
| 14 | 16,384    | hours       | 35.50 ms       |           |
| 16 | 65,536    | hours       | 154.15 ms      |           |

The speedup is asymptotic: dense OLS scales as `O(n^3)` in the solve step while FWHT scales as `O(n log n)`.

!!! tip

    Memory also scales favorably. Dense OLS allocates an `n x n` float64 design matrix; at `L = 16` that is roughly 32 GB. The FWHT path allocates only a handful of float64 vectors.

## Rust kernel

The inner butterfly transform is implemented in Rust as `epistasis._core.fwht`:

```python
def fwht(data: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

The kernel runs an iterative in-place butterfly on the input vector and returns a new `float64` array of the same length. `fwht_ols_coefficients` calls it once and divides by `n` to recover the Walsh-Hadamard coefficients.

## `fwht_ols_coefficients`

`binary_packed` (`np.ndarray`, required)
:   uint8 array of shape `(n, n_bits)` with values in `{0, 1}`. Must represent a complete combinatorial biallelic library: `n == 2 ** n_bits` with every distinct bit pattern present exactly once. Obtain this from `gpm.binary_packed`.

`y` (`np.ndarray`, required)
:   float64 phenotype array of shape `(n,)`. Row order must match `binary_packed`.

`sites` (`Sequence[Site]`, required)
:   Interaction sites in the same order used by the design matrix. Must contain exactly `2 ** n_bits` entries whose bitmasks cover `[0, 2^n_bits)` bijectively (full order). Obtain via `encoding_to_sites` from `epistasis.mapping`.

`model_type` (`str`, default `"global"`)
:   Encoding type. Only `"global"` is supported; any other value returns `None` without raising.

**Returns** (`np.ndarray[float64] | None`)
:   Coefficient array of shape `(len(sites),)` in the same order as `sites`, or `None` when the fast path does not apply. When `None` is returned, fall back to your preferred dense solver.

## Direct usage example

```python
import numpy as np
from epistasis.fast import fwht_ols_coefficients
from epistasis.mapping import encoding_to_sites

# Assume `gpm` is a GenotypePhenotypeMap with a complete L=10 biallelic library
binary_packed = gpm.binary_packed          # shape (1024, 10), dtype uint8
y = gpm.phenotypes.astype(np.float64)      # shape (1024,)
sites = encoding_to_sites(10, gpm.encoding_table)  # 1024 sites for full order at L=10

beta = fwht_ols_coefficients(binary_packed, y, sites, model_type="global")

if beta is None:
    print("Fast path did not apply; use a dense solver.")
else:
    print(f"Solved {len(beta)} coefficients via FWHT")
    print(f"Intercept (site 0): {beta[0]:.6f}")
```

## Using via `EpistasisLinearRegression`

The most common way to benefit from the fast path is to attach a full GPM and call `fit()` with no arguments. The fast path engages automatically when conditions are met, and you can verify it engaged by checking that standard errors are `NaN` (the system is exactly determined and `(X'X)^{-1}` is not computed).

```python
from epistasis.models.linear import EpistasisLinearRegression

model = EpistasisLinearRegression(order=10)  # full order for L=10
model.add_gpm(gpm)
model.fit()  # FWHT fast path engages automatically for full biallelic libraries

# Standard errors are NaN for exactly-determined FWHT fits
print(model.epistasis.stdeviations)

# predict and score work normally through the sklearn composition boundary
r2 = model.score()
print(f"R^2 = {r2:.4f}")
```
