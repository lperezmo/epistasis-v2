---
title: "epistasis-v2 changelog and version history"
description: "Track breaking changes, new features, and bug fixes across epistasis-v2 releases. Covers the full rewrite from v1 and all subsequent versions."
---

# epistasis-v2 changelog and version history

`epistasis-v2` follows [Semantic Versioning](https://semver.org/). Each release includes a corresponding PyPI wheel with pre-built Rust extensions for all supported platforms.

!!! warning

    `epistasis-v2` has **no backward compatibility** with v1 (`harmslab/epistasis`). If you depend on v1 behavior, pin the original package: `epistasis==0.7.5`. The two packages cannot be installed in the same environment.

## v1.1.1 (2026-05-03)

Tags: Bug Fix, Chore

### Bug fixes

- **benchmarks**: Fixed a loop-variable capture bug in benchmark lambdas where all closures were referencing the same final loop value. Also wrapped an overlong docstring line.

### Streamlit showcase app

A multi-page Streamlit demo was added under `examples/` and published at [epistasis-v2.streamlit.app](https://epistasis-v2.streamlit.app/). The app includes seven pages:

- **Overview**: 3D hypercube of sequence space with phenotype-colored vertices, switchable between `L=3` and `L=4` (tesseract projection).
- **Design matrix**: interactive `L` / order / encoding controls, Plotly heatmaps of `X` and `X^T X` showing Hadamard orthogonality.
- **Linear fit**: simulate known coefficients, fit, scatter predicted vs. observed, coefficient bars with analytic OLS standard errors.
- **FWHT fast path**: live benchmark of dense `lstsq` against `fwht_ols_coefficients`, asymptotic-cost curves, `L` slider capped at 12.
- **Simulate**: synthetic GPM from `simulate_random_linear_gpm`, phenotype histogram, genotype preview.
- **Cross validation**: per-fold R^2 bars from `epistasis.validate.k_fold`.
- **About**: install instructions, links, credits.

Dark and light theme support was added to the Streamlit app, along with an interaction site grid and a dedicated `examples/` dependency group that does not affect the core dev or CI environments.

## v1.1.0 (2026-04-20)

Tags: Feature, Performance

### Rust hot-path kernels (`epistasis._core`)

Design-matrix construction and the FWHT were ported to a Rust crate (`epistasis-core`) and exposed via PyO3. The pure-NumPy implementations remain in `_reference.py` as a correctness oracle verified by the parity test suite on every fit.

Three kernels shipped:

- **`encode_vectors`**: parallel int8 conversion from `uint8` binary_packed; serial threshold at 2^18 cells to skip `rayon` overhead on small inputs.
- **`build_model_matrix`**: ragged sites packed as flat int64 indices plus int64 offsets; parallel over output columns with a serial threshold at 2^15 cells. Measured 3x to 6x faster than the NumPy reference across `L` from 8 to 16.
- **`fwht`**: iterative in-place butterfly, zero extra dependencies.

### FWHT fast path in `EpistasisLinearRegression`

`EpistasisLinearRegression.fit()` now detects full-order biallelic libraries with global (Hadamard) encoding and solves OLS in `O(n log n)` via the Fast Walsh-Hadamard Transform. The check is bitmask-based: genotype bitmasks must cover `[0, 2^L)` bijectively and site bitmasks must do the same.

After a successful FWHT fit, the sklearn estimator's fitted attributes (`coef_`, `intercept_`, `n_features_in_`) are synced so `predict` and `score` continue to work through the existing composition boundary. Standard errors are `NaN` because the system is exactly determined.

**Benchmark, full-order fit (v2 FWHT vs. `numpy.linalg.lstsq`):**

| L  | Genotypes | `lstsq` (ms) | FWHT (ms) | Speedup   |
|----|-----------|--------------|-----------|-----------|
| 10 | 1,024     | 3,005        | 3.10      | ~969x     |
| 12 | 4,096     | 59,344       | 8.97      | >6,000x   |
| 14 | 16,384    | hours        | 35.50     |           |
| 16 | 65,536    | hours        | 154.15    |           |

The new `epistasis.fast` module exposes `fwht_ols_coefficients` as a reusable utility for custom pipelines and benchmarking.

## v1.0.0 (2026-04-20): initial rewrite

Tags: Release, Breaking Change

v1.0.0 is the first production release of the clean-break rewrite. All modules below were ported from [harmslab/epistasis](https://github.com/harmslab/epistasis) with updated APIs, type hints, and a composition-over-MRO-injection design.

### Ported Python modules

??? note "epistasis.mapping"

    Sites, coefficients, and `EpistasisMap`. `EpistasisMapReference` was collapsed into `EpistasisMap.get_orders`, which returns a fresh `EpistasisMap` instead of a view-like proxy. Reads `site_index` (not the deprecated `genotype_index`) from `gpmap-v2`'s `encoding_table`.

??? note "epistasis.matrix"

    Encoded vectors and design matrix construction. `encode_vectors` converts `uint8` `binary_packed` to int8 Hadamard or local encoding. `build_model_matrix` produces the dense design matrix as int8. The NumPy implementations serve as correctness references; the Rust kernels in `epistasis._core` are the production path as of v1.1.0.

??? note "epistasis.exceptions"

    `EpistasisError`, `XMatrixError`, and `FittingError`. Renamed from v1's `*Exception` classes for PEP 8 consistency.

??? note "epistasis.utils"

    `genotypes_to_X` helper. Reads `binary_packed` from `gpmap-v2` instead of parsing string binary representations.

??? note "epistasis.models.base"

    `AbstractEpistasisModel` (three abstract methods: `fit`, `predict`, `hypothesis`) and `EpistasisBaseModel`. Composition replaces the `@use_sklearn` MRO injection that broke with sklearn >= 1.2. The base model carries `add_gpm`, property guards, resolver logic for `X`/`y`/`yerr`, and a Gaussian log-likelihood default.

??? note "epistasis.models.linear"

    - `EpistasisLinearRegression`: OLS with analytic coefficient standard errors via `σ_hat^2 (X'X)^{-1}`. FWHT fast path added in v1.1.0.
    - `EpistasisRidge`: L2-regularized linear regression.
    - `EpistasisLasso`: L1-regularized; `compression_ratio` tracks sparsity.
    - `EpistasisElasticNet`: mixed L1 + L2. Fixes a v1 bug where `l1_ratio` was silently overwritten to `1.0`.

??? note "epistasis.models.nonlinear"

    `EpistasisNonlinearRegression` and `FunctionMinimizer`. Two-stage fit: an order-1 additive linear model approximates average per-mutation effects, then a user-supplied `f(x, *params)` is fit via Levenberg-Marquardt (`lmfit`). `power.py` and `spline.py` variants are deferred.

??? note "epistasis.models.classifiers"

    `EpistasisLogisticRegression` for viability classification. Binarizes observed phenotypes at a threshold and uses the projected additive design matrix as features. LDA, QDA, Gaussian Process, and GMM classifiers are deferred.

??? note "epistasis.simulate"

    `simulate_linear_gpm` and `simulate_random_linear_gpm`. Functional API: no `BaseSimulation` subclass hierarchy or mutable `.build()` hooks. Supports ternary and higher-alphabet positions.

??? note "epistasis.stats"

    `pearson`, `r_squared`, `rmsd`, `ss_residuals`, `aic`, `split_gpm`. Dropped v1 functions that were unused (`gmean`, `incremental_*`), redundant (`explained_variance`), or brittle (`chi_squared`, false-rate helpers).

??? note "epistasis.validate"

    `k_fold(gpm, model, k, rng)` returns per-fold R^2 scores. `holdout(gpm, model, fraction, repeat, rng)` returns train and test score lists.

??? note "epistasis.sampling.bayesian"

    `BayesianSampler` wraps a fitted model via the `SamplerModel` protocol (`thetas` and `lnlikelihood(thetas=)`). Modernized for `emcee` 3: uses `State` objects, `run_mcmc` with positional initial state, and `reset()` to drop burn-in samples.

??? note "epistasis.fast"

    `fwht_ols_coefficients`: closed-form OLS via the Fast Walsh-Hadamard Transform. Returns `None` when the fast path does not apply so callers can fall back transparently.

### Rust kernels in `epistasis._core`

- `encode_vectors`: uint8 binary_packed to int8 Hadamard/local encoding
- `build_model_matrix`: parallel site-product over genotype rows; flat ragged sites layout
- `fwht`: iterative butterfly Fast Walsh-Hadamard Transform

## Pending items

The following items are planned but not yet shipped:

- Sparse design matrix path for Lasso / ElasticNet (memory concern at `L >= 20`)
- `power.py` and `spline.py` nonlinear variants
- Remaining classifier implementations (LDA, QDA, Gaussian Process, GMM) if demand surfaces
- ReadTheDocs build
