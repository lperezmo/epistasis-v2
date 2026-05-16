---
title: "epistasis-v2: High-Performance Epistasis Fitting"
description: "epistasis-v2 fits high-order epistatic interactions in genotype-phenotype maps. Rust kernels and a FWHT fast path deliver massive speedups over dense OLS."
---

# epistasis-v2

epistasis-v2 is a high-performance Python library for fitting epistatic interactions in genotype-phenotype maps (GPMs). It provides linear and nonlinear regression models, Bayesian sampling, and simulation utilities, all backed by Rust-accelerated kernels and a Walsh-Hadamard fast path that delivers up to 6,000x speedup over the previous generation at full interaction order.

<div class="grid cards" markdown>

-   **Quick Start**

    ---

    Install the library and fit your first epistasis model in minutes.

    [Quick Start](quickstart.md)

-   **Installation**

    ---

    Requirements, pip install, and build from source with Rust.

    [Installation](installation.md)

-   **Core Concepts**

    ---

    Understand genotype-phenotype maps, epistasis, and design matrices.

    [Core Concepts](concepts/genotype-phenotype-maps.md)

-   **Models**

    ---

    Linear, regularized, nonlinear, and classifier epistasis models.

    [Models](models/linear.md)

-   **Simulation**

    ---

    Generate synthetic GPMs with known epistatic coefficients.

    [Simulation](guides/simulation.md)

-   **Cross-Validation**

    ---

    Evaluate model performance with k-fold and holdout validation.

    [Cross-Validation](guides/cross-validation.md)

-   **Bayesian Sampling**

    ---

    Quantify parameter uncertainty with MCMC ensemble sampling.

    [Bayesian Sampling](guides/bayesian-sampling.md)

-   **API Reference**

    ---

    Complete reference for stats, fast paths, and exceptions.

    [API Reference](reference/stats.md)

</div>

## Why epistasis-v2?

1. **Install**

    ```bash
    pip install epistasis-v2
    ```

2. **Load your data**

    Wrap your genotypes and phenotypes in a `GenotypePhenotypeMap` from `gpmap-v2`.

3. **Fit a model**

    Choose a model (`EpistasisLinearRegression`, `EpistasisLasso`, or `EpistasisNonlinearRegression`), attach your GPM, and call `.fit()`.

4. **Inspect coefficients**

    Read fitted epistatic coefficients and standard errors from `model.epistasis.values` and `model.epistasis.stdeviations`.

!!! note

    epistasis-v2 is currently in **alpha**. The public API is stable for the ported modules, but some features (sparse Lasso at very high order, power/spline nonlinear variants) are still in progress.
