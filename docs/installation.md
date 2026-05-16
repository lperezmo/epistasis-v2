---
title: "Install epistasis-v2 from PyPI or source"
description: "Install epistasis-v2 from PyPI to get pre-built wheels with Rust extensions, or compile from source with a Rust toolchain. Supports Python 3.10 through 3.13."
---

# Install epistasis-v2

epistasis-v2 is distributed as a source distribution and a set of platform wheels on PyPI. The wheels bundle pre-compiled Rust extensions, so most users can install the package with a single `pip` command and no Rust toolchain. This page covers both paths and lists every runtime dependency.

## Prerequisites

- **Python 3.10 or later.** Supported versions: 3.10, 3.11, 3.12, 3.13.
- **pip** (any recent version). No Rust toolchain is needed when installing from a PyPI wheel.

=== "PyPI install"

    Install the latest release from PyPI:

    ```bash
    pip install epistasis-v2
    ```

    This pulls a pre-built wheel for your platform. The Rust extension (`epistasis._core`) is already compiled inside the wheel, so the install completes without a Rust toolchain.

    ### Verify the installation

    Open a Python interpreter and confirm the package is importable:

    ```python
    import epistasis
    print(epistasis.__version__)
    ```

    !!! tip

        If you are working inside a virtual environment or conda environment, activate it before running `pip install`.

=== "Build from source"

    Building from source lets you work with the latest unreleased code or contribute to the library. You need:

    - A **Rust toolchain** (stable channel). Install via [rustup.rs](https://rustup.rs).
    - **uv** for dependency management. Install via `pip install uv` or the [uv docs](https://docs.astral.sh/uv/).

    Clone the repository and build:

    ```bash
    git clone https://github.com/lperezmo/epistasis-v2
    cd epistasis-v2
    uv sync
    uv run maturin develop --release
    ```

    `uv sync` installs all Python dependencies (including dev extras). `maturin develop --release` compiles the Rust crate and links it into the active environment in release mode.

    ### Verify the build

    ```bash
    uv run python -c "import epistasis; print(epistasis.__version__)"
    ```

    !!! warning

        The `--release` flag is important. Without it, the Rust kernels compile in debug mode and run significantly slower.

## Runtime dependencies

All dependencies below are installed automatically when you run `pip install epistasis-v2`.

| Package | Minimum version | Purpose |
| --- | --- | --- |
| `numpy` | 1.23 | Array operations and matrix algebra |
| `pandas` | 2.0 | Tabular storage for the `EpistasisMap` |
| `scipy` | 1.11 | Statistical distributions and linear algebra utilities |
| `scikit-learn` | 1.3 | Underlying estimators for linear and regularized models |
| `lmfit` | 1.2 | Parameter fitting for nonlinear models |
| `emcee` | 3.1 | MCMC ensemble sampler for Bayesian coefficient estimation |
| `matplotlib` | 3.7 | Plotting utilities |
| `gpmap-v2` | 1.0.0 | `GenotypePhenotypeMap` objects and encoding tables |

!!! note

    `gpmap-v2` is the companion library that provides `GenotypePhenotypeMap`. epistasis-v2 models accept GPM objects directly through `add_gpm()`. The two libraries are versioned and released together; install a matching pair to avoid compatibility issues.

## Python version support

epistasis-v2 is tested on Python 3.10, 3.11, 3.12, and 3.13 in CI. Python versions older than 3.10 are not supported.

## Upgrading

To upgrade to the latest release:

```bash
pip install --upgrade epistasis-v2
```
