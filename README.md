# epistasis-v2

High-performance Python library for fitting high-order epistatic interactions in genotype-phenotype maps. A clean-break rewrite of the original [harmslab/epistasis](https://github.com/harmslab/epistasis) package.

Status: pre-alpha, active development. See `scratchpad.md` for the full plan.

## What changed from v1

- Rust hot-path kernels via PyO3 (`epistasis._core`) instead of a shipped Cython `.c` blob.
- `uv` + `maturin` build, `pyproject.toml`-only, no `setup.py`.
- Python 3.10 through 3.13. No Python 3.6-3.9.
- Type hints on the public API.
- Walsh-Hadamard fast-path for Hadamard-encoded OLS fits.
- Sparse design matrix path for Lasso / ElasticNet at high order.
- Coordinated rewrite of the [gpmap](https://github.com/harmslab/gpmap) dependency as [gpmap-v2](https://github.com/lperezmo/gpmap-v2).
- No backward compatibility with v1.

## Installation

Not yet published. Local dev install:

```bash
uv sync
uv run maturin develop --release
uv run pytest
```

## License

Unlicense (public domain). See `UNLICENSE`.
