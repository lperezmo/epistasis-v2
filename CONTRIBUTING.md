# Contributing

PRs welcome, as long as they make damn sense.

Short version: keep it working, keep it typed, keep the commit messages in Conventional Commits format so the release robot can do its job.

## Prerequisites

- Python >= 3.10
- Rust toolchain (`rustup default stable`)
- [`uv`](https://docs.astral.sh/uv/)

`gpmap-v2` is installed from PyPI. If you want to co-develop against a local checkout, add this to `pyproject.toml`:

```toml
[tool.uv.sources]
gpmap-v2 = { path = "../gpmap", editable = true }
```

and drop it before committing.

## Setup

```bash
uv sync
uv run maturin develop --release
```

After edits under `crates/epistasis-core/`, rerun `maturin develop --release`.

## Everyday loop

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy python/epistasis
```

CI runs the same checks on Python 3.10 through 3.13 across Ubuntu, macOS, and Windows. If CI is green, the PR is probably fine.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/). `python-semantic-release` reads your commit messages to decide the next version and generate the changelog, so the format matters.

```
<type>(<scope>): <summary>
```

Bumps the version: `feat` (minor), `fix` (patch), `perf` (patch).
Does not bump: `refactor`, `docs`, `test`, `build`, `ci`, `chore`, `style`, `revert`.

Breaking change: append `!` after the type (`feat!:`, `fix(scope)!:`) and add a `BREAKING CHANGE:` footer. Forces a major bump.

House rules:

- No `Co-Authored-By` footers.
- No em dashes anywhere (commit bodies, code, docs).
- No emojis in code or commit messages.

## Release flow

Push to `main`. GitHub Actions runs `python-semantic-release`, which bumps the version, writes `CHANGELOG.md`, tags, builds wheels for every OS and Python version via `maturin-action`, and publishes to PyPI with OIDC. No manual steps.

## Coordinating with gpmap-v2

This package depends on the gpmap-v2 `GenotypePhenotypeMap` API and `encoding_table` schema. The contract lives in `tests/test_gpmap_bridge.py`. If a gpmap-side change breaks any of those, stop and coordinate.

## Legacy source

`_legacy/` holds the v1 tree for reference during the Phase 1 port. It is gitignored and not shipped. Delete the folder once nothing needs to look back at it.
