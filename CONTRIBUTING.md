# Contributing

Thanks for considering a contribution. epistasis-v2 uses a small set of conventions so that releases and changelog updates are automatic. The whole pipeline, from your commit on `main` to a published PyPI wheel, is driven by the commit message format.

## Prerequisites

- Python >= 3.10
- Rust toolchain (`rustup default stable`)
- `uv` (install from https://docs.astral.sh/uv/)
- A local checkout of `gpmap-v2` at `../gpmap` during pre-release co-development (this is wired via `[tool.uv.sources]` in `pyproject.toml`)

## First-time setup

```bash
uv sync                           # install runtime + dev deps, builds the Rust ext
uv run maturin develop --release  # rebuild the Rust ext after any Rust change
```

## Day-to-day loop

```bash
uv run pytest                     # run the test suite
uv run ruff check .               # lint
uv run ruff format .              # autoformat
uv run mypy python/epistasis      # type-check (strict)
```

After any edit under `crates/epistasis-core/`, rerun `maturin develop --release` to rebuild the extension.

## Commit conventions

This repo uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Every commit on `main` is consumed by `python-semantic-release`, which decides the next version number and generates the changelog entry.

Format:

```
<type>(<scope>): <summary>

<body>
```

Types that bump the version:

| Type    | Bump  | Use for                               |
|---------|-------|---------------------------------------|
| `feat`  | minor | user-visible new capability           |
| `fix`   | patch | bug fix                               |
| `perf`  | patch | performance change with same outcome  |

Types that do not bump the version (CHANGELOG only, or silent):

| Type       | Use for                                 |
|------------|-----------------------------------------|
| `refactor` | internal restructuring, no behavior     |
| `docs`     | doc / comment edits only                |
| `test`     | test-only changes                       |
| `build`    | build config                            |
| `ci`       | GitHub Actions workflows                |
| `chore`    | tooling, deps, housekeeping             |
| `style`    | formatting, no code change              |
| `revert`   | revert of a prior commit                |

Breaking changes: append `!` after the type, e.g. `feat(mapping)!: rename encoding_to_sites to sites_from_encoding`, and include a `BREAKING CHANGE:` footer explaining the migration. This forces a major bump.

Scope is optional but strongly preferred. Typical scopes:

- `mapping`, `matrix`, `models`, `simulate`, `sampling`, `stats`
- `core` for Rust kernel work
- `ci`, `build`, `deps`

Examples from this repo:

```
feat(matrix): port design-matrix construction with NumPy backend
feat(mapping): port mapping module against gpmap-v2 schema
feat: wire gpmap-v2 as editable dep and add API contract tests
```

Do not add `Co-Authored-By` footers and do not use em dashes in commit messages or code.

## Pull request checklist

Before opening a PR:

- [ ] `uv run pytest` passes locally on Python 3.10 and latest (we test 3.10 through 3.13 in CI).
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run mypy python/epistasis` clean (strict mode).
- [ ] Commit messages follow Conventional Commits.
- [ ] Any behavior change has a test.
- [ ] Any schema change coordinated with `gpmap-v2`, with a bridge test in `tests/test_gpmap_bridge.py` updated accordingly.

## Release flow

Maintainers push to `main`. GitHub Actions then:

1. Runs `python-semantic-release` to decide the next version from commit history, writes `CHANGELOG.md`, bumps `pyproject.toml` and `python/epistasis/_version.py`, tags `vX.Y.Z`.
2. Builds wheels on Linux / macOS (x86_64 and aarch64) / Windows for Python 3.10 through 3.13 via `maturin-action`, plus an sdist.
3. Publishes to PyPI with OIDC trusted publishing.
4. Uploads all artifacts to the GitHub Release.

No manual `pip publish`, no keys to manage, no version file to edit.

## Project layout

See the "Repository layout" section in `README.md`.

## Coordinating schema changes with gpmap-v2

`epistasis-v2` depends on gpmap-v2's `GenotypePhenotypeMap` API and `encoding_table` schema. The contract is encoded in `tests/test_gpmap_bridge.py`. If a change on the gpmap side breaks any of those tests, stop and coordinate before shipping.

## Legacy source

`_legacy/` holds the v1 tree for reference during porting. It is gitignored and not part of the distribution. When Phase 1 is complete and no module needs to look back at it, delete the folder.
