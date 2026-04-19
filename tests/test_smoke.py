"""Smoke test: the Rust extension loads and the package imports cleanly."""

import epistasis
from epistasis import _core


def test_version_matches() -> None:
    assert epistasis.__version__
    assert _core.version()


def test_rust_python_version_agree() -> None:
    assert epistasis.__version__ == _core.version()
