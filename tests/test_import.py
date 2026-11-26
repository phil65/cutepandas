"""Smoke test: the package imports and reports its version."""

import cutepandas


def test_package_imports() -> None:
    """The package can be imported and exposes a version string."""
    assert isinstance(cutepandas.__version__, str)
