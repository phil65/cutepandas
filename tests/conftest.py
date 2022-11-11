"""Tests for `datacook` package."""
from __future__ import annotations

import pytest

import pandas as pd
from prettyqt import widgets


@pytest.fixture(scope="session")
def qapp_cls():
    return widgets.Application


@pytest.fixture
def df():
    return pd.DataFrame(dict(a=["a", "b", "c"], b=["d", "e", "f"]))
