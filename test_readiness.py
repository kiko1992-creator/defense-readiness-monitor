"""
Tests for the readiness model.

These check the pure scoring logic — normalization and edge cases —
without needing the live World Bank API, so they run instantly and
prove the maths is correct.
"""

import pandas as pd
from readiness_model import normalize


def test_normalize_basic():
    """Min-max normalization maps the smallest value to 0 and largest to 1."""
    s = pd.Series([1.0, 2.0, 3.0])
    result = normalize(s)
    assert result.min() == 0.0
    assert result.max() == 1.0


def test_normalize_midpoint():
    """The middle value lands proportionally between 0 and 1."""
    s = pd.Series([0.0, 5.0, 10.0])
    result = normalize(s)
    assert result.iloc[1] == 0.5


def test_normalize_identical_values():
    """When all values are identical, avoid divide-by-zero and return 0.5."""
    s = pd.Series([3.0, 3.0, 3.0])
    result = normalize(s)
    assert (result == 0.5).all()


def test_normalize_output_range():
    """Every normalized value stays within [0, 1]."""
    s = pd.Series([4.48, 2.05, 1.28, 3.37, 1.90])
    result = normalize(s)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    