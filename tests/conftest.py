"""
Test Fixtures and Factories

Provides deterministic test data for all detector tests.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def simple_df():
    """Small, simple DataFrame for basic tests."""
    return pd.DataFrame({
        "id": range(1, 101),
        "name": [f"user_{i}" for i in range(1, 101)],
        "age": np.random.randint(18, 80, 100),
        "income": np.random.lognormal(10, 0.5, 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "active": np.random.choice([True, False], 100),
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with various missing data patterns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "complete_col": range(n),
        "low_missing": np.where(np.random.random(n) < 0.05, np.nan, np.random.randn(n)),
        "high_missing": np.where(np.random.random(n) < 0.6, np.nan, np.random.randn(n)),
        "sentinel_col": np.where(np.random.random(n) < 0.2, -999, np.random.randn(n)),
        "cat_col": np.where(
            np.random.random(n) < 0.1, None,
            np.random.choice(["A", "B", "C"], n),
        ),
    })
    return df


@pytest.fixture
def df_with_outliers():
    """DataFrame with clear outliers."""
    np.random.seed(42)
    n = 200
    normal = np.random.normal(50, 10, n - 5)
    outliers = np.array([500, -300, 1000, -500, 800])
    values = np.concatenate([normal, outliers])
    np.random.shuffle(values)

    return pd.DataFrame({
        "value": values,
        "normal_col": np.random.normal(100, 15, n),
    })


@pytest.fixture
def df_imbalanced():
    """DataFrame with class imbalance."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "feature": np.random.randn(n),
        "target": np.random.choice(
            ["majority", "minority"], n, p=[0.95, 0.05]
        ),
        "region": np.random.choice(
            ["A", "B", "C", "D"], n, p=[0.7, 0.15, 0.1, 0.05]
        ),
    })


@pytest.fixture
def df_with_pii():
    """DataFrame with PII columns."""
    return pd.DataFrame({
        "user_id": range(1, 51),
        "email": [f"user{i}@example.com" for i in range(1, 51)],
        "phone": [f"+1-555-{i:04d}" for i in range(1, 51)],
        "ip_address": [f"192.168.1.{i}" for i in range(1, 51)],
        "first_name": [f"Name_{i}" for i in range(1, 51)],
        "score": np.random.randn(50),
    })


@pytest.fixture
def df_benford():
    """DataFrame with Benford-applicable data (lognormal spanning magnitudes)."""
    np.random.seed(42)
    return pd.DataFrame({
        "population": np.random.lognormal(10, 2, 500),
        "bounded_var": np.random.uniform(1, 10, 500),
        "record_id": range(1, 501),
    })


@pytest.fixture
def df_leakage():
    """DataFrame with simulated target leakage."""
    np.random.seed(42)
    n = 300
    target = np.random.randn(n)
    return pd.DataFrame({
        "feature_good": np.random.randn(n),
        "feature_leaked": target * 1.01 + np.random.normal(0, 0.01, n),
        "feature_correlated": target * 0.5 + np.random.randn(n),
        "target": target,
    })


@pytest.fixture
def empty_df():
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def single_row_df():
    """DataFrame with a single row."""
    return pd.DataFrame({"a": [1], "b": ["x"], "c": [3.14]})


@pytest.fixture
def constant_df():
    """DataFrame with constant columns."""
    return pd.DataFrame({
        "const_int": [42] * 100,
        "const_str": ["same"] * 100,
        "varied": range(100),
    })


@pytest.fixture
def all_null_df():
    """DataFrame with all-null columns."""
    return pd.DataFrame({
        "all_null": [None] * 50,
        "all_nan": [float("nan")] * 50,
        "some_data": range(50),
    })
