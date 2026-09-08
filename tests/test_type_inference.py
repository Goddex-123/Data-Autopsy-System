"""Tests for schema / semantic type inference."""

import pytest
import pandas as pd
import numpy as np

from autopsy.core.schema import SchemaAnalyzer
from autopsy.core.models import SemanticType


class TestSchemaAnalyzer:
    def test_identifier_detection(self, simple_df):
        sa = SchemaAnalyzer(simple_df)
        profile = sa.analyze()
        assert profile.columns["id"].semantic_type == SemanticType.IDENTIFIER

    def test_categorical_detection(self, simple_df):
        sa = SchemaAnalyzer(simple_df)
        profile = sa.analyze()
        assert profile.columns["region"].semantic_type == SemanticType.CATEGORICAL

    def test_boolean_detection(self, simple_df):
        sa = SchemaAnalyzer(simple_df)
        profile = sa.analyze()
        # active column is boolean
        assert profile.columns["active"].semantic_type == SemanticType.BOOLEAN

    def test_numeric_detection(self, simple_df):
        sa = SchemaAnalyzer(simple_df)
        profile = sa.analyze()
        assert profile.columns["income"].semantic_type in (
            SemanticType.NUMERICAL_CONTINUOUS,
            SemanticType.NUMERICAL_DISCRETE,
        )

    def test_constant_detection(self, constant_df):
        sa = SchemaAnalyzer(constant_df)
        profile = sa.analyze()
        assert profile.columns["const_int"].is_constant
        assert profile.columns["const_str"].is_constant

    def test_email_detection(self, df_with_pii):
        sa = SchemaAnalyzer(df_with_pii)
        profile = sa.analyze()
        assert profile.columns["email"].semantic_type == SemanticType.PII_EMAIL

    def test_phone_detection(self, df_with_pii):
        sa = SchemaAnalyzer(df_with_pii)
        profile = sa.analyze()
        assert profile.columns["phone"].semantic_type == SemanticType.PII_PHONE

    def test_ip_detection(self, df_with_pii):
        sa = SchemaAnalyzer(df_with_pii)
        profile = sa.analyze()
        assert profile.columns["ip_address"].semantic_type == SemanticType.PII_IP

    def test_datetime_detection(self):
        df = pd.DataFrame({
            "created_at": pd.date_range("2023-01-01", periods=50),
            "value": range(50),
        })
        sa = SchemaAnalyzer(df)
        profile = sa.analyze()
        assert profile.columns["created_at"].semantic_type == SemanticType.DATETIME

    def test_profile_metrics(self, simple_df):
        sa = SchemaAnalyzer(simple_df)
        profile = sa.analyze()
        assert profile.row_count == 100
        assert profile.column_count == 6
        assert profile.memory_usage_mb >= 0

    def test_analyzable_numeric_excludes_ids(self, simple_df):
        sa = SchemaAnalyzer(simple_df)
        cols = sa.get_analyzable_numeric_columns()
        assert "id" not in cols
        assert "age" in cols or "income" in cols

    def test_empty_dataframe(self, empty_df):
        sa = SchemaAnalyzer(empty_df)
        profile = sa.analyze()
        assert profile.row_count == 0
        assert profile.column_count == 0
