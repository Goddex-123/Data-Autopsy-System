"""
Schema Analyzer and Semantic Type Inference

Infers semantic types for columns beyond pandas dtype.
Distinguishes identifiers, PII, datetime, categorical, etc.
This prevents treating IDs like meaningful numeric distributions.
"""

import re
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import SemanticType, ColumnProfile, DatasetProfile

logger = logging.getLogger(__name__)

# Column name patterns for type inference
_IDENTIFIER_PATTERNS = re.compile(
    r"(?:^|_)(id|uuid|guid|key|pk|code|index|record|serial|number|no|num)(?:$|_)",
    re.IGNORECASE,
)

_DATETIME_PATTERNS = re.compile(
    r"(?:^|_)(date|time|timestamp|created|updated|modified|born|dob|dt|at)(?:$|_)",
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

_PHONE_PATTERN = re.compile(
    r"^[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}$"
)

_IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"
)

_PII_NAME_PATTERNS = re.compile(
    r"(?:^|_)(email|e_mail|phone|tel|mobile|cell|ip|ip_address|"
    r"ssn|social_security|aadhaar|aadhar|pan|passport|"
    r"credit_card|card_number|cvv|"
    r"name|first_name|last_name|full_name|surname|"
    r"address|street|city|zip|postal|zipcode)(?:$|_)",
    re.IGNORECASE,
)


class SchemaAnalyzer:
    """
    Analyzes dataset schema and infers semantic types for each column.

    Semantic type inference prevents misapplication of statistical tests
    (e.g., Benford's Law on ID columns, outlier detection on identifiers).
    """

    def __init__(self, data: pd.DataFrame, sample_size: int = 1000):
        """
        Initialize schema analyzer.

        Args:
            data: The dataset to analyze.
            sample_size: Number of values to sample for pattern detection.
        """
        self.data = data
        self.sample_size = min(sample_size, len(data))

    def analyze(self) -> DatasetProfile:
        """
        Analyze the dataset and return a complete profile.

        Returns:
            DatasetProfile with column-level semantic types.
        """
        columns = {}
        for col in self.data.columns:
            columns[col] = self._profile_column(col)

        total_cells = self.data.size
        missing_cells = int(self.data.isnull().sum().sum())
        dup_rows = int(self.data.duplicated().sum())

        return DatasetProfile(
            row_count=len(self.data),
            column_count=len(self.data.columns),
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_percentage=round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0.0,
            duplicate_rows=dup_rows,
            duplicate_percentage=round(dup_rows / len(self.data) * 100, 2) if len(self.data) > 0 else 0.0,
            memory_usage_mb=round(self.data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            columns=columns,
        )

    def _profile_column(self, col: str) -> ColumnProfile:
        """Build a profile for a single column."""
        series = self.data[col]
        non_null = series.dropna()
        n = len(series)

        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique())

        semantic_type = self._infer_semantic_type(col, series)
        is_id = semantic_type == SemanticType.IDENTIFIER
        is_const = unique_count <= 1 and len(non_null) > 0

        if is_const:
            semantic_type = SemanticType.CONSTANT

        sample_vals = non_null.head(5).tolist() if len(non_null) > 0 else []

        return ColumnProfile(
            name=col,
            pandas_dtype=str(series.dtype),
            semantic_type=semantic_type,
            non_null_count=len(non_null),
            null_count=null_count,
            null_percentage=round(null_count / n * 100, 2) if n > 0 else 0.0,
            unique_count=unique_count,
            unique_percentage=round(unique_count / n * 100, 2) if n > 0 else 0.0,
            sample_values=sample_vals,
            is_identifier=is_id,
            is_constant=is_const,
        )

    def _infer_semantic_type(self, col: str, series: pd.Series) -> SemanticType:
        """
        Infer the semantic type of a column using name patterns,
        value patterns, and statistical heuristics.
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return SemanticType.UNKNOWN

        # 1. Check column name for PII indicators
        if _PII_NAME_PATTERNS.search(col):
            pii_type = self._check_pii_by_name(col)
            if pii_type is not None:
                return pii_type

        # 2. Check column name for identifier patterns
        if self._is_likely_identifier(col, series):
            return SemanticType.IDENTIFIER

        # 3. Check for datetime
        if self._is_likely_datetime(col, series):
            return SemanticType.DATETIME

        # 4. Check for boolean
        if self._is_likely_boolean(series):
            return SemanticType.BOOLEAN

        # 5. Numeric types
        if pd.api.types.is_numeric_dtype(series):
            return self._classify_numeric(col, series)

        # 6. String/object types
        if series.dtype == "object" or pd.api.types.is_string_dtype(series):
            return self._classify_string(col, non_null)

        # 7. Actual datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.DATETIME

        return SemanticType.UNKNOWN

    def _check_pii_by_name(self, col: str) -> Optional[SemanticType]:
        """Check if column name suggests PII."""
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["email", "e_mail"]):
            return SemanticType.PII_EMAIL
        if any(kw in col_lower for kw in ["phone", "tel", "mobile", "cell"]):
            return SemanticType.PII_PHONE
        if any(kw in col_lower for kw in ["ip", "ip_address"]):
            return SemanticType.PII_IP
        if any(kw in col_lower for kw in [
            "ssn", "social_security", "aadhaar", "aadhar", "pan",
            "passport", "credit_card", "card_number", "cvv",
            "name", "first_name", "last_name", "full_name", "surname",
            "address", "street", "zip", "postal", "zipcode",
        ]):
            return SemanticType.PII_OTHER
        return None

    def _is_likely_identifier(self, col: str, series: pd.Series) -> bool:
        """Determine if a column is likely an identifier/primary key."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        # Name-based check
        if _IDENTIFIER_PATTERNS.search(col):
            # Confirm: high uniqueness or sequential integers
            uniqueness = non_null.nunique() / len(non_null)
            if uniqueness > 0.9:
                return True

        # Value-based check: all unique numeric values that look sequential
        if pd.api.types.is_integer_dtype(series):
            uniqueness = non_null.nunique() / len(non_null)
            if uniqueness > 0.99:
                # Check if roughly sequential
                sorted_vals = non_null.sort_values()
                diffs = sorted_vals.diff().dropna()
                if len(diffs) > 0 and diffs.median() == 1.0:
                    return True

        return False

    def _is_likely_datetime(self, col: str, series: pd.Series) -> bool:
        """Check if a column is likely datetime."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        if _DATETIME_PATTERNS.search(col) and series.dtype == "object":
            sample = series.dropna().head(min(self.sample_size, 50))
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().mean()
                return success_rate > 0.8
            except Exception:
                return False

        return False

    def _is_likely_boolean(self, series: pd.Series) -> bool:
        """Check if column is boolean-like."""
        non_null = series.dropna()
        unique_vals = set(non_null.unique())

        boolean_sets = [
            {True, False},
            {0, 1},
            {0.0, 1.0},
            {"true", "false"},
            {"yes", "no"},
            {"y", "n"},
            {"t", "f"},
            {"1", "0"},
        ]

        # Normalize to lowercase strings for comparison
        normalized = {str(v).lower().strip() for v in unique_vals}

        for bool_set in boolean_sets:
            if normalized == {str(v).lower() for v in bool_set}:
                return True

        return False

    def _classify_numeric(self, col: str, series: pd.Series) -> SemanticType:
        """Classify a numeric column as continuous or discrete."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return SemanticType.UNKNOWN

        # Integer type with few unique values relative to count = discrete
        if pd.api.types.is_integer_dtype(series):
            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio < 0.05 and non_null.nunique() <= 20:
                return SemanticType.CATEGORICAL
            return SemanticType.NUMERICAL_DISCRETE

        # Float with few unique values = possibly categorical encoding
        if pd.api.types.is_float_dtype(series):
            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio < 0.01 and non_null.nunique() <= 10:
                return SemanticType.CATEGORICAL

            # Check if all values are whole numbers (stored as float)
            if (non_null == non_null.astype(int)).all():
                return SemanticType.NUMERICAL_DISCRETE

            return SemanticType.NUMERICAL_CONTINUOUS

        return SemanticType.NUMERICAL_CONTINUOUS

    def _classify_string(self, col: str, non_null: pd.Series) -> SemanticType:
        """Classify a string column."""
        unique_count = non_null.nunique()
        n = len(non_null)

        # Check for PII patterns in values
        sample = non_null.head(min(self.sample_size, len(non_null))).astype(str)
        pii_type = self._detect_pii_in_values(sample)
        if pii_type is not None:
            return pii_type

        # Low cardinality = categorical
        if unique_count <= 50 or (unique_count / n < 0.05):
            return SemanticType.CATEGORICAL

        # High cardinality string = text
        if unique_count / n > 0.5:
            # Check average string length
            avg_len = sample.str.len().mean()
            if avg_len > 50:
                return SemanticType.TEXT

        return SemanticType.CATEGORICAL

    def _detect_pii_in_values(self, sample: pd.Series) -> Optional[SemanticType]:
        """Detect PII by checking value patterns in a sample."""
        if len(sample) == 0:
            return None

        str_sample = sample.astype(str)

        # Email check
        email_matches = str_sample.str.match(_EMAIL_PATTERN, na=False).mean()
        if email_matches > 0.5:
            return SemanticType.PII_EMAIL

        # Phone check
        phone_matches = str_sample.str.match(_PHONE_PATTERN, na=False).mean()
        if phone_matches > 0.5:
            return SemanticType.PII_PHONE

        # IP check
        ip_matches = str_sample.str.match(_IP_PATTERN, na=False).mean()
        if ip_matches > 0.5:
            return SemanticType.PII_IP

        return None

    def get_columns_by_type(self, *types: SemanticType) -> List[str]:
        """Get column names matching any of the given semantic types."""
        profile = self.analyze()
        return [
            col for col, cp in profile.columns.items()
            if cp.semantic_type in types
        ]

    def get_analyzable_numeric_columns(self) -> List[str]:
        """
        Get numeric columns suitable for statistical analysis.
        Excludes identifiers, constants, and PII columns.
        """
        profile = self.analyze()
        excluded_types = {
            SemanticType.IDENTIFIER,
            SemanticType.CONSTANT,
            SemanticType.PII_EMAIL,
            SemanticType.PII_PHONE,
            SemanticType.PII_IP,
            SemanticType.PII_OTHER,
        }
        return [
            col for col, cp in profile.columns.items()
            if cp.semantic_type not in excluded_types
            and pd.api.types.is_numeric_dtype(self.data[col])
        ]
