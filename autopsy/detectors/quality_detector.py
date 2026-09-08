"""
Data Quality Detector

Comprehensive data quality checks:
- Completeness, validity, consistency, uniqueness
- Constant columns, identifier detection
- Case variant detection, invalid value patterns
"""

import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.models import Finding, Severity, DatasetProfile, SemanticType
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)


class QualityDetector:
    """Detects data quality issues across completeness, validity, and consistency."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
        profile: Optional[DatasetProfile] = None,
    ):
        self.data = data
        self.config = config or AnalysisConfig()
        self.profile = profile

    def analyze(self) -> List[Finding]:
        """Run all quality checks."""
        findings: List[Finding] = []
        findings.extend(self._detect_constant_columns())
        findings.extend(self._detect_duplicate_columns())
        findings.extend(self._detect_case_inconsistency())
        findings.extend(self._detect_whitespace_issues())
        findings.extend(self._detect_mixed_types())
        findings.extend(self._detect_candidate_keys())
        return findings

    def _detect_constant_columns(self) -> List[Finding]:
        """Detect columns with a single unique value."""
        findings = []
        for col in self.data.columns:
            nunique = self.data[col].nunique(dropna=True)
            if nunique <= self.config.constant_column_threshold and len(self.data[col].dropna()) > 0:
                val = self.data[col].dropna().iloc[0] if len(self.data[col].dropna()) > 0 else None
                findings.append(Finding(
                    id=f"DQ-CONST-{col[:20].upper().replace(' ', '_')}",
                    category="data_quality",
                    severity=Severity.MEDIUM,
                    confidence=0.95,
                    title=f"Constant column: '{col}'",
                    description=f"Column contains only {nunique} unique value(s) and provides no analytical value.",
                    column=col,
                    evidence={"unique_values": nunique, "value": str(val)[:100]},
                    recommendation="Consider removing this column from analysis.",
                ))
        return findings

    def _detect_duplicate_columns(self) -> List[Finding]:
        """Detect columns with identical content."""
        findings = []
        cols = list(self.data.columns)
        seen = set()

        for i, col1 in enumerate(cols):
            if col1 in seen:
                continue
            for col2 in cols[i + 1:]:
                if col2 in seen:
                    continue
                try:
                    if self.data[col1].equals(self.data[col2]):
                        findings.append(Finding(
                            id=f"DQ-DUPCOL-{col1[:10]}_{col2[:10]}".upper().replace(" ", "_"),
                            category="data_quality",
                            severity=Severity.MEDIUM,
                            confidence=0.99,
                            title=f"Duplicate columns: '{col1}' ≡ '{col2}'",
                            description="These columns contain identical data.",
                            columns=[col1, col2],
                            evidence={},
                            recommendation="Remove one of the duplicate columns.",
                        ))
                        seen.add(col2)
                except Exception:
                    continue

        return findings

    def _detect_case_inconsistency(self) -> List[Finding]:
        """Detect inconsistent casing in categorical columns (e.g., 'Mumbai' vs 'mumbai')."""
        findings = []
        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            values = self.data[col].dropna().astype(str)
            if len(values) == 0:
                continue

            unique_vals = values.unique()
            if len(unique_vals) > 100:
                continue

            # Group by lowercased version
            lower_map = {}
            for v in unique_vals:
                key = v.lower().strip()
                if key not in lower_map:
                    lower_map[key] = []
                lower_map[key].append(v)

            inconsistencies = {k: v for k, v in lower_map.items() if len(v) > 1}

            if inconsistencies:
                examples = {k: v for k, v in list(inconsistencies.items())[:5]}
                findings.append(Finding(
                    id=f"DQ-CONSISTENCY-CASE-{col[:20].upper().replace(' ', '_')}",
                    category="data_quality",
                    severity=Severity.MEDIUM,
                    confidence=0.9,
                    title=f"Inconsistent casing in '{col}'",
                    description=(
                        f"{len(inconsistencies)} group(s) of values differ only in casing or whitespace."
                    ),
                    column=col,
                    evidence={"variants": examples},
                    recommendation="Normalize casing to ensure consistent categorical values.",
                ))

        return findings

    def _detect_whitespace_issues(self) -> List[Finding]:
        """Detect leading/trailing whitespace in string columns."""
        findings = []
        cat_cols = self.data.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            values = self.data[col].dropna().astype(str)
            if len(values) == 0:
                continue

            has_whitespace = values.str.len() != values.str.strip().str.len()
            ws_count = int(has_whitespace.sum())

            if ws_count > 0:
                pct = ws_count / len(values) * 100
                findings.append(Finding(
                    id=f"DQ-WHITESPACE-{col[:20].upper().replace(' ', '_')}",
                    category="data_quality",
                    severity=Severity.LOW,
                    confidence=0.95,
                    title=f"Whitespace issues in '{col}'",
                    description=f"{ws_count} values ({pct:.1f}%) have leading or trailing whitespace.",
                    column=col,
                    evidence={"affected_count": ws_count, "percentage": round(pct, 2)},
                    recommendation="Strip whitespace to ensure clean joins and comparisons.",
                ))

        return findings

    def _detect_mixed_types(self) -> List[Finding]:
        """Detect columns where values appear to have mixed data types."""
        findings = []

        for col in self.data.select_dtypes(include=["object"]).columns:
            values = self.data[col].dropna().astype(str)
            if len(values) < 10:
                continue

            sample = values.head(min(500, len(values)))
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            numeric_pct = numeric_count / len(sample) * 100

            if 10 < numeric_pct < 90:
                findings.append(Finding(
                    id=f"DQ-MIXEDTYPE-{col[:20].upper().replace(' ', '_')}",
                    category="data_quality",
                    severity=Severity.MEDIUM,
                    confidence=0.7,
                    title=f"Mixed data types in '{col}'",
                    description=(
                        f"Column contains both numeric ({numeric_pct:.0f}%) and "
                        f"non-numeric values, suggesting inconsistent data entry."
                    ),
                    column=col,
                    evidence={"numeric_percentage": round(numeric_pct, 1)},
                    recommendation="Investigate whether this column should be numeric or categorical.",
                ))

        return findings

    def _detect_candidate_keys(self) -> List[Finding]:
        """Identify columns that could serve as primary keys."""
        findings = []

        for col in self.data.columns:
            non_null = self.data[col].dropna()
            if len(non_null) == 0:
                continue

            uniqueness = non_null.nunique() / len(non_null)
            if uniqueness > self.config.high_cardinality_threshold and len(non_null) > 10:
                if self.profile and col in self.profile.columns:
                    if self.profile.columns[col].is_identifier:
                        continue

                findings.append(Finding(
                    id=f"DQ-HIGHCARD-{col[:20].upper().replace(' ', '_')}",
                    category="data_quality",
                    severity=Severity.INFO,
                    confidence=0.6,
                    title=f"High cardinality in '{col}'",
                    description=(
                        f"{uniqueness*100:.1f}% unique values. "
                        f"This column may be an identifier or high-cardinality feature."
                    ),
                    column=col,
                    evidence={
                        "unique_count": int(non_null.nunique()),
                        "uniqueness_ratio": round(uniqueness, 3),
                    },
                    recommendation=(
                        "If this is an identifier, exclude from statistical analysis. "
                        "If categorical, consider grouping rare categories."
                    ),
                ))

        return findings
