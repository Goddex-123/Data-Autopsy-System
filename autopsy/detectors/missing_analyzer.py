"""
Missing Data Analyzer — Refactored

Scientifically correct missing data analysis that:
- Never claims "Likely MCAR" from failure to detect MAR
- Explicitly states MNAR cannot be established from observed data
- Reports evidence strength rather than mechanism labels
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.models import Finding, Severity
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)


class MissingDataAnalyzer:
    """
    Analyzes missing data patterns with scientifically correct language
    about missingness mechanisms.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
    ):
        self.data = data
        self.config = config or AnalysisConfig()

    def analyze(self) -> List[Finding]:
        """Run complete missing data analysis."""
        findings: List[Finding] = []
        findings.extend(self._analyze_column_missingness())
        findings.extend(self._analyze_missing_patterns())
        findings.extend(self._analyze_missing_mechanisms())
        findings.extend(self._detect_sentinel_values())
        findings.extend(self._detect_coverage_gaps())
        return findings

    # ── Column-Level Missingness ─────────────────────────────────────

    def _analyze_column_missingness(self) -> List[Finding]:
        """Analyze missingness at the column level."""
        findings = []

        for col in self.data.columns:
            missing_count = int(self.data[col].isnull().sum())
            if missing_count == 0:
                continue

            missing_pct = missing_count / len(self.data) * 100

            if missing_pct >= self.config.missing_critical_threshold:
                severity = Severity.CRITICAL
            elif missing_pct >= self.config.missing_high_threshold:
                severity = Severity.HIGH
            elif missing_pct >= 5:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            findings.append(Finding(
                id=f"MISS-COL-{col[:20].upper().replace(' ', '_')}",
                category="missing_data",
                severity=severity,
                evidence_score=0.95,
                title=f"Missing values in '{col}'",
                description=f"{missing_count:,} values ({missing_pct:.1f}%) are missing.",
                column=col,
                evidence={
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_pct, 2),
                    "present_count": len(self.data) - missing_count,
                },
                recommendation=(
                    "Investigate the cause of missingness. Consider whether imputation, "
                    "exclusion, or indicator variables are appropriate."
                ),
            ))

        return findings

    # ── Missing Patterns ─────────────────────────────────────────────

    def _analyze_missing_patterns(self) -> List[Finding]:
        """Detect patterns: columns missing together, sequential gaps."""
        findings = []
        missing_matrix = self.data.isnull()
        cols_with_missing = [c for c in self.data.columns if missing_matrix[c].any()]

        # Columns missing together
        if len(cols_with_missing) >= 2:
            for i, col1 in enumerate(cols_with_missing):
                for col2 in cols_with_missing[i + 1:]:
                    both = (missing_matrix[col1] & missing_matrix[col2]).sum()
                    either = (missing_matrix[col1] | missing_matrix[col2]).sum()
                    if either > 0:
                        together_rate = both / either
                        if together_rate > 0.7:
                            findings.append(Finding(
                                id=f"MISS-TOGETHER-{col1[:10]}_{col2[:10]}".upper().replace(" ", "_"),
                                category="missing_data",
                                severity=Severity.MEDIUM,
                                evidence_score=round(together_rate, 2),
                                title=f"Correlated missingness: '{col1}' ↔ '{col2}'",
                                description=(
                                    f"These columns tend to be missing together "
                                    f"({together_rate*100:.0f}% co-occurrence rate). "
                                    f"This suggests a shared cause of missingness."
                                ),
                                columns=[col1, col2],
                                evidence={
                                    "co_missing_rate": round(together_rate, 3),
                                    "both_missing_count": int(both),
                                    "either_missing_count": int(either),
                                },
                                recommendation="Investigate whether these columns share a data source or collection mechanism.",
                            ))

        # Sequential missing patterns
        for col in cols_with_missing:
            mask = missing_matrix[col].values
            runs = self._find_runs(mask)
            long_runs = [r for r in runs if r["length"] >= 5]

            if long_runs:
                max_run = max(r["length"] for r in long_runs)
                findings.append(Finding(
                    id=f"MISS-SEQ-{col[:20].upper().replace(' ', '_')}",
                    category="missing_data",
                    severity=Severity.MEDIUM,
                    evidence_score=0.7,
                    title=f"Sequential missing values in '{col}'",
                    description=(
                        f"{len(long_runs)} run(s) of ≥5 consecutive missing values detected. "
                        f"Longest run: {max_run} rows."
                    ),
                    column=col,
                    evidence={
                        "long_runs_count": len(long_runs),
                        "max_run_length": max_run,
                    },
                    recommendation="Sequential gaps may indicate data collection interruptions or systematic omissions.",
                ))

        return findings

    @staticmethod
    def _find_runs(mask: np.ndarray) -> List[dict]:
        """Find runs of True values in a boolean array."""
        runs = []
        in_run = False
        run_start = 0

        for i, val in enumerate(mask):
            if val and not in_run:
                in_run = True
                run_start = i
            elif not val and in_run:
                in_run = False
                runs.append({"start": run_start, "length": i - run_start})

        if in_run:
            runs.append({"start": run_start, "length": len(mask) - run_start})

        return runs

    # ── Missingness Mechanisms ───────────────────────────────────────

    def _analyze_missing_mechanisms(self) -> List[Finding]:
        """
        Analyze evidence for MCAR vs MAR.

        IMPORTANT: We NEVER claim to detect MNAR, because MNAR cannot be
        established from observed data alone.
        """
        findings = []
        missing_cols = [c for c in self.data.columns if self.data[c].isnull().any()]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in missing_cols:
            missing_mask = self.data[col].isnull()
            n_missing = missing_mask.sum()
            n_present = (~missing_mask).sum()

            if n_missing < 10 or n_present < 10:
                continue

            # Test for MAR: check if missingness relates to other variables
            mar_evidence = []

            for other_col in numeric_cols:
                if other_col == col:
                    continue

                present_vals = self.data.loc[~missing_mask, other_col].dropna()
                missing_vals = self.data.loc[missing_mask, other_col].dropna()

                if len(present_vals) < 10 or len(missing_vals) < 10:
                    continue

                try:
                    stat, p_value = stats.mannwhitneyu(
                        present_vals, missing_vals, alternative="two-sided"
                    )
                    if p_value < 0.05:
                        mar_evidence.append({
                            "related_column": other_col,
                            "p_value": round(p_value, 4),
                            "test": "Mann-Whitney U",
                        })
                except ValueError:
                    continue

            if mar_evidence:
                findings.append(Finding(
                    id=f"MISS-MAR-{col[:20].upper().replace(' ', '_')}",
                    category="missing_data",
                    severity=Severity.MEDIUM,
                    evidence_score=round(min(0.8, 0.5 + len(mar_evidence) * 0.1), 2),
                    title=f"Evidence of non-random missingness in '{col}'",
                    description=(
                        f"Missingness in '{col}' appears related to {len(mar_evidence)} "
                        f"other variable(s). This suggests the data may be Missing At Random (MAR) "
                        f"— i.e., missingness depends on observed variables."
                    ),
                    column=col,
                    evidence={
                        "mar_relationships": mar_evidence[:5],
                        "missing_count": int(n_missing),
                    },
                    statistical_test="Mann-Whitney U test (missingness vs. other variables)",
                    recommendation=(
                        "Consider using imputation methods that account for MAR patterns "
                        "(e.g., multiple imputation, regression imputation)."
                    ),
                    limitations=(
                        "This test detects relationships between missingness and observed "
                        "variables. It CANNOT distinguish MAR from MNAR. MNAR (Missing Not "
                        "At Random) can never be ruled out from observed data alone."
                    ),
                ))
            else:
                findings.append(Finding(
                    id=f"MISS-MCAR-{col[:20].upper().replace(' ', '_')}",
                    category="missing_data",
                    severity=Severity.INFO,
                    evidence_score=0.4,
                    title=f"No strong MAR evidence for '{col}'",
                    description=(
                        f"No statistically significant relationship found between "
                        f"missingness in '{col}' and other observed variables. "
                        f"Insufficient evidence to reject MCAR."
                    ),
                    column=col,
                    evidence={"missing_count": int(n_missing)},
                    recommendation="Treat as potentially MCAR, but remember this does not prove randomness.",
                    limitations=(
                        "Failure to detect MAR DOES NOT prove MCAR. The test has limited "
                        "power with small samples and cannot detect nonlinear relationships. "
                        "MNAR remains a strong possibility that cannot be ruled out."
                    ),
                ))

        return findings

    # ── Sentinel Values ──────────────────────────────────────────────

    def _detect_sentinel_values(self) -> List[Finding]:
        """Detect values likely used as placeholders for missing data."""
        findings = []

        numeric_sentinels = [0, -1, -999, 999, 9999, -9999, 99, -99]
        string_sentinels = [
            "NA", "N/A", "NULL", "None", "", " ", "nan",
            "Unknown", "UNKNOWN", "missing", "MISSING", "-", ".",
        ]

        for col in self.data.columns:
            values = self.data[col]

            if values.dtype in ["int64", "float64"]:
                value_counts = values.value_counts()
                for sentinel in numeric_sentinels:
                    if sentinel in value_counts.index:
                        count = int(value_counts[sentinel])
                        pct = count / len(values) * 100
                        if pct > self.config.sentinel_frequency_threshold and count > 10:
                            findings.append(Finding(
                                id=f"MISS-SENTINEL-{col[:20].upper().replace(' ', '_')}",
                                category="missing_data",
                                severity=Severity.MEDIUM,
                                evidence_score=0.7,
                                title=f"Possible sentinel value in '{col}'",
                                description=(
                                    f"Value {sentinel} appears {count} times ({pct:.1f}%). "
                                    f"This may represent disguised missing data."
                                ),
                                column=col,
                                evidence={
                                    "sentinel_value": sentinel,
                                    "count": count,
                                    "percentage": round(pct, 2),
                                },
                                recommendation="Verify whether this value represents actual data or is a placeholder.",
                                limitations="Context is required. Some sentinel-like values (e.g. 0 or 99) may be valid measurements depending on the domain."
                            ))
                            break

            elif values.dtype == "object":
                value_counts = values.value_counts()
                for sentinel in string_sentinels:
                    if sentinel in value_counts.index:
                        count = int(value_counts[sentinel])
                        if count > 5:
                            findings.append(Finding(
                                id=f"MISS-DEFAULT-{col[:20].upper().replace(' ', '_')}",
                                category="missing_data",
                                severity=Severity.LOW,
                                evidence_score=0.6,
                                title=f"Default/placeholder value in '{col}'",
                                description=(
                                    f"'{sentinel}' appears {count} times, possibly representing missing data."
                                ),
                                column=col,
                                evidence={
                                    "default_value": sentinel,
                                    "count": count,
                                },
                                recommendation="Consider replacing with proper null values.",
                            ))
                            break

        return findings

    # ── Coverage Gaps ────────────────────────────────────────────────

    def _detect_coverage_gaps(self) -> List[Finding]:
        """Detect temporal and categorical coverage gaps."""
        findings = []

        for col in self.data.columns:
            try:
                if self.data[col].dtype == "datetime64[ns]":
                    dates = self.data[col]
                elif self.data[col].dtype == "object":
                    dates = pd.to_datetime(self.data[col], errors="coerce")
                    if dates.isna().mean() > 0.5:
                        continue
                else:
                    continue

                valid = dates.dropna()
                if len(valid) < 10:
                    continue

                years = valid.dt.year.unique()
                if len(years) >= 2:
                    full_range = set(range(int(years.min()), int(years.max()) + 1))
                    missing_years = full_range - set(years)
                    if missing_years:
                        findings.append(Finding(
                            id=f"MISS-YEARSGAP-{col[:20].upper().replace(' ', '_')}",
                            category="missing_data",
                            severity=Severity.LOW,
                            evidence_score=0.7,
                            title=f"Missing year(s) in '{col}'",
                            description=f"{len(missing_years)} year(s) not represented in the data.",
                            column=col,
                            evidence={
                                "missing_years": sorted(list(missing_years)),
                                "year_range": [int(years.min()), int(years.max())],
                            },
                            recommendation="Verify whether missing time periods are expected.",
                        ))
            except Exception:
                continue

        return findings
