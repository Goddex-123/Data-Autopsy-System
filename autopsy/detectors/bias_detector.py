"""
Bias Detector — Refactored

Population-aware bias detection that:
- Does NOT assume binary gender = bias
- Supports user-provided expected distributions
- Uses statistical tests (chi-square) for representation comparison
- Distinguishes imbalance from bias
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.models import Finding, Severity
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Detects representation imbalances and potential sampling concerns.

    IMPORTANT: This detector identifies statistical imbalances,
    NOT moral or societal bias. Whether an imbalance constitutes
    "bias" depends on the context and intended use of the data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
    ):
        self.data = data
        self.config = config or AnalysisConfig()

    def analyze(self) -> List[Finding]:
        """Run all bias detection methods."""
        findings: List[Finding] = []
        findings.extend(self._detect_class_imbalance())
        findings.extend(self._detect_distribution_skew())
        findings.extend(self._detect_population_mismatch())
        findings.extend(self._detect_correlation_concerns())
        findings.extend(self._detect_temporal_coverage())
        findings.extend(self._detect_sample_size_concerns())
        return findings

    # ── Class Imbalance ──────────────────────────────────────────────

    def _detect_class_imbalance(self) -> List[Finding]:
        """Detect class imbalance in categorical columns."""
        findings = []
        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            value_counts = self.data[col].value_counts(dropna=True)
            total = value_counts.sum()
            if total == 0 or len(value_counts) < 2:
                continue

            max_prop = value_counts.iloc[0] / total
            min_prop = value_counts.iloc[-1] / total
            imbalance_ratio = max_prop / min_prop if min_prop > 0 else float("inf")

            if imbalance_ratio > self.config.imbalance_ratio_warning:
                severity = Severity.HIGH if imbalance_ratio > 50 else Severity.MEDIUM
                findings.append(Finding(
                    id=f"BIAS-IMBAL-{col[:20].upper().replace(' ', '_')}",
                    category="bias",
                    severity=severity,
                    evidence_score=round(min(0.85, 0.5 + imbalance_ratio / 200), 2),
                    title=f"Class imbalance in '{col}'",
                    description=(
                        f"The most frequent value '{value_counts.index[0]}' appears "
                        f"{max_prop*100:.1f}% of the time. Imbalance ratio: {imbalance_ratio:.1f}:1."
                    ),
                    column=col,
                    evidence={
                        "dominant_value": str(value_counts.index[0]),
                        "dominant_proportion": round(max_prop, 3),
                        "minority_value": str(value_counts.index[-1]),
                        "minority_proportion": round(min_prop, 3),
                        "imbalance_ratio": round(imbalance_ratio, 2),
                        "n_categories": len(value_counts),
                        "distribution": {
                            str(k): int(v) for k, v in value_counts.head(10).items()
                        },
                    },
                    recommendation=(
                        "Determine whether this imbalance reflects the population or "
                        "indicates sampling issues. For ML tasks, consider stratified "
                        "sampling or class-weight adjustments."
                    ),
                    limitations=(
                        "Imbalance alone does not mean the data is biased. "
                        "Many real-world distributions are naturally imbalanced."
                    ),
                ))

            # Dominant category (>80%)
            if max_prop > 0.8 and imbalance_ratio <= self.config.imbalance_ratio_warning:
                findings.append(Finding(
                    id=f"BIAS-DOMINANT-{col[:20].upper().replace(' ', '_')}",
                    category="bias",
                    severity=Severity.LOW,
                    evidence_score=0.7,
                    title=f"Dominant category in '{col}'",
                    description=(
                        f"'{value_counts.index[0]}' represents {max_prop*100:.1f}% of values."
                    ),
                    column=col,
                    evidence={
                        "dominant_value": str(value_counts.index[0]),
                        "proportion": round(max_prop, 3),
                    },
                    recommendation="Verify whether this distribution is expected for the population.",
                ))

        return findings

    # ── Distribution Skew ────────────────────────────────────────────

    def _detect_distribution_skew(self) -> List[Finding]:
        """Detect highly skewed numeric distributions."""
        findings = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = self.data[col].dropna()
            if len(values) < 20:
                continue

            skewness = float(stats.skew(values))
            kurtosis = float(stats.kurtosis(values))

            if abs(skewness) > 2.0:
                findings.append(Finding(
                    id=f"BIAS-SKEW-{col[:20].upper().replace(' ', '_')}",
                    category="bias",
                    severity=Severity.LOW,
                    evidence_score=round(min(0.7, abs(skewness) / 10), 2),
                    title=f"Highly skewed distribution in '{col}'",
                    description=(
                        f"Skewness: {skewness:.2f} "
                        f"({'right-skewed' if skewness > 0 else 'left-skewed'}). "
                        f"Kurtosis: {kurtosis:.2f}."
                    ),
                    column=col,
                    evidence={
                        "skewness": round(skewness, 3),
                        "kurtosis": round(kurtosis, 3),
                        "direction": "right" if skewness > 0 else "left",
                    },
                    recommendation=(
                        "Highly skewed distributions may affect mean-based statistics. "
                        "Consider using robust measures (median, MAD) or transformations."
                    ),
                    limitations="Skewness is common in many natural distributions (e.g., income, prices).",
                ))

        return findings

    # ── Population Mismatch ──────────────────────────────────────────

    def _detect_population_mismatch(self) -> List[Finding]:
        """
        Compare observed distributions against expected population distributions
        when the user provides them.
        """
        findings = []
        expected = self.config.expected_distributions
        if not expected:
            return findings

        for col, expected_dist in expected.items():
            if col not in self.data.columns:
                continue

            observed_counts = self.data[col].value_counts(dropna=True)
            total = observed_counts.sum()
            if total == 0:
                continue

            # Build observed and expected arrays for chi-square
            all_categories = set(observed_counts.index) | set(expected_dist.keys())
            obs_vals = []
            exp_vals = []
            comparison = {}

            for cat in all_categories:
                obs_count = int(observed_counts.get(cat, 0))
                exp_proportion = expected_dist.get(str(cat), 0)
                exp_count = exp_proportion * total

                obs_vals.append(obs_count)
                exp_vals.append(max(exp_count, 0.001))  # Avoid zero

                comparison[str(cat)] = {
                    "observed_pct": round(obs_count / total * 100, 1),
                    "expected_pct": round(exp_proportion * 100, 1),
                    "difference_pct": round((obs_count / total - exp_proportion) * 100, 1),
                }

            if len(obs_vals) < 2:
                continue

            try:
                chi2, p_value = stats.chisquare(obs_vals, exp_vals)
            except ValueError:
                continue

            if p_value < 0.05:
                severity = Severity.HIGH if p_value < 0.001 else Severity.MEDIUM
                findings.append(Finding(
                    id=f"BIAS-POP-{col[:20].upper().replace(' ', '_')}",
                    category="bias",
                    severity=severity,
                    evidence_score=round(min(0.9, 1 - p_value), 2),
                    title=f"Population mismatch in '{col}'",
                    description=(
                        f"The distribution of '{col}' significantly differs from "
                        f"the expected population distribution (p={p_value:.4f})."
                    ),
                    column=col,
                    evidence={
                        "chi2_statistic": round(chi2, 2),
                        "p_value": round(p_value, 6),
                        "comparison": comparison,
                    },
                    statistical_test="Chi-square goodness-of-fit vs expected population",
                    recommendation=(
                        "Investigate whether this mismatch reflects sampling bias "
                        "or is expected for the specific data collection context."
                    ),
                ))

        return findings

    # ── Correlation Concerns ─────────────────────────────────────────

    def _detect_correlation_concerns(self) -> List[Finding]:
        """Detect suspiciously strong or uniform correlations."""
        findings = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return findings

        corr_matrix = self.data[numeric_cols].corr()

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr = corr_matrix.loc[col1, col2]
                if pd.isna(corr):
                    continue

                if abs(corr) > 0.95:
                    findings.append(Finding(
                        id=f"BIAS-CORR-{col1[:10]}_{col2[:10]}".upper().replace(" ", "_"),
                        category="bias",
                        severity=Severity.MEDIUM,
                        evidence_score=round(abs(corr), 2),
                        title=f"Near-perfect correlation: '{col1}' ↔ '{col2}'",
                        description=(
                            f"Pearson correlation of {corr:.3f} suggests these columns "
                            f"may contain redundant or derived information."
                        ),
                        columns=[col1, col2],
                        evidence={"correlation": round(corr, 3)},
                        recommendation=(
                            "Verify whether one column is derived from the other. "
                            "Consider removing redundant features before modeling."
                        ),
                    ))

        return findings

    # ── Temporal Coverage ────────────────────────────────────────────

    def _detect_temporal_coverage(self) -> List[Finding]:
        """Check for time-related coverage issues."""
        findings = []

        for col in self.data.columns:
            try:
                if self.data[col].dtype == "datetime64[ns]":
                    dates = self.data[col]
                elif self.data[col].dtype == "object":
                    dates = pd.to_datetime(self.data[col], errors="coerce")
                else:
                    continue

                valid_dates = dates.dropna()
                if len(valid_dates) < 10:
                    continue

                months = valid_dates.dt.month.value_counts()
                missing_months = set(range(1, 13)) - set(months.index)

                if missing_months and len(missing_months) < 12:
                    findings.append(Finding(
                        id=f"BIAS-TEMPORAL-{col[:20].upper().replace(' ', '_')}",
                        category="bias",
                        severity=Severity.LOW,
                        evidence_score=0.6,
                        title=f"Incomplete temporal coverage in '{col}'",
                        description=(
                            f"{len(missing_months)} month(s) not represented in the data."
                        ),
                        column=col,
                        evidence={
                            "missing_months": sorted(list(missing_months)),
                            "months_present": sorted(list(months.index)),
                        },
                        recommendation="Verify whether missing time periods are expected.",
                    ))
            except Exception:
                continue

        return findings

    # ── Sample Size ──────────────────────────────────────────────────

    def _detect_sample_size_concerns(self) -> List[Finding]:
        """Flag potential sample size issues."""
        findings = []
        n = len(self.data)

        if n < 30:
            findings.append(Finding(
                id="BIAS-SIZE-001",
                category="bias",
                severity=Severity.HIGH,
                evidence_score=0.9,
                title="Very small sample size",
                description=f"Dataset contains only {n} rows. Statistical analyses may be unreliable.",
                evidence={"row_count": n},
                recommendation="Collect more data if possible, or use appropriate small-sample methods.",
            ))
        elif n < 100:
            findings.append(Finding(
                id="BIAS-SIZE-001",
                category="bias",
                severity=Severity.MEDIUM,
                evidence_score=0.7,
                title="Small sample size",
                description=f"Dataset contains {n} rows. Some analyses have limited statistical power.",
                evidence={"row_count": n},
                recommendation="Be cautious with conclusions that require larger samples.",
            ))

        return findings
