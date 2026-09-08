"""
Data Drift Detector

Compares a reference dataset against a current dataset using:
- KS test (numerical)
- PSI (Population Stability Index)
- Chi-square (categorical)
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.models import Finding, Severity
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects statistical drift between reference and current datasets.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
    ):
        self.reference = reference
        self.current = current
        self.config = config or AnalysisConfig()

    def analyze(self) -> List[Finding]:
        """Run drift analysis on all shared columns."""
        findings: List[Finding] = []

        shared_cols = list(set(self.reference.columns) & set(self.current.columns))
        if not shared_cols:
            findings.append(Finding(
                id="DRIFT-NOCOLS-001",
                category="drift",
                severity=Severity.HIGH,
                evidence_score=0.95,
                title="No shared columns between reference and current datasets",
                description="Cannot perform drift analysis without shared columns.",
                evidence={
                    "reference_columns": list(self.reference.columns),
                    "current_columns": list(self.current.columns),
                },
                recommendation="Ensure datasets share the same schema.",
            ))
            return findings

        for col in shared_cols:
            if pd.api.types.is_numeric_dtype(self.reference[col]) and pd.api.types.is_numeric_dtype(self.current[col]):
                finding = self._numerical_drift(col)
            elif self.reference[col].dtype == "object" or self.current[col].dtype == "object":
                finding = self._categorical_drift(col)
            else:
                continue

            if finding:
                findings.append(finding)

        return findings

    def _numerical_drift(self, col: str) -> Optional[Finding]:
        """Detect drift in a numerical column using KS test and PSI."""
        ref_vals = self.reference[col].dropna()
        cur_vals = self.current[col].dropna()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            return None

        # KS test
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)
            wasserstein_dist = stats.wasserstein_distance(ref_vals, cur_vals)
        except Exception:
            return None

        # PSI calculation
        psi = self._calculate_psi(ref_vals, cur_vals)

        # Determine severity
        if psi > self.config.drift_high_threshold or ks_pvalue < 0.001:
            severity = Severity.HIGH
            drift_label = "HIGH"
        elif psi > self.config.drift_low_threshold or ks_pvalue < 0.05:
            severity = Severity.MEDIUM
            drift_label = "MEDIUM"
        else:
            return None  # No significant drift

        return Finding(
            id=f"DRIFT-NUM-{col[:20].upper().replace(' ', '_')}",
            category="drift",
            severity=severity,
            evidence_score=round(min(0.9, 1 - ks_pvalue), 2),
            title=f"Numerical drift detected in '{col}'",
            description=(
                f"Distribution of '{col}' has shifted between reference and current. "
                f"Drift level: {drift_label}."
            ),
            column=col,
            evidence={
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pvalue, 6),
                "psi": round(psi, 4),
                "wasserstein_distance": round(wasserstein_dist, 4),
                "magnitude": drift_label,
                "ref_mean": round(float(ref_vals.mean()), 4),
                "cur_mean": round(float(cur_vals.mean()), 4),
                "ref_std": round(float(ref_vals.std()), 4),
                "cur_std": round(float(cur_vals.std()), 4),
                "ref_count": len(ref_vals),
                "cur_count": len(cur_vals),
            },
            statistical_test="KS test + PSI + Wasserstein distance",
            recommendation=(
                "Investigate whether the distribution shift reflects a real change "
                "or a data collection issue. Retrain models if drift is significant."
            ),
            limitations=(
                "Statistical drift does not necessarily imply practical significance. "
                "Small p-values can occur with large sample sizes even for trivial differences."
            ),
        )

    def _categorical_drift(self, col: str) -> Optional[Finding]:
        """Detect drift in a categorical column using chi-square."""
        ref_vals = self.reference[col].dropna()
        cur_vals = self.current[col].dropna()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            return None

        all_cats = sorted(set(ref_vals.unique()) | set(cur_vals.unique()), key=str)
        if len(all_cats) < 2 or len(all_cats) > 100:
            return None

        ref_counts = ref_vals.value_counts()
        cur_counts = cur_vals.value_counts()

        obs = [int(cur_counts.get(c, 0)) for c in all_cats]
        exp_proportions = [ref_counts.get(c, 0) / len(ref_vals) for c in all_cats]
        exp = [p * len(cur_vals) for p in exp_proportions]

        # Avoid zero expected
        exp = [max(e, 0.001) for e in exp]

        try:
            chi2, p_value = stats.chisquare(obs, exp)
        except ValueError:
            return None

        if p_value >= 0.05:
            return None

        severity = Severity.HIGH if p_value < 0.001 else Severity.MEDIUM
        magnitude = "HIGH" if p_value < 0.001 else "MEDIUM"

        return Finding(
            id=f"DRIFT-CAT-{col[:20].upper().replace(' ', '_')}",
            category="drift",
            severity=severity,
            evidence_score=round(min(0.9, 1 - p_value), 2),
            title=f"Categorical drift detected in '{col}'",
            description=(
                f"Category distribution of '{col}' has significantly changed "
                f"between reference and current datasets (p={p_value:.4f}). "
                f"Magnitude: {magnitude}."
            ),
            column=col,
            evidence={
                "chi2_statistic": round(chi2, 2),
                "p_value": round(p_value, 6),
                "magnitude": magnitude,
                "n_categories": len(all_cats),
                "ref_count": len(ref_vals),
                "cur_count": len(cur_vals),
            },
            statistical_test="Chi-square goodness-of-fit",
            recommendation="Review whether category proportions have shifted meaningfully.",
        )

    @staticmethod
    def _calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)

            if len(breakpoints) < 3:
                return 0.0

            ref_hist = np.histogram(reference, bins=breakpoints)[0]
            cur_hist = np.histogram(current, bins=breakpoints)[0]

            ref_pct = ref_hist / ref_hist.sum()
            cur_pct = cur_hist / cur_hist.sum()

            # Avoid log(0)
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return float(psi)
        except Exception:
            return 0.0
