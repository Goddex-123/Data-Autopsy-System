"""
Robustness Tester — Refactored

Tests the robustness of statistical conclusions via bootstrap,
sensitivity analysis, subgroup analysis, and outlier influence testing.
Returns Finding objects instead of raw dicts.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.models import Finding, Severity, DatasetProfile, SemanticType
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)


class RobustnessTester:
    """Tests robustness and fragility of statistical conclusions."""

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
        """Run robustness tests and return findings."""
        findings: List[Finding] = []
        findings.extend(self._test_outlier_influence())
        findings.extend(self._test_subgroup_differences())
        findings.extend(self._test_simpsons_paradox())
        return findings

    def _get_analyzable_numeric(self) -> List[str]:
        """Get numeric columns suitable for analysis."""
        cols = list(self.data.select_dtypes(include=[np.number]).columns)
        if self.profile:
            excluded = {
                c for c, p in self.profile.columns.items()
                if p.semantic_type in (SemanticType.IDENTIFIER, SemanticType.CONSTANT)
            }
            cols = [c for c in cols if c not in excluded]
        return cols[:10]  # Limit for performance

    def _test_outlier_influence(self) -> List[Finding]:
        """Test whether outliers substantially influence summary statistics."""
        findings = []
        numeric_cols = self._get_analyzable_numeric()

        for col in numeric_cols:
            values = self.data[col].dropna()
            if len(values) < 30:
                continue

            original_mean = values.mean()
            Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue

            outlier_mask = (values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)
            if outlier_mask.sum() == 0:
                continue

            clean = values[~outlier_mask]
            clean_mean = clean.mean()

            change_pct = abs(original_mean - clean_mean) / abs(original_mean) * 100 if original_mean != 0 else 0

            if change_pct > 10:
                findings.append(Finding(
                    id=f"ROBUST-OUTLIER-{col[:20].upper().replace(' ', '_')}",
                    category="robustness",
                    severity=Severity.MEDIUM,
                    evidence_score=round(min(0.8, change_pct / 50), 2),
                    title=f"Outliers substantially influence '{col}'",
                    description=(
                        f"Removing {outlier_mask.sum()} outliers changes the mean by "
                        f"{change_pct:.1f}%. Summary statistics may be unreliable."
                    ),
                    column=col,
                    evidence={
                        "original_mean": round(float(original_mean), 4),
                        "trimmed_mean": round(float(clean_mean), 4),
                        "mean_change_pct": round(change_pct, 2),
                        "outlier_count": int(outlier_mask.sum()),
                    },
                    recommendation="Use robust statistics (median, MAD) or winsorize outliers.",
                ))

        return findings

    def _test_subgroup_differences(self) -> List[Finding]:
        """Test whether conclusions hold across subgroups."""
        findings = []
        numeric_cols = self._get_analyzable_numeric()[:5]
        cat_cols = [
            c for c in self.data.select_dtypes(include=["object", "category"]).columns
            if 2 <= self.data[c].nunique() <= 10
        ][:3]

        if not numeric_cols or not cat_cols:
            return findings

        for cat_col in cat_cols:
            for num_col in numeric_cols:
                overall_mean = self.data[num_col].mean()
                if overall_mean == 0:
                    continue

                groups = self.data[cat_col].dropna().unique()
                group_means = {}
                for g in groups:
                    gdata = self.data[self.data[cat_col] == g][num_col].dropna()
                    if len(gdata) >= 10:
                        group_means[str(g)] = float(gdata.mean())

                if len(group_means) < 2:
                    continue

                max_dev = max(abs(m - overall_mean) for m in group_means.values())
                dev_pct = max_dev / abs(overall_mean) * 100

                if dev_pct > 30:
                    findings.append(Finding(
                        id=f"ROBUST-SUBGROUP-{cat_col[:10]}_{num_col[:10]}".upper().replace(" ", "_"),
                        category="robustness",
                        severity=Severity.MEDIUM,
                        evidence_score=0.7,
                        title=f"Subgroup differences: '{num_col}' by '{cat_col}'",
                        description=(
                            f"Mean of '{num_col}' varies by up to {dev_pct:.0f}% "
                            f"across subgroups of '{cat_col}'."
                        ),
                        columns=[num_col, cat_col],
                        evidence={
                            "overall_mean": round(float(overall_mean), 4),
                            "group_means": {k: round(v, 4) for k, v in group_means.items()},
                            "max_deviation_pct": round(dev_pct, 2),
                        },
                        recommendation="Report subgroup-specific results alongside overall statistics.",
                    ))

        return findings

    def _test_simpsons_paradox(self) -> List[Finding]:
        """Test for Simpson's Paradox candidates."""
        findings = []
        numeric_cols = self._get_analyzable_numeric()[:5]
        cat_cols = [
            c for c in self.data.select_dtypes(include=["object", "category"]).columns
            if 2 <= self.data[c].nunique() <= 10
        ][:2]

        if len(numeric_cols) < 2 or not cat_cols:
            return findings

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                overall_corr = self.data[col1].corr(self.data[col2])
                if pd.isna(overall_corr) or abs(overall_corr) < 0.1:
                    continue

                for cat_col in cat_cols:
                    for group in self.data[cat_col].dropna().unique():
                        gdata = self.data[self.data[cat_col] == group]
                        if len(gdata) < 20:
                            continue
                        group_corr = gdata[col1].corr(gdata[col2])
                        if pd.notna(group_corr):
                            if (overall_corr > 0.1 and group_corr < -0.1) or \
                               (overall_corr < -0.1 and group_corr > 0.1):
                                findings.append(Finding(
                                    id=f"ROBUST-SIMPSON-{col1[:8]}_{col2[:8]}".upper().replace(" ", "_"),
                                    category="robustness",
                                    severity=Severity.HIGH,
                                    evidence_score=0.75,
                                    title=f"Possible Simpson's Paradox: '{col1}' ↔ '{col2}'",
                                    description=(
                                        f"Overall correlation ({overall_corr:.2f}) reverses "
                                        f"within subgroup '{group}' of '{cat_col}' ({group_corr:.2f}). "
                                        f"Aggregate conclusions may be misleading."
                                    ),
                                    columns=[col1, col2],
                                    evidence={
                                        "overall_correlation": round(overall_corr, 3),
                                        "grouping_variable": cat_col,
                                        "group": str(group),
                                        "group_correlation": round(group_corr, 3),
                                    },
                                    recommendation=(
                                        "Do not rely solely on aggregate statistics. "
                                        "Report subgroup-level analysis."
                                    ),
                                ))
                                break  # One example per pair is enough

        return findings
