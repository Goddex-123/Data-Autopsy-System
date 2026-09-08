"""
Target Leakage Detector

ML-focused analysis:
- Target leakage detection via correlation
- Feature-target relationship analysis
- Class imbalance for classification targets
- Feature redundancy detection
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.models import Finding, Severity
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)


class LeakageDetector:
    """
    Detects potential target leakage and analyzes feature-target relationships
    for ML dataset auditing.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
        target_column: Optional[str] = None,
    ):
        self.data = data
        self.config = config or AnalysisConfig()
        self.target_column = target_column

    def analyze(self) -> List[Finding]:
        """Run ML audit analysis."""
        findings: List[Finding] = []

        if not self.target_column:
            findings.extend(self._suggest_potential_targets())
            return findings

        if self.target_column not in self.data.columns:
            findings.append(Finding(
                id="ML-TARGET-MISSING",
                category="ml_audit",
                severity=Severity.HIGH,
                evidence_score=1.0,
                title="Target column not found",
                description=f"The specified target column '{self.target_column}' is not in the dataset.",
                recommendation="Verify the target column name.",
            ))
            return findings

        target = self.data[self.target_column].dropna()
        if len(target) < 10 or target.nunique() < 2:
            findings.append(Finding(
                id="ML-TARGET-INVALID",
                category="ml_audit",
                severity=Severity.HIGH,
                evidence_score=1.0,
                title="Invalid target column",
                description=f"Target column '{self.target_column}' has insufficient data or < 2 unique values.",
                column=self.target_column,
                recommendation="A target column must have at least 2 distinct values and sufficient non-null data for ML.",
            ))
            return findings

        findings.extend(self._detect_target_leakage())
        findings.extend(self._analyze_feature_relationships())
        findings.extend(self._detect_class_imbalance())
        findings.extend(self._detect_feature_redundancy())
        return findings

    def _suggest_potential_targets(self) -> List[Finding]:
        """Suggest columns that might be ML targets based on names."""
        suggestions = []
        target_keywords = ["target", "label", "class", "outcome", "status", "result"]
        
        for col in self.data.columns:
            if any(kw in col.lower() for kw in target_keywords):
                suggestions.append(col)
                
        if suggestions:
            return [Finding(
                id="ML-TARGET-SUGGEST",
                category="ml_audit",
                severity=Severity.INFO,
                evidence_score=0.8,
                title="Potential ML targets found",
                description=f"No target column specified, but found potential targets: {', '.join(suggestions[:5])}.",
                evidence={"suggested_targets": suggestions},
                recommendation="Specify a target column to enable Target Leakage and ML Feature analyses.",
            )]
        return []

    def _detect_target_leakage(self) -> List[Finding]:
        """Detect features suspiciously correlated with the target."""
        findings = []
        target = self.data[self.target_column]

        if not pd.api.types.is_numeric_dtype(target):
            return findings

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_column]

        for col in numeric_cols:
            try:
                valid = self.data[[col, self.target_column]].dropna()
                if len(valid) < 10:
                    continue

                corr = valid[col].corr(valid[self.target_column])
                if pd.isna(corr):
                    continue

                if abs(corr) >= self.config.leakage_correlation_threshold:
                    findings.append(Finding(
                        id=f"ML-LEAK-{col[:20].upper().replace(' ', '_')}",
                        category="ml_audit",
                        severity=Severity.CRITICAL,
                        evidence_score=round(abs(corr), 2),
                        title=f"Potential target leakage: '{col}'",
                        description=(
                            f"Feature '{col}' has {abs(corr):.3f} correlation with target "
                            f"'{self.target_column}'. This may contain post-outcome information."
                        ),
                        column=col,
                        evidence={
                            "pearson_correlation": round(corr, 3),
                            "target_column": self.target_column,
                            "sample_size": len(valid),
                        },
                        statistical_test="Pearson correlation",
                        recommendation=(
                            "Verify whether this feature is available at prediction time. "
                            "If it's derived from the target, exclude it before training."
                        ),
                        limitations=(
                            "High correlation alone does not prove leakage. "
                            "Domain knowledge is required to determine causality."
                        ),
                    ))
            except Exception:
                continue

        return findings

    def _analyze_feature_relationships(self) -> List[Finding]:
        """Analyze feature-target correlations and report top features."""
        target = self.data[self.target_column]
        if not pd.api.types.is_numeric_dtype(target):
            return []

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_column]

        correlations = []
        for col in numeric_cols:
            try:
                valid = self.data[[col, self.target_column]].dropna()
                if len(valid) < 10:
                    continue

                pearson = valid[col].corr(valid[self.target_column])
                spearman = valid[col].corr(valid[self.target_column], method="spearman")

                if pd.notna(pearson):
                    correlations.append({
                        "feature": col,
                        "pearson": round(pearson, 3),
                        "spearman": round(spearman, 3) if pd.notna(spearman) else None,
                        "abs_pearson": abs(pearson),
                    })
            except Exception:
                continue

        if correlations:
            correlations.sort(key=lambda x: x["abs_pearson"], reverse=True)

            return [Finding(
                id="ML-FEATURES-001",
                category="ml_audit",
                severity=Severity.INFO,
                evidence_score=0.8,
                title=f"Feature-target relationship summary",
                description=(
                    f"Top correlated features with '{self.target_column}' identified."
                ),
                evidence={
                    "target": self.target_column,
                    "top_features": correlations[:10],
                    "total_numeric_features": len(numeric_cols),
                },
                recommendation="Review feature importance for model selection.",
            )]

        return []

    def _detect_class_imbalance(self) -> List[Finding]:
        """Detect class imbalance in the target variable."""
        target = self.data[self.target_column].dropna()
        findings = []

        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
            return findings  # Regression target, skip

        value_counts = target.value_counts()
        if len(value_counts) < 2:
            return findings

        majority = value_counts.iloc[0]
        minority = value_counts.iloc[-1]
        ratio = majority / minority if minority > 0 else float("inf")

        if ratio > 3:
            severity = Severity.HIGH if ratio > 10 else Severity.MEDIUM
            findings.append(Finding(
                id="ML-IMBALANCE-001",
                category="ml_audit",
                severity=severity,
                evidence_score=0.9,
                title=f"Target class imbalance ({ratio:.1f}:1)",
                description=(
                    f"Target '{self.target_column}' has an imbalance ratio of {ratio:.1f}:1. "
                    f"Majority class: '{value_counts.index[0]}' ({majority} samples), "
                    f"Minority class: '{value_counts.index[-1]}' ({minority} samples)."
                ),
                column=self.target_column,
                evidence={
                    "imbalance_ratio": round(ratio, 2),
                    "class_distribution": {
                        str(k): int(v) for k, v in value_counts.items()
                    },
                },
                recommendation=(
                    "Consider using stratified splits, class weights, "
                    "oversampling (SMOTE), or evaluation metrics robust to imbalance (F1, PR-AUC)."
                ),
            ))

        return findings

    def _detect_feature_redundancy(self) -> List[Finding]:
        """Detect highly correlated feature pairs (redundancy)."""
        findings = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_column]

        if len(numeric_cols) < 2:
            return findings

        corr_matrix = self.data[numeric_cols].corr()
        redundant_pairs = []

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr = corr_matrix.loc[col1, col2]
                if pd.notna(corr) and abs(corr) >= self.config.feature_redundancy_threshold:
                    redundant_pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(corr, 3),
                    })

        if redundant_pairs:
            findings.append(Finding(
                id="ML-REDUNDANCY-001",
                category="ml_audit",
                severity=Severity.MEDIUM,
                evidence_score=0.8,
                title=f"{len(redundant_pairs)} redundant feature pair(s) detected",
                description=(
                    f"Features with >{self.config.feature_redundancy_threshold:.0%} "
                    f"correlation may contain duplicate information."
                ),
                evidence={"redundant_pairs": redundant_pairs[:10]},
                recommendation=(
                    "Consider removing one feature from each redundant pair "
                    "to reduce multicollinearity."
                ),
            ))

        return findings
