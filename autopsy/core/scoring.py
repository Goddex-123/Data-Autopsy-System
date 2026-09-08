"""
Evidence-Based Scoring Engine

Replaces the ad-hoc additive scoring with normalized, documented component scores.
Each component score is 0-100, and the overall health score is a weighted average.
"""

import logging
from typing import Dict, List, Optional

from .models import Finding, Severity, HealthScore, ScoreComponent

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Calculates dataset health scores from findings.

    Each score component (completeness, validity, etc.) is independently
    scored 0-100, then combined via configurable weights into an overall
    health score.
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Initialize scoring engine.

        Args:
            weights: Component name -> weight mapping. Must sum to ~1.0.
        """
        self.weights = weights

    def calculate_health_score(
        self,
        findings: List[Finding],
        profile: Optional['DatasetProfile'] = None,
        completeness_pct: float = 100.0,
        validity_pct: float = 100.0,
        consistency_pct: float = 100.0,
    ) -> HealthScore:
        """
        Calculate overall health score from all findings.

        Args:
            findings: All findings from all detectors.
            profile: The complete dataset profile.
            completeness_pct: Percentage of non-null cells (0-100).
            validity_pct: Percentage of valid values (0-100).
            consistency_pct: Percentage of consistent values (0-100).

        Returns:
            HealthScore with component breakdown.
        """
        components: Dict[str, ScoreComponent] = {}
        total_rows = profile.row_count if profile else 1000
        total_cols = profile.column_count if profile else 10

        # Completeness: directly from missing data percentage
        components["completeness"] = ScoreComponent(
            name="Completeness",
            score=min(100, max(0, completeness_pct)),
            weight=self.weights.get("completeness", 0.20),
            details=f"{completeness_pct:.1f}% of cells contain data",
            findings_count=self._count_by_category(findings, "missing_data"),
        )

        # Validity: from quality findings
        components["validity"] = ScoreComponent(
            name="Validity",
            score=min(100, max(0, validity_pct)),
            weight=self.weights.get("validity", 0.15),
            details=f"{validity_pct:.1f}% of values pass validity checks",
            findings_count=self._count_by_category(findings, "data_quality"),
        )

        # Consistency
        components["consistency"] = ScoreComponent(
            name="Consistency",
            score=min(100, max(0, consistency_pct)),
            weight=self.weights.get("consistency", 0.15),
            details=f"{consistency_pct:.1f}% consistency score",
            findings_count=0,
        )

        # Anomaly Risk: penalty-based from anomaly findings
        anomaly_findings = [f for f in findings if f.category == "anomaly"]
        anomaly_score = self._penalty_score(anomaly_findings, total_rows, total_cols)
        components["anomaly_risk"] = ScoreComponent(
            name="Anomaly Risk",
            score=anomaly_score,
            weight=self.weights.get("anomaly_risk", 0.15),
            details=f"{len(anomaly_findings)} anomaly finding(s)",
            findings_count=len(anomaly_findings),
        )

        # Bias Risk
        bias_findings = [f for f in findings if f.category == "bias"]
        bias_score = self._penalty_score(bias_findings, total_rows, total_cols)
        components["bias_risk"] = ScoreComponent(
            name="Bias Risk",
            score=bias_score,
            weight=self.weights.get("bias_risk", 0.10),
            details=f"{len(bias_findings)} bias finding(s)",
            findings_count=len(bias_findings),
        )

        # Privacy Risk
        privacy_findings = [f for f in findings if f.category == "privacy"]
        privacy_score = self._penalty_score(privacy_findings, total_rows, total_cols)
        components["privacy_risk"] = ScoreComponent(
            name="Privacy Risk",
            score=privacy_score,
            weight=self.weights.get("privacy_risk", 0.10),
            details=f"{len(privacy_findings)} privacy finding(s)",
            findings_count=len(privacy_findings),
        )

        # Robustness
        robustness_findings = [f for f in findings if f.category == "robustness"]
        robustness_score = self._penalty_score(robustness_findings, total_rows, total_cols)
        components["robustness"] = ScoreComponent(
            name="Robustness",
            score=robustness_score,
            weight=self.weights.get("robustness", 0.15),
            details=f"{len(robustness_findings)} robustness finding(s)",
            findings_count=len(robustness_findings),
        )

        # Calculate weighted overall
        overall = sum(
            comp.weighted_score()
            for comp in components.values()
        )
        overall = min(100.0, max(0.0, overall))

        # Determine verdict
        if overall >= 80:
            verdict = "HEALTHY"
            emoji = "✅"
        elif overall >= 60:
            verdict = "MODERATE CONCERNS"
            emoji = "⚠️"
        elif overall >= 40:
            verdict = "SIGNIFICANT ISSUES"
            emoji = "🟠"
        else:
            verdict = "CRITICAL ISSUES"
            emoji = "🔴"

        return HealthScore(
            overall=round(overall, 1),
            components=components,
            verdict=verdict,
            verdict_emoji=emoji,
        )

    def _penalty_score(self, findings: List[Finding], total_rows: int, total_cols: int) -> float:
        """
        Calculate a score where 100 = no issues.
        Penalties are proportional to the magnitude of the problem (affected rows/columns).
        """
        score = 100.0

        # Base multipliers by severity
        severity_multiplier = {
            Severity.CRITICAL: 1.0,
            Severity.HIGH: 0.8,
            Severity.MEDIUM: 0.5,
            Severity.LOW: 0.2,
            Severity.INFO: 0.05,
        }

        for finding in findings:
            severity = finding.severity if isinstance(finding.severity, Severity) else Severity(finding.severity)
            multiplier = severity_multiplier.get(severity, 0.5)
            
            # Determine affected proportion
            proportion = 0.01  # default small penalty
            
            evidence = finding.evidence or {}
            
            # If finding affects specific counts of rows/cells
            affected_count = evidence.get("affected_rows", evidence.get("affected_cells", evidence.get("outlier_count", evidence.get("anomaly_count"))))
            if affected_count is not None and total_rows > 0:
                proportion = float(affected_count) / float(total_rows)
            # If finding is column-level (e.g. bias, drift)
            elif finding.column or finding.columns:
                num_cols = len(finding.columns) if finding.columns else 1
                proportion = float(num_cols) / float(total_cols) if total_cols > 0 else 0.1

            # Prevent proportion from being wildly over 1.0
            proportion = min(1.0, max(0.01, proportion))
            
            # Max penalty per finding is 30 points (CRITICAL) scaled by proportion and evidence
            max_points = 30.0
            penalty = max_points * multiplier * proportion * finding.evidence_score
            
            score -= penalty

        return max(0.0, min(100.0, score))

    @staticmethod
    def _count_by_category(findings: List[Finding], category: str) -> int:
        """Count findings in a specific category."""
        return sum(1 for f in findings if f.category == category)

    @staticmethod
    def count_by_severity(findings: List[Finding]) -> Dict[str, int]:
        """Count findings grouped by severity."""
        counts = {s.value: 0 for s in Severity}
        for f in findings:
            sev = f.severity.value if isinstance(f.severity, Severity) else f.severity
            counts[sev] = counts.get(sev, 0) + 1
        return counts
