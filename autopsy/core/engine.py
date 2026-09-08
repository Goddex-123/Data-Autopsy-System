"""
Central Analysis Engine

Orchestrates all detectors, collects findings, and produces the final report.
"""

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from .models import Finding, DatasetProfile, HealthScore
from .config import AnalysisConfig
from .schema import SchemaAnalyzer
from .scoring import ScoringEngine

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """
    Central analysis engine that orchestrates all detectors and produces
    a unified dataset audit report.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
        source_path: str = "DataFrame",
    ):
        """
        Initialize the analysis engine.

        Args:
            data: The dataset to analyze.
            config: Analysis configuration (uses defaults if None).
            source_path: Path to the original data file.
        """
        self.data = data
        self.config = config or AnalysisConfig()
        self.source_path = source_path
        self.findings: List[Finding] = []
        self.profile: Optional[DatasetProfile] = None
        self.health_score: Optional[HealthScore] = None
        self._timings: Dict[str, float] = {}

    def run_full_analysis(
        self,
        target_column: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run the complete analysis pipeline.

        Args:
            target_column: Optional target column for ML audit.
            reference_data: Optional reference dataset for drift detection.

        Returns:
            Dictionary containing all results.
        """
        logger.info("Starting full analysis pipeline...")
        self.findings = []

        # 1. Schema / Profile
        self._run_timed("schema", self._run_schema_analysis)

        # 2. Data Quality
        self._run_timed("quality", self._run_quality_analysis)

        # 3. Missing Data
        self._run_timed("missing", self._run_missing_analysis)

        # 4. Anomaly Detection
        self._run_timed("anomaly", self._run_anomaly_analysis)

        # 5. Bias Detection
        self._run_timed("bias", self._run_bias_analysis)

        # 6. Privacy Detection
        self._run_timed("privacy", self._run_privacy_analysis)

        # 7. Robustness Testing
        self._run_timed("robustness", self._run_robustness_analysis)

        # 8. ML Audit (if target specified)
        if target_column:
            self._run_timed("ml_audit", lambda: self._run_leakage_analysis(target_column))

        # 9. Drift Detection (if reference provided)
        if reference_data is not None:
            self._run_timed("drift", lambda: self._run_drift_analysis(reference_data))

        # 10. Calculate health score
        self._calculate_health_score()

        logger.info(
            "Analysis complete. %d findings, health score: %.1f",
            len(self.findings),
            self.health_score.overall if self.health_score else 0,
        )

        return self._build_results()

    def _run_timed(self, name: str, func):
        """Run a function and record its execution time."""
        start = time.time()
        try:
            func()
        except Exception as e:
            logger.error("Error in %s analysis: %s", name, e)
            self.findings.append(Finding(
                id=f"ERR-{name.upper()}-001",
                category="error",
                severity="info",
                evidence_score=1.0,
                title=f"Analysis error in {name}",
                description=f"The {name} analysis encountered an error: {str(e)}",
                recommendation="Check data format and try again.",
            ))
        elapsed = time.time() - start
        self._timings[name] = round(elapsed, 3)
        logger.info("  %s analysis completed in %.3fs", name, elapsed)

    def _run_schema_analysis(self):
        """Run schema analysis to build dataset profile."""
        analyzer = SchemaAnalyzer(self.data, self.config.pii_sample_size)
        self.profile = analyzer.analyze()
        logger.info(
            "Schema analysis: %d rows × %d cols, %.1f%% missing",
            self.profile.row_count,
            self.profile.column_count,
            self.profile.missing_percentage,
        )

    def _run_quality_analysis(self):
        """Run data quality checks."""
        from ..detectors.quality_detector import QualityDetector
        detector = QualityDetector(self.data, self.config, self.profile)
        self.findings.extend(detector.analyze())

    def _run_missing_analysis(self):
        """Run missing data analysis."""
        from ..detectors.missing_analyzer import MissingDataAnalyzer
        analyzer = MissingDataAnalyzer(self.data, self.config)
        self.findings.extend(analyzer.analyze())

    def _run_anomaly_analysis(self):
        """Run anomaly detection."""
        from ..detectors.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(self.data, self.config, self.profile)
        self.findings.extend(detector.analyze())

    def _run_bias_analysis(self):
        """Run bias detection."""
        from ..detectors.bias_detector import BiasDetector
        detector = BiasDetector(self.data, self.config)
        self.findings.extend(detector.analyze())

    def _run_privacy_analysis(self):
        """Run privacy/PII detection."""
        from ..detectors.privacy_detector import PrivacyDetector
        detector = PrivacyDetector(self.data, self.config, self.profile)
        self.findings.extend(detector.analyze())

    def _run_robustness_analysis(self):
        """Run robustness analysis."""
        from ..detectors.robustness_tester import RobustnessTester
        tester = RobustnessTester(self.data, self.config, self.profile)
        self.findings.extend(tester.analyze())

    def _run_leakage_analysis(self, target_column: str):
        """Run target leakage detection."""
        from ..detectors.leakage_detector import LeakageDetector
        detector = LeakageDetector(self.data, self.config, target_column)
        self.findings.extend(detector.analyze())

    def _run_drift_analysis(self, reference_data: pd.DataFrame):
        """Run data drift detection."""
        from ..detectors.drift_detector import DriftDetector
        detector = DriftDetector(reference_data, self.data, self.config)
        self.findings.extend(detector.analyze())

    def _calculate_health_score(self):
        """Calculate the overall health score from all findings."""
        scorer = ScoringEngine(self.config.score_weights)

        completeness = 100.0 - (self.profile.missing_percentage if self.profile else 0)

        # Count validity issues
        validity_issues = sum(
            1 for f in self.findings
            if f.category == "data_quality" and "invalid" in f.id.lower()
        )
        total_cols = self.profile.column_count if self.profile else 1
        validity_pct = max(0, 100 - (validity_issues / max(total_cols, 1) * 100))

        # Consistency
        consistency_issues = sum(
            1 for f in self.findings
            if f.category == "data_quality" and "consistency" in f.id.lower()
        )
        consistency_pct = max(0, 100 - (consistency_issues / max(total_cols, 1) * 100))

        self.health_score = scorer.calculate_health_score(
            self.findings,
            profile=self.profile,
            completeness_pct=completeness,
            validity_pct=validity_pct,
            consistency_pct=consistency_pct,
        )

    def _build_results(self) -> Dict:
        """Build the final results dictionary."""
        # Sort findings by severity (most severe first)
        sorted_findings = sorted(
            self.findings,
            key=lambda f: (
                f.severity.numeric if isinstance(f.severity, type(f.severity)) and hasattr(f.severity, 'numeric')
                else 0
            ),
            reverse=True,
        )

        severity_counts = ScoringEngine.count_by_severity(self.findings)

        return {
            "profile": self.profile.to_dict() if self.profile else {},
            "health_score": self.health_score.to_dict() if self.health_score else {},
            "findings": [f.to_dict() for f in sorted_findings],
            "findings_count": len(self.findings),
            "severity_counts": severity_counts,
            "timings": self._timings,
            "source_path": self.source_path,
        }

    def get_findings_by_category(self, category: str) -> List[Finding]:
        """Get findings filtered by category."""
        return [f for f in self.findings if f.category == category]

    def get_findings_by_severity(self, *severities) -> List[Finding]:
        """Get findings filtered by severity level(s)."""
        return [f for f in self.findings if f.severity in severities]
