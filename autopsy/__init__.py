"""
Data Autopsy System — Public API

Professional data quality, statistical forensics, and ML dataset auditing.

Usage:
    from autopsy import DataAutopsy

    autopsy = DataAutopsy(df)
    results = autopsy.investigate()
    autopsy.report.save("report.html")
"""

__version__ = "2.0.0"

import logging
from typing import Dict, Optional

import pandas as pd

from .core.config import AnalysisConfig
from .core.engine import AnalysisEngine
from .core.models import Finding, Severity, SemanticType, HealthScore, DatasetProfile
from .core.schema import SchemaAnalyzer
from .core.scoring import ScoringEngine
from .provenance.fingerprint import DatasetFingerprint
from .reporting.generator import ReportGenerator
from .visualization.charts import ForensicCharts

logger = logging.getLogger(__name__)


class DataAutopsy:
    """
    Main entry point for the Data Autopsy System.

    Orchestrates dataset analysis across data quality, missing data,
    anomaly detection, bias, privacy, ML audit, drift, and robustness.
    """

    def __init__(
        self,
        data,
        config: Optional[AnalysisConfig] = None,
        source_path: str = "DataFrame",
    ):
        """
        Initialize Data Autopsy.

        Args:
            data: Either a pandas DataFrame or a file path (CSV).
            config: Optional analysis configuration.
            source_path: Label for the data source.
        """
        if isinstance(data, str):
            self.source_path = data
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
            self.source_path = source_path
        else:
            raise TypeError(f"Expected DataFrame or file path, got {type(data)}")

        self.config = config or AnalysisConfig()
        self.engine = AnalysisEngine(self.data, self.config, self.source_path)
        self.report: Optional[ReportGenerator] = None
        self.findings = {}  # Legacy compatibility
        self._results: Optional[Dict] = None

    def quick_scan(self) -> dict:
        """
        Run a quick scan returning basic dataset statistics.

        Returns:
            Dictionary with basic profile information.
        """
        schema = SchemaAnalyzer(self.data)
        profile = schema.analyze()

        return {
            "rows": profile.row_count,
            "columns": profile.column_count,
            "total_cells": profile.total_cells,
            "missing_cells": profile.missing_cells,
            "missing_percentage": round(profile.missing_percentage, 2),
            "duplicate_rows": profile.duplicate_rows,
            "duplicate_percentage": round(profile.duplicate_percentage, 2),
            "memory_usage_mb": profile.memory_usage_mb,
            "numeric_columns": len(self.data.select_dtypes(include=["number"]).columns),
            "categorical_columns": len(self.data.select_dtypes(include=["object", "category"]).columns),
        }

    def investigate(
        self,
        output_dir: str = "output",
        target_column: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None,
        generate_visuals: bool = True,
    ) -> ReportGenerator:
        """
        Run full forensic investigation.

        Args:
            output_dir: Directory for output files.
            target_column: Optional target column for ML audit.
            reference_data: Optional reference dataset for drift detection.
            generate_visuals: Whether to generate visualization files.

        Returns:
            ReportGenerator instance for saving reports.
        """
        logger.info("Starting full investigation...")

        # Run analysis engine
        self._results = self.engine.run_full_analysis(
            target_column=target_column,
            reference_data=reference_data,
        )

        # Generate fingerprint
        fp = DatasetFingerprint(self.data, self.source_path)
        fingerprint = fp.generate()

        # Generate visualizations
        viz_paths = {}
        if generate_visuals:
            try:
                charts = ForensicCharts(self.data, self.engine.profile)
                viz_paths = charts.generate_all(
                    self.engine.findings,
                    self.engine.health_score,
                    output_dir,
                )
            except Exception as e:
                logger.warning("Visualization generation failed: %s", e)

        # Build legacy findings dict for backward compatibility
        self.findings = self._build_legacy_findings(viz_paths)

        # Create report generator
        self.report = ReportGenerator(
            findings=self.engine.findings,
            health_score=self.engine.health_score,
            profile=self.engine.profile,
            fingerprint=fingerprint,
            source_path=self.source_path,
            output_dir=output_dir,
        )

        return self.report

    def _build_legacy_findings(self, viz_paths: dict) -> dict:
        """
        Build a legacy-compatible findings dictionary.
        This preserves backward compatibility with the old app.py.
        """
        findings = self.engine.findings
        health = self.engine.health_score
        profile = self.engine.profile

        # Group findings by category
        by_category = {}
        for f in findings:
            if f.category not in by_category:
                by_category[f.category] = []
            by_category[f.category].append(f)

        # Build legacy structure
        legacy = {
            "provenance": {
                "metadata": {
                    "source": self.source_path,
                    "rows": profile.row_count if profile else 0,
                    "columns": profile.column_count if profile else 0,
                    "memory_usage_mb": profile.memory_usage_mb if profile else 0,
                },
                "credibility_score": health.overall if health else 0,
                "concerns": [
                    f.description for f in by_category.get("provenance", [])
                ],
            },
            "bias": {
                "overall_bias_score": max(
                    0, 100 - health.components.get("bias_risk", type("", (), {"score": 100})).score
                ) if health else 0,
                "critical_warnings": [
                    f.description for f in by_category.get("bias", [])
                    if f.severity in (Severity.CRITICAL, Severity.HIGH)
                ],
            },
            "anomalies": {
                "overall_anomaly_score": max(
                    0, 100 - health.components.get("anomaly_risk", type("", (), {"score": 100})).score
                ) if health else 0,
                "red_flags": [
                    f.description for f in by_category.get("anomaly", [])
                    if f.severity in (Severity.CRITICAL, Severity.HIGH)
                ],
            },
            "missing": {
                "overall_missing_score": max(
                    0, 100 - health.components.get("completeness", type("", (), {"score": 100})).score
                ) if health else 0,
                "overview": {
                    "total_cells": profile.total_cells if profile else 0,
                    "missing_cells": profile.missing_cells if profile else 0,
                    "missing_percentage": profile.missing_percentage if profile else 0,
                    "complete_percentage": 100 - (profile.missing_percentage if profile else 0),
                    "columns_with_missing": sum(
                        1 for cp in (profile.columns.values() if profile else [])
                        if cp.null_count > 0
                    ),
                },
                "concerns": [
                    f.description for f in by_category.get("missing_data", [])
                ],
            },
            "robustness": {
                "overall_robustness_score": (
                    health.components.get("robustness", type("", (), {"score": 100})).score
                ) if health else 100,
                "fragility_warnings": [
                    f.description for f in by_category.get("robustness", [])
                ],
            },
            "visualizations": viz_paths,
            "health_score": health.to_dict() if health else {},
            "all_findings": [f.to_dict() for f in findings],
        }

        return legacy


__all__ = [
    "DataAutopsy",
    "AnalysisConfig",
    "Finding",
    "Severity",
    "SemanticType",
    "HealthScore",
    "DatasetProfile",
]
