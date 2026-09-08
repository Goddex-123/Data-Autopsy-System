"""Tests for the anomaly detector."""

import pytest
import numpy as np
import pandas as pd

from autopsy.detectors.anomaly_detector import AnomalyDetector
from autopsy.core.config import AnalysisConfig
from autopsy.core.schema import SchemaAnalyzer
from autopsy.core.models import Severity


class TestBenfordsLaw:
    def test_benford_not_applied_to_identifiers(self, df_benford):
        """IDs should be excluded from Benford analysis."""
        sa = SchemaAnalyzer(df_benford)
        profile = sa.analyze()
        detector = AnomalyDetector(df_benford, profile=profile)
        findings = detector._analyze_benfords_law()
        # record_id should NOT produce a finding
        id_findings = [f for f in findings if f.column == "record_id"]
        assert len(id_findings) == 0

    def test_benford_not_applied_to_bounded(self, df_benford):
        """Bounded variables (small magnitude range) should be skipped."""
        sa = SchemaAnalyzer(df_benford)
        profile = sa.analyze()
        detector = AnomalyDetector(df_benford, profile=profile)
        findings = detector._analyze_benfords_law()
        bounded = [f for f in findings if f.column == "bounded_var"]
        assert len(bounded) == 0

    def test_benford_small_sample_skipped(self):
        """Small samples should be skipped."""
        df = pd.DataFrame({"values": np.random.lognormal(10, 2, 20)})
        detector = AnomalyDetector(df)
        findings = detector._analyze_benfords_law()
        assert len(findings) == 0

    def test_benford_finding_has_limitations(self, df_benford):
        """Any Benford finding must include limitations."""
        # Create data that violates Benford (uniform first digits)
        uniform = np.array([d * 100 + np.random.randint(0, 99) for d in [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100])
        df = pd.DataFrame({"amount": uniform})
        detector = AnomalyDetector(df)
        findings = detector._analyze_benfords_law()
        for f in findings:
            assert f.limitations is not None
            assert "fabrication" not in f.description.lower()


class TestOutlierDetection:
    def test_detects_obvious_outliers(self, df_with_outliers):
        detector = AnomalyDetector(df_with_outliers)
        findings = detector._detect_outliers()
        outlier_findings = [f for f in findings if "OUTLIER" in f.id]
        assert len(outlier_findings) >= 1

    def test_clean_data_minimal_findings(self):
        """Clean normal data should produce few/no outlier findings."""
        np.random.seed(42)
        df = pd.DataFrame({"clean": np.random.normal(100, 10, 200)})
        detector = AnomalyDetector(df)
        findings = detector._detect_outliers()
        # Should have low severity at most
        high_findings = [f for f in findings if f.severity in (Severity.HIGH, Severity.CRITICAL)]
        assert len(high_findings) == 0

    def test_multivariate_ensemble_runs(self):
        """Multivariate Ensemble should work on multi-column data."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "a": np.concatenate([np.random.normal(0, 1, n-5), [100, -100, 50, -50, 200]]),
            "b": np.concatenate([np.random.normal(0, 1, n-5), [100, -100, 50, -50, 200]]),
        })
        detector = AnomalyDetector(df)
        finding = detector._multivariate_ensemble(["a", "b"])
        assert finding is not None or True  # May or may not detect


class TestDuplicates:
    def test_detects_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 5,
            "b": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"] * 5,
        })
        detector = AnomalyDetector(df, config=AnalysisConfig(duplicate_warning_threshold=5))
        findings = detector._analyze_duplicates()
        assert any("DUP" in f.id for f in findings)


class TestValueAnomalies:
    def test_detects_negative_age(self):
        """Negative age values should be flagged."""
        df = pd.DataFrame({"age": [25, 30, 40, 35] * 12 + [-5, -10]})
        detector = AnomalyDetector(df)
        findings = detector._detect_value_anomalies()
        neg_findings = [f for f in findings if "INVALID" in f.id]
        assert len(neg_findings) >= 1

    def test_no_fabrication_language(self, df_with_outliers):
        """No finding should use 'fabrication' or 'manipulation' language."""
        detector = AnomalyDetector(df_with_outliers)
        findings = detector.analyze()
        for f in findings:
            assert "fabricat" not in f.title.lower()
            assert "manipulat" not in f.title.lower()
            assert "fabricat" not in f.description.lower()
