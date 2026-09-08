"""Tests for the bias detector."""

import pytest
import numpy as np
import pandas as pd

from autopsy.detectors.bias_detector import BiasDetector
from autopsy.core.config import AnalysisConfig
from autopsy.core.models import Severity


class TestClassImbalance:
    def test_detects_imbalance(self, df_imbalanced):
        detector = BiasDetector(df_imbalanced)
        findings = detector._detect_class_imbalance()
        imbal = [f for f in findings if "IMBAL" in f.id]
        assert len(imbal) >= 1

    def test_balanced_no_imbalance_finding(self):
        df = pd.DataFrame({
            "cat": np.random.choice(["A", "B", "C"], 300, p=[0.34, 0.33, 0.33]),
        })
        detector = BiasDetector(df)
        findings = detector._detect_class_imbalance()
        imbal = [f for f in findings if "IMBAL" in f.id]
        assert len(imbal) == 0


class TestNoBinaryGenderBias:
    def test_binary_gender_not_automatically_biased(self):
        """Binary gender should NOT automatically trigger a bias finding."""
        df = pd.DataFrame({
            "gender": np.random.choice(["Male", "Female"], 200, p=[0.52, 0.48]),
        })
        detector = BiasDetector(df)
        findings = detector.analyze()
        # No finding about "non-binary representation"
        for f in findings:
            assert "non-binary" not in f.description.lower()
            assert "binary gender" not in f.title.lower()


class TestPopulationComparison:
    def test_detects_mismatch_with_expected(self):
        """When expected distributions are provided, detect mismatches."""
        df = pd.DataFrame({
            "region": np.random.choice(
                ["A", "B", "C"], 500, p=[0.8, 0.15, 0.05]
            ),
        })
        config = AnalysisConfig(
            expected_distributions={
                "region": {"A": 0.33, "B": 0.33, "C": 0.34},
            }
        )
        detector = BiasDetector(df, config)
        findings = detector._detect_population_mismatch()
        pop_findings = [f for f in findings if "POP" in f.id]
        assert len(pop_findings) >= 1

    def test_matching_distribution_no_finding(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "status": np.random.choice(
                ["X", "Y"], 1000, p=[0.50, 0.50]
            ),
        })
        config = AnalysisConfig(
            expected_distributions={"status": {"X": 0.50, "Y": 0.50}}
        )
        detector = BiasDetector(df, config)
        findings = detector._detect_population_mismatch()
        pop_findings = [f for f in findings if "POP" in f.id]
        assert len(pop_findings) == 0


class TestSampleSize:
    def test_small_sample_warning(self):
        df = pd.DataFrame({"a": range(20), "b": range(20)})
        detector = BiasDetector(df)
        findings = detector._detect_sample_size_concerns()
        assert any("SIZE" in f.id for f in findings)

    def test_large_sample_no_warning(self):
        df = pd.DataFrame({"a": range(500)})
        detector = BiasDetector(df)
        findings = detector._detect_sample_size_concerns()
        assert len(findings) == 0


class TestCorrelationConcerns:
    def test_near_perfect_correlation(self):
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({
            "x": x,
            "y": x * 1.001 + np.random.normal(0, 0.001, 100),
        })
        detector = BiasDetector(df)
        findings = detector._detect_correlation_concerns()
        corr_findings = [f for f in findings if "CORR" in f.id]
        assert len(corr_findings) >= 1
        # Should NOT say "manipulation"
        for f in corr_findings:
            assert "manipulat" not in f.description.lower()
