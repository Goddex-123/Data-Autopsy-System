"""Tests for the missing data analyzer."""

import pytest
import numpy as np
import pandas as pd

from autopsy.detectors.missing_analyzer import MissingDataAnalyzer
from autopsy.core.models import Severity


class TestColumnMissingness:
    def test_detects_missing_columns(self, df_with_missing):
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._analyze_column_missingness()
        assert len(findings) > 0
        missing_cols = [f.column for f in findings]
        assert "high_missing" in missing_cols

    def test_complete_column_no_finding(self, df_with_missing):
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._analyze_column_missingness()
        assert "complete_col" not in [f.column for f in findings]

    def test_high_missing_severity(self, df_with_missing):
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._analyze_column_missingness()
        high = [f for f in findings if f.column == "high_missing"]
        assert len(high) == 1
        assert high[0].severity in (Severity.CRITICAL, Severity.HIGH)


class TestMissingMechanisms:
    def test_no_likely_mcar_language(self, df_with_missing):
        """The term 'Likely MCAR' should never appear."""
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._analyze_missing_mechanisms()
        for f in findings:
            assert "likely mcar" not in f.description.lower()
            assert "likely mcar" not in f.title.lower()

    def test_mnar_not_claimed(self, df_with_missing):
        """MNAR should never be claimed as detected."""
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer.analyze()
        for f in findings:
            assert "likely mnar" not in f.description.lower()

    def test_mnar_untestable_stated(self, df_with_missing):
        """Limitations should mention MNAR cannot be determined from observed data."""
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._analyze_missing_mechanisms()
        mcar_findings = [f for f in findings if "MCAR" in f.id]
        for f in mcar_findings:
            assert f.limitations is not None
            assert "mnar" in f.limitations.lower() or "observed data" in f.limitations.lower()

    def test_correct_insufficient_evidence_language(self, df_with_missing):
        """Should say 'insufficient evidence to reject MCAR' not 'Likely MCAR'."""
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._analyze_missing_mechanisms()
        mcar_findings = [f for f in findings if "MCAR" in f.id]
        for f in mcar_findings:
            assert "insufficient evidence" in f.description.lower() or "no " in f.description.lower()


class TestSentinelValues:
    def test_detects_sentinel_values(self, df_with_missing):
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._detect_sentinel_values()
        sentinel = [f for f in findings if "SENTINEL" in f.id]
        assert len(sentinel) >= 1

    def test_sentinel_column_correct(self, df_with_missing):
        analyzer = MissingDataAnalyzer(df_with_missing)
        findings = analyzer._detect_sentinel_values()
        sentinel = [f for f in findings if "SENTINEL" in f.id]
        if sentinel:
            assert sentinel[0].column == "sentinel_col"


class TestMissingPatterns:
    def test_sequential_missing(self):
        """Detect consecutive runs of missing data."""
        n = 100
        data = np.random.randn(n)
        data[20:35] = np.nan  # 15 consecutive NaN
        df = pd.DataFrame({"values": data})
        analyzer = MissingDataAnalyzer(df)
        findings = analyzer._analyze_missing_patterns()
        seq = [f for f in findings if "SEQ" in f.id]
        assert len(seq) >= 1
