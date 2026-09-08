"""Tests for the quality detector."""

import pytest
import numpy as np
import pandas as pd

from autopsy.detectors.quality_detector import QualityDetector
from autopsy.core.config import AnalysisConfig


class TestConstantColumns:
    def test_detects_constant(self, constant_df):
        detector = QualityDetector(constant_df)
        findings = detector._detect_constant_columns()
        const_findings = [f for f in findings if "CONST" in f.id]
        assert len(const_findings) >= 2  # const_int and const_str


class TestCaseInconsistency:
    def test_detects_case_variants(self):
        df = pd.DataFrame({
            "city": ["Mumbai", "mumbai", "MUMBAI", "Delhi", "delhi", "Chennai"],
        })
        detector = QualityDetector(df)
        findings = detector._detect_case_inconsistency()
        assert len(findings) >= 1

    def test_no_case_issue_when_consistent(self):
        df = pd.DataFrame({"city": ["Mumbai", "Delhi", "Chennai"] * 10})
        detector = QualityDetector(df)
        findings = detector._detect_case_inconsistency()
        assert len(findings) == 0


class TestWhitespace:
    def test_detects_whitespace(self):
        df = pd.DataFrame({"name": ["  Alice", "Bob  ", " Charlie ", "Dave"]})
        detector = QualityDetector(df)
        findings = detector._detect_whitespace_issues()
        assert len(findings) >= 1


class TestMixedTypes:
    def test_detects_mixed_types(self):
        df = pd.DataFrame({
            "mixed": ["100", "hello", "200", "world", "300", "foo"] * 10,
        })
        detector = QualityDetector(df)
        findings = detector._detect_mixed_types()
        assert len(findings) >= 1


class TestDuplicateColumns:
    def test_detects_duplicate_columns(self):
        df = pd.DataFrame({
            "col_a": range(50),
            "col_b": range(50),
            "col_c": range(50, 100),
        })
        detector = QualityDetector(df)
        findings = detector._detect_duplicate_columns()
        assert len(findings) >= 1
