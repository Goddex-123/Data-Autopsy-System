"""Tests for the privacy/PII detector."""

import pytest
import pandas as pd
import numpy as np

from autopsy.detectors.privacy_detector import PrivacyDetector
from autopsy.core.models import Severity


class TestPIIDetection:
    def test_detects_email(self, df_with_pii):
        detector = PrivacyDetector(df_with_pii)
        findings = detector.analyze()
        email_findings = [f for f in findings if "email" in f.evidence.get("pii_type", "")]
        assert len(email_findings) >= 1

    def test_detects_phone(self, df_with_pii):
        detector = PrivacyDetector(df_with_pii)
        findings = detector.analyze()
        phone_findings = [f for f in findings if "phone" in f.evidence.get("pii_type", "")]
        assert len(phone_findings) >= 1

    def test_detects_ip(self, df_with_pii):
        detector = PrivacyDetector(df_with_pii)
        findings = detector.analyze()
        ip_findings = [f for f in findings if "ip_address" in f.evidence.get("pii_type", "")]
        assert len(ip_findings) >= 1

    def test_detects_name_by_column_name(self, df_with_pii):
        detector = PrivacyDetector(df_with_pii)
        findings = detector.analyze()
        name_findings = [f for f in findings if "name" in f.evidence.get("pii_type", "")]
        assert len(name_findings) >= 1


class TestNoValueExposure:
    def test_no_actual_values_in_evidence(self, df_with_pii):
        """PII findings must NEVER contain actual sensitive values."""
        detector = PrivacyDetector(df_with_pii)
        findings = detector.analyze()
        for f in findings:
            evidence_str = str(f.evidence)
            assert "@example.com" not in evidence_str
            assert "555-" not in evidence_str
            assert "192.168" not in evidence_str


class TestNoPIIDataset:
    def test_no_pii_no_findings(self):
        df = pd.DataFrame({
            "quantity": range(50),
            "price": np.random.uniform(10, 100, 50),
            "category": np.random.choice(["A", "B"], 50),
        })
        detector = PrivacyDetector(df)
        findings = detector.analyze()
        # Should have no PII findings (maybe no overall risk either)
        pii_findings = [f for f in findings if "PII" in f.id]
        assert len(pii_findings) == 0


class TestPrivacyRisk:
    def test_multiple_pii_columns_high_risk(self, df_with_pii):
        detector = PrivacyDetector(df_with_pii)
        findings = detector.analyze()
        risk_findings = [f for f in findings if "RISK" in f.id]
        assert len(risk_findings) >= 1
        assert risk_findings[0].severity in (Severity.CRITICAL, Severity.HIGH)
