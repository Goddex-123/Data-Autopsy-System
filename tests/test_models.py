"""Tests for core models."""

import pytest
from autopsy.core.models import Finding, Severity, SemanticType, ColumnProfile, HealthScore, ScoreComponent


class TestSeverity:
    def test_severity_ordering(self):
        assert Severity.CRITICAL.numeric > Severity.HIGH.numeric
        assert Severity.HIGH.numeric > Severity.MEDIUM.numeric
        assert Severity.MEDIUM.numeric > Severity.LOW.numeric
        assert Severity.LOW.numeric > Severity.INFO.numeric

    def test_severity_from_string(self):
        assert Severity("critical") == Severity.CRITICAL
        assert Severity("info") == Severity.INFO


class TestFinding:
    def test_basic_creation(self):
        f = Finding(
            id="TEST-001",
            category="test",
            severity=Severity.HIGH,
            evidence_score=0.8,
            title="Test finding",
            description="A test",
        )
        assert f.id == "TEST-001"
        assert f.severity == Severity.HIGH

    def test_confidence_validation(self):
        with pytest.raises(ValueError):
            Finding(
                id="BAD", category="test", severity=Severity.LOW,
                evidence_score=1.5, title="Bad", description="Invalid",
            )

    def test_confidence_zero_valid(self):
        f = Finding(
            id="OK", category="test", severity=Severity.INFO,
            evidence_score=0.0, title="Ok", description="Valid",
        )
        assert f.evidence_score == 0.0

    def test_string_severity_conversion(self):
        f = Finding(
            id="CONV", category="test", severity="medium",
            evidence_score=0.5, title="Conv", description="Converts",
        )
        assert f.severity == Severity.MEDIUM

    def test_to_dict(self):
        f = Finding(
            id="DICT", category="anomaly", severity=Severity.HIGH,
            evidence_score=0.9, title="Dict", description="To dict test",
            evidence={"key": "value"},
        )
        d = f.to_dict()
        assert d["severity"] == "high"
        assert d["evidence"] == {"key": "value"}
        assert isinstance(d, dict)

    def test_to_json(self):
        f = Finding(
            id="JSON", category="test", severity=Severity.LOW,
            evidence_score=0.5, title="JSON", description="JSON test",
        )
        j = f.to_json()
        assert '"severity": "low"' in j

    def test_severity_emoji(self):
        f = Finding(
            id="EMOJI", category="test", severity=Severity.CRITICAL,
            evidence_score=1.0, title="Emoji", description="Test",
        )
        assert f.severity_emoji == "🔴"

    def test_optional_fields(self):
        f = Finding(
            id="OPT", category="test", severity=Severity.INFO,
            evidence_score=0.5, title="Opt", description="Optional fields",
            column="col_a", columns=["col_a", "col_b"],
            statistical_test="t-test", recommendation="Do X",
            limitations="Cannot prove Y",
        )
        assert f.column == "col_a"
        assert len(f.columns) == 2


class TestHealthScore:
    def test_to_dict(self):
        hs = HealthScore(
            overall=75.5,
            verdict="HEALTHY",
            verdict_emoji="✅",
            components={
                "comp1": ScoreComponent(
                    name="Completeness", score=90, weight=0.5,
                    details="Good", findings_count=2,
                ),
            },
        )
        d = hs.to_dict()
        assert d["overall"] == 75.5
        assert "comp1" in d["components"]
        assert d["components"]["comp1"]["weighted_score"] == 45.0
