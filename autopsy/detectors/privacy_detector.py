"""
Privacy / PII Detector

Detects potential personally identifiable information (PII) using:
- Column name heuristics
- Regex pattern matching on sampled values
- Semantic type inference

IMPORTANT: Never exposes actual sensitive values in findings.
"""

import logging
import re
from typing import List, Optional

import pandas as pd

from ..core.models import Finding, Severity, DatasetProfile, SemanticType
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)

# PII patterns
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_PHONE_RE = re.compile(r"^[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}$")
_IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_CREDIT_CARD_RE = re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$")
_AADHAAR_RE = re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$")
_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_URL_RE = re.compile(r"^https?://[^\s]+$")

_PII_COLUMN_KEYWORDS = {
    "email": "email",
    "e_mail": "email",
    "phone": "phone",
    "telephone": "phone",
    "tel": "phone",
    "mobile": "phone",
    "cell": "phone",
    "ip_address": "ip_address",
    "ip": "ip_address",
    "ssn": "government_id",
    "social_security": "government_id",
    "aadhaar": "government_id",
    "aadhar": "government_id",
    "pan": "government_id",
    "passport": "government_id",
    "credit_card": "financial",
    "card_number": "financial",
    "card_no": "financial",
    "cvv": "financial",
    "name": "name",
    "first_name": "name",
    "last_name": "name",
    "full_name": "name",
    "surname": "name",
    "address": "address",
    "street": "address",
    "city": "address",
    "zip": "address",
    "zipcode": "address",
    "postal": "address",
    "latitude": "geolocation",
    "longitude": "geolocation",
    "lat": "geolocation",
    "lng": "geolocation",
    "lon": "geolocation",
}


class PrivacyDetector:
    """
    Detects potential PII and privacy risks in datasets.
    Never exposes actual sensitive values in findings.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
        profile: Optional[DatasetProfile] = None,
    ):
        self.data = data
        self.config = config or AnalysisConfig()
        self.profile = profile

    def analyze(self) -> List[Finding]:
        """Run PII detection and return findings."""
        findings: List[Finding] = []
        pii_columns = []

        for col in self.data.columns:
            pii_type = self._detect_pii_column(col)
            if pii_type:
                pii_columns.append((col, pii_type))
                non_null_count = int(self.data[col].notna().sum())
                populated_pct = non_null_count / len(self.data) * 100 if len(self.data) > 0 else 0

                findings.append(Finding(
                    id=f"PRIV-PII-{col[:20].upper().replace(' ', '_')}",
                    category="privacy",
                    severity=Severity.HIGH,
                    confidence=round(pii_type["confidence"], 2),
                    title=f"Potential PII detected: '{col}' ({pii_type['type']})",
                    description=(
                        f"Column '{col}' appears to contain {pii_type['type']} data "
                        f"({populated_pct:.1f}% populated). "
                        f"Detection method: {pii_type['method']}."
                    ),
                    column=col,
                    evidence={
                        "pii_type": pii_type["type"],
                        "detection_method": pii_type["method"],
                        "populated_percentage": round(populated_pct, 1),
                        "match_rate": round(pii_type.get("match_rate", 0) * 100, 1),
                        # Deliberately NOT including sample values
                    },
                    recommendation=(
                        "Review this column for PII. Consider removing, tokenizing, "
                        "hashing, or restricting access before sharing the dataset."
                    ),
                ))

        # Overall privacy risk summary
        if pii_columns:
            risk_level = "HIGH" if len(pii_columns) >= 3 else "MEDIUM" if len(pii_columns) >= 1 else "LOW"
            severity = Severity.CRITICAL if len(pii_columns) >= 3 else Severity.HIGH

            findings.append(Finding(
                id="PRIV-RISK-001",
                category="privacy",
                severity=severity,
                confidence=0.85,
                title=f"Privacy risk: {len(pii_columns)} potential PII column(s)",
                description=(
                    f"Dataset contains {len(pii_columns)} column(s) with potential "
                    f"personally identifiable information. Risk level: {risk_level}."
                ),
                evidence={
                    "pii_columns": [col for col, _ in pii_columns],
                    "pii_types": [t["type"] for _, t in pii_columns],
                    "risk_level": risk_level,
                },
                recommendation=(
                    "Perform a privacy review before sharing or publishing this dataset. "
                    "Consider data anonymization, pseudonymization, or access controls."
                ),
            ))

        return findings

    def _detect_pii_column(self, col: str) -> Optional[dict]:
        """
        Check if a column contains PII.

        Returns dict with type, confidence, method, or None.
        """
        # 1. Column name heuristics
        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        for keyword, pii_type in _PII_COLUMN_KEYWORDS.items():
            if keyword in col_lower:
                return {
                    "type": pii_type,
                    "confidence": 0.7,
                    "method": "column_name_heuristic",
                    "match_rate": 0,
                }

        # 2. Value-based detection (sample)
        if self.data[col].dtype == "object":
            return self._detect_pii_in_values(col)

        return None

    def _detect_pii_in_values(self, col: str) -> Optional[dict]:
        """Detect PII patterns in column values."""
        non_null = self.data[col].dropna().astype(str)
        if len(non_null) == 0:
            return None

        sample = non_null.head(min(self.config.pii_sample_size, len(non_null)))

        # Test each pattern
        patterns = [
            ("email", _EMAIL_RE),
            ("phone", _PHONE_RE),
            ("ip_address", _IP_RE),
            ("financial", _CREDIT_CARD_RE),
            ("government_id", _AADHAAR_RE),
            ("government_id", _PAN_RE),
        ]

        for pii_type, pattern in patterns:
            matches = sample.str.match(pattern, na=False)
            match_rate = matches.mean()

            if match_rate >= self.config.pii_confidence_threshold:
                return {
                    "type": pii_type,
                    "confidence": min(0.9, match_rate),
                    "method": "value_pattern_matching",
                    "match_rate": match_rate,
                }

        return None
