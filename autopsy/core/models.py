"""
Core Data Models

Unified finding model, severity levels, and semantic types used across all detectors.
Every detector returns findings in this consistent structure.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class Severity(str, Enum):
    """Finding severity levels, ordered from most to least severe."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def numeric(self) -> int:
        """Numeric value for sorting (higher = more severe)."""
        return {
            Severity.CRITICAL: 5,
            Severity.HIGH: 4,
            Severity.MEDIUM: 3,
            Severity.LOW: 2,
            Severity.INFO: 1,
        }[self]


class EvidenceLevel(str, Enum):
    """Consistent evidence strength levels across the system."""
    VERY_LOW = "very low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very high"


class SemanticType(str, Enum):
    """Inferred semantic types for columns."""
    IDENTIFIER = "identifier"
    NUMERICAL_CONTINUOUS = "numerical/continuous"
    NUMERICAL_DISCRETE = "numerical/discrete"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    CONSTANT = "constant"
    URL = "url"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    GEOLOCATION = "geolocation"
    PII_NAME = "pii/name"
    PII_ADDRESS = "pii/address"
    PII_SSN = "pii/ssn"
    PII_EMAIL = "pii/email"
    PII_PHONE = "pii/phone"
    PII_IP = "pii/ip"
    PII_OTHER = "pii/other"
    UNKNOWN = "unknown"


class FindingCategory(str, Enum):
    """Categories for findings."""
    DATA_QUALITY = "data_quality"
    MISSING_DATA = "missing_data"
    ANOMALY = "anomaly"
    BIAS = "bias"
    PRIVACY = "privacy"
    ML_AUDIT = "ml_audit"
    DRIFT = "drift"
    PROVENANCE = "provenance"
    ROBUSTNESS = "robustness"


@dataclass
class Finding:
    """
    Unified finding object returned by all detectors.

    Every analysis result across the entire system uses this structure,
    ensuring consistent reporting and evidence-based conclusions.
    """
    id: str
    category: str
    severity: Severity
    evidence_score: float  # Replaces 'confidence' to avoid p-value confusion
    title: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    statistical_test: Optional[str] = None
    recommendation: str = ""
    limitations: Optional[str] = None
    column: Optional[str] = None
    columns: Optional[List[str]] = None

    def __post_init__(self):
        """Validate finding fields."""
        if not 0.0 <= self.evidence_score <= 1.0:
            raise ValueError(f"Evidence score must be between 0.0 and 1.0, got {self.evidence_score}")
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)

    @property
    def evidence_level(self) -> EvidenceLevel:
        """Map the numeric evidence score to a consistent qualitative level."""
        if self.evidence_score >= 0.9:
            return EvidenceLevel.VERY_HIGH
        elif self.evidence_score >= 0.7:
            return EvidenceLevel.HIGH
        elif self.evidence_score >= 0.5:
            return EvidenceLevel.MODERATE
        elif self.evidence_score >= 0.3:
            return EvidenceLevel.LOW
        else:
            return EvidenceLevel.VERY_LOW

    def to_dict(self) -> dict:
        """Convert finding to a serializable dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        result["evidence_level"] = self.evidence_level.value
        return result

    def to_json(self) -> str:
        """Serialize finding to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @property
    def severity_emoji(self) -> str:
        """Emoji representation for UI display."""
        return {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🔵",
            Severity.INFO: "ℹ️",
        }[self.severity]


@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any] = field(default_factory=list)
    is_identifier: bool = False
    is_constant: bool = False

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        result = asdict(self)
        result["semantic_type"] = self.semantic_type.value
        return result


@dataclass
class DatasetProfile:
    """Complete profile of a dataset."""
    row_count: int
    column_count: int
    total_cells: int
    missing_cells: int
    missing_percentage: float
    duplicate_rows: int
    duplicate_percentage: float
    memory_usage_mb: float
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        result = asdict(self)
        for col_name, col_profile in result.get("columns", {}).items():
            col_profile["semantic_type"] = self.columns[col_name].semantic_type.value
        return result


@dataclass
class ScoreComponent:
    """A single component of the overall health score."""
    name: str
    score: float  # 0-100
    weight: float  # 0-1
    details: str = ""
    findings_count: int = 0

    def weighted_score(self) -> float:
        """Return the weighted contribution of this component."""
        return self.score * self.weight


@dataclass
class HealthScore:
    """Overall dataset health score composed of multiple components."""
    overall: float  # 0-100
    components: Dict[str, ScoreComponent] = field(default_factory=dict)
    verdict: str = ""
    verdict_emoji: str = ""

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "overall": round(self.overall, 1),
            "verdict": self.verdict,
            "components": {
                name: {
                    "score": round(comp.score, 1),
                    "weight": comp.weight,
                    "weighted_score": round(comp.weighted_score(), 1),
                    "details": comp.details,
                    "findings_count": comp.findings_count,
                }
                for name, comp in self.components.items()
            },
        }
