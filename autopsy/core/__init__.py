"""
Data Autopsy Core Module

Core infrastructure: models, scoring, schema inference, configuration, and engine.
"""

from .models import Finding, Severity, SemanticType
from .scoring import ScoringEngine
from .schema import SchemaAnalyzer
from .config import AnalysisConfig
from .engine import AnalysisEngine

__all__ = [
    "Finding",
    "Severity",
    "SemanticType",
    "ScoringEngine",
    "SchemaAnalyzer",
    "AnalysisConfig",
    "AnalysisEngine",
]
