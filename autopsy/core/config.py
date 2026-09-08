"""
Analysis Configuration

Central configuration for all analysis modules.
Allows customization of thresholds, limits, and behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AnalysisConfig:
    """
    Configuration for the Data Autopsy analysis pipeline.

    All thresholds and limits are documented and configurable.
    """

    # --- General ---
    max_sample_size: int = 100_000
    """Maximum rows to use for expensive operations (sampling applied above this)."""

    random_seed: int = 42
    """Seed for reproducibility of sampling and ML-based detection."""

    # --- Data Quality ---
    duplicate_warning_threshold: float = 5.0
    """Percentage of duplicates above which a warning is raised."""

    constant_column_threshold: int = 1
    """Number of unique values at or below which a column is flagged as constant."""

    high_cardinality_threshold: float = 0.95
    """Uniqueness ratio above which a column is flagged as high-cardinality."""

    # --- Missing Data ---
    missing_high_threshold: float = 20.0
    """Column missing percentage above which severity is HIGH."""

    missing_critical_threshold: float = 50.0
    """Column missing percentage above which severity is CRITICAL."""

    sentinel_frequency_threshold: float = 10.0
    """Percentage above which a suspected sentinel value is flagged."""

    # --- Anomaly Detection ---
    benford_min_sample_size: int = 100
    """Minimum sample size for Benford's Law analysis."""

    benford_min_magnitude_range: float = 2.0
    """Minimum orders of magnitude range for Benford applicability."""

    outlier_iqr_multiplier: float = 1.5
    """IQR multiplier for outlier detection bounds."""

    zscore_threshold: float = 3.0
    """Z-score threshold for extreme outlier detection."""

    isolation_forest_contamination: float = 0.05
    """Contamination parameter for Isolation Forest."""

    isolation_forest_max_samples: int = 10_000
    """Max samples for Isolation Forest (for performance)."""

    # --- Bias ---
    imbalance_ratio_warning: float = 10.0
    """Imbalance ratio above which a warning is raised."""

    expected_distributions: Optional[Dict[str, Dict[str, float]]] = None
    """Optional population distributions for bias comparison."""

    # --- Privacy ---
    pii_sample_size: int = 1000
    """Number of values to sample for PII pattern detection."""

    pii_confidence_threshold: float = 0.5
    """Minimum match rate to flag a column as potential PII."""

    # --- ML Audit ---
    leakage_correlation_threshold: float = 0.90
    """Correlation threshold above which target leakage is suspected."""

    high_correlation_threshold: float = 0.70
    """Correlation threshold for flagging strong feature relationships."""

    feature_redundancy_threshold: float = 0.95
    """Correlation threshold for flagging redundant features."""

    # --- Drift ---
    drift_low_threshold: float = 0.1
    """Drift score below which drift is LOW."""

    drift_high_threshold: float = 0.2
    """Drift score above which drift is HIGH."""

    # --- Scoring Weights ---
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "completeness": 0.20,
        "validity": 0.15,
        "consistency": 0.15,
        "anomaly_risk": 0.15,
        "bias_risk": 0.10,
        "privacy_risk": 0.10,
        "robustness": 0.15,
    })
    """Weights for overall health score components. Must sum to 1.0."""

    def __post_init__(self):
        """Validate configuration."""
        weight_sum = sum(self.score_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Score weights must sum to 1.0, got {weight_sum:.3f}"
            )
