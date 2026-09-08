"""
Legacy module. Please use autopsy.detectors.anomaly_detector instead.
"""
import warnings

warnings.warn(
    "autopsy.anomaly_detector is deprecated and will be removed. "
    "Use autopsy.detectors.anomaly_detector instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autopsy.detectors.anomaly_detector import AnomalyDetector
