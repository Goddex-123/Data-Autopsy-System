"""
Legacy module. Please use autopsy.detectors.missing_analyzer instead.
"""
import warnings

warnings.warn(
    "autopsy.missing_analyzer is deprecated and will be removed. "
    "Use autopsy.detectors.missing_analyzer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autopsy.detectors.missing_analyzer import MissingAnalyzer
