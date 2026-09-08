"""
Legacy module. Please use autopsy.detectors.bias_detector instead.
"""
import warnings

warnings.warn(
    "autopsy.bias_detector is deprecated and will be removed. "
    "Use autopsy.detectors.bias_detector instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autopsy.detectors.bias_detector import BiasDetector
