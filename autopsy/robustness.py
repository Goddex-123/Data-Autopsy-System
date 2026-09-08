"""
Legacy module. Please use autopsy.detectors.robustness_tester instead.
"""
import warnings

warnings.warn(
    "autopsy.robustness is deprecated and will be removed. "
    "Use autopsy.detectors.robustness_tester instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autopsy.detectors.robustness_tester import RobustnessTester
