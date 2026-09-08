"""
Legacy module. Please use autopsy.provenance.fingerprint instead.
"""
import warnings

warnings.warn(
    "autopsy.provenance is deprecated and will be removed. "
    "Use autopsy.provenance.fingerprint instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autopsy.provenance.fingerprint import DatasetFingerprint as DataProvenance
