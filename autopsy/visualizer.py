"""
Legacy module. Please use autopsy.visualization.charts instead.
"""
import warnings

warnings.warn(
    "autopsy.visualizer is deprecated and will be removed. "
    "Use autopsy.visualization.charts instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autopsy.visualization.charts import ForensicVisualizer
