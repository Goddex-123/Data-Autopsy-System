"""
JSON Report Renderer

Machine-readable JSON report format.
"""

import json
from datetime import datetime
from typing import List

from ..core.models import Finding, HealthScore, DatasetProfile
from ..core.scoring import ScoringEngine


def render_json_report(
    findings: List[Finding],
    health_score: HealthScore,
    profile: DatasetProfile,
    fingerprint: dict,
    source_path: str,
    timestamp: datetime,
) -> str:
    """Render a machine-readable JSON report."""
    report = {
        "metadata": {
            "source": source_path,
            "analysis_date": timestamp.isoformat(),
            "generator": "Data Autopsy System v2.0",
        },
        "fingerprint": fingerprint,
        "profile": profile.to_dict(),
        "health_score": health_score.to_dict(),
        "findings": [f.to_dict() for f in findings],
        "severity_counts": ScoringEngine.count_by_severity(findings),
    }
    return json.dumps(report, indent=2, default=str)
