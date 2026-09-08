"""
Report Generator

Orchestrates report generation in multiple formats.
Delegates to format-specific renderers.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.models import Finding, HealthScore, DatasetProfile, Severity
from ..core.scoring import ScoringEngine

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates professional audit reports from analysis results.
    Supports HTML, JSON, and Markdown output formats.
    """

    def __init__(
        self,
        findings: List[Finding],
        health_score: HealthScore,
        profile: DatasetProfile,
        fingerprint: dict,
        source_path: str = "DataFrame",
        output_dir: str = "output",
    ):
        self.findings = findings
        self.health_score = health_score
        self.profile = profile
        self.fingerprint = fingerprint
        self.source_path = source_path
        self.output_dir = output_dir
        self.timestamp = datetime.now(timezone.utc)

    def save(self, filename: str) -> str:
        """Save report in the format determined by file extension."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        if filename.endswith(".html"):
            content = self._generate_html()
        elif filename.endswith(".json"):
            content = self._generate_json()
        elif filename.endswith(".md"):
            content = self._generate_markdown()
        else:
            content = self._generate_markdown()
            filepath += ".md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Report saved to: %s", filepath)
        return filepath

    def print_summary(self):
        """Print a console summary."""
        severity_counts = ScoringEngine.count_by_severity(self.findings)

        print("\n" + "=" * 55)
        print("🔬 DATA AUTOPSY REPORT")
        print("=" * 55)
        print(f"\nSource: {self.source_path}")
        print(f"Rows: {self.profile.row_count:,}  |  Columns: {self.profile.column_count}")
        print(f"Missing: {self.profile.missing_percentage:.1f}%  |  Duplicates: {self.profile.duplicate_percentage:.1f}%")
        print(f"\n{self.health_score.verdict_emoji} Overall Health: {self.health_score.overall:.0f}/100 — {self.health_score.verdict}")
        print(f"\nFindings: {len(self.findings)} total")
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "ℹ️"}[sev]
                print(f"  {emoji} {sev.upper()}: {count}")
        print("=" * 55 + "\n")

    def _generate_json(self) -> str:
        """Generate JSON report."""
        report = {
            "metadata": {
                "source": self.source_path,
                "analysis_date": self.timestamp.isoformat(),
                "generator": "Data Autopsy System v2.0",
            },
            "fingerprint": self.fingerprint,
            "profile": self.profile.to_dict(),
            "health_score": self.health_score.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "severity_counts": ScoringEngine.count_by_severity(self.findings),
        }
        return json.dumps(report, indent=2, default=str)

    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        severity_counts = ScoringEngine.count_by_severity(self.findings)
        lines = [
            "# 🔬 Data Autopsy Report",
            "",
            f"**Source:** {self.source_path}",
            f"**Analysis Date:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"**{self.health_score.verdict_emoji} Overall Health: {self.health_score.overall:.0f}/100 — {self.health_score.verdict}**",
            "",
            "| Component | Score | Weight | Findings |",
            "|-----------|-------|--------|----------|",
        ]

        for name, comp in self.health_score.components.items():
            lines.append(f"| {comp.name} | {comp.score:.0f}/100 | {comp.weight:.0%} | {comp.findings_count} |")

        lines.extend([
            "",
            f"**Total Findings:** {len(self.findings)}",
            "",
            "| Severity | Count |",
            "|----------|-------|",
        ])
        for sev in ["critical", "high", "medium", "low", "info"]:
            c = severity_counts.get(sev, 0)
            if c > 0:
                lines.append(f"| {sev.upper()} | {c} |")

        lines.extend(["", "---", "", "## Dataset Profile", ""])
        lines.append(f"- **Rows:** {self.profile.row_count:,}")
        lines.append(f"- **Columns:** {self.profile.column_count}")
        lines.append(f"- **Missing:** {self.profile.missing_percentage:.1f}%")
        lines.append(f"- **Duplicates:** {self.profile.duplicate_percentage:.1f}%")
        lines.append(f"- **Memory:** {self.profile.memory_usage_mb:.2f} MB")

        # Findings by category
        categories = sorted(set(f.category for f in self.findings))
        for cat in categories:
            cat_findings = [f for f in self.findings if f.category == cat]
            lines.extend(["", "---", "", f"## {cat.replace('_', ' ').title()}", ""])
            for f in cat_findings:
                lines.append(f"### {f.severity_emoji} {f.title}")
                lines.append(f"*Severity: {f.severity.value} | Confidence: {f.confidence:.0%}*")
                lines.append(f"\n{f.description}")
                if f.recommendation:
                    lines.append(f"\n**Recommendation:** {f.recommendation}")
                if f.limitations:
                    lines.append(f"\n*Limitation: {f.limitations}*")
                lines.append("")

        lines.extend(["---", "", "*Generated by Data Autopsy System v2.0*"])
        return "\n".join(lines)

    def _generate_html(self) -> str:
        """Generate professional HTML report."""
        from .html_report import render_html_report
        return render_html_report(
            findings=self.findings,
            health_score=self.health_score,
            profile=self.profile,
            fingerprint=self.fingerprint,
            source_path=self.source_path,
            timestamp=self.timestamp,
        )
