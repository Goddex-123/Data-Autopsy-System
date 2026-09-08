"""
Forensic Visualization Charts

Refactored visualization module that generates matplotlib figures
from Finding objects and dataset profiles.
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

from ..core.models import Finding, DatasetProfile, HealthScore, Severity

logger = logging.getLogger(__name__)

# Professional forensic theme colors
COLORS = {
    "primary": "#2C3E50",
    "critical": "#E74C3C",
    "high": "#E67E22",
    "medium": "#F1C40F",
    "low": "#3498DB",
    "info": "#95A5A6",
    "success": "#27AE60",
    "bg": "#FAFAFA",
}

SEVERITY_COLORS = {
    "critical": COLORS["critical"],
    "high": COLORS["high"],
    "medium": COLORS["medium"],
    "low": COLORS["low"],
    "info": COLORS["info"],
}


class ForensicCharts:
    """Generates professional forensic visualizations."""

    def __init__(self, data: pd.DataFrame, profile: Optional[DatasetProfile] = None):
        self.data = data
        self.profile = profile

    def generate_all(
        self,
        findings: List[Finding],
        health_score: Optional[HealthScore] = None,
        output_dir: str = "output",
    ) -> Dict[str, str]:
        """Generate all visualization files."""
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        try:
            path = os.path.join(output_dir, "data_overview.png")
            self._create_overview(path)
            generated["data_overview"] = path
        except Exception as e:
            logger.warning("Could not create overview: %s", e)

        try:
            path = os.path.join(output_dir, "findings_summary.png")
            self._create_findings_summary(findings, path)
            generated["findings_summary"] = path
        except Exception as e:
            logger.warning("Could not create findings summary: %s", e)

        if health_score:
            try:
                path = os.path.join(output_dir, "health_score.png")
                self._create_health_dashboard(health_score, path)
                generated["health_score"] = path
            except Exception as e:
                logger.warning("Could not create health dashboard: %s", e)

        try:
            path = os.path.join(output_dir, "missing_data.png")
            self._create_missing_chart(path)
            generated["missing_data"] = path
        except Exception as e:
            logger.warning("Could not create missing chart: %s", e)

        return generated

    def _create_overview(self, path: str):
        """Create data overview dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("DATA OVERVIEW DASHBOARD", fontsize=16, fontweight="bold", y=1.02)

        # Data Types
        ax = axes[0, 0]
        dtype_counts = self.data.dtypes.value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))
        ax.pie(dtype_counts.values, labels=[str(d) for d in dtype_counts.index],
               autopct="%1.0f%%", colors=colors, startangle=90)
        ax.set_title("Data Types", fontweight="bold")

        # Missing Data
        ax = axes[0, 1]
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100)
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
        if len(missing_pct) > 0:
            ax.barh(range(len(missing_pct)), missing_pct.values, color=COLORS["critical"])
            ax.set_yticks(range(len(missing_pct)))
            ax.set_yticklabels([str(c)[:15] for c in missing_pct.index])
            ax.set_xlabel("Missing %")
            ax.set_title("Top Missing Columns", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No Missing Data", ha="center", va="center", fontsize=14)
            ax.set_title("Missing Data", fontweight="bold")
            ax.axis("off")

        # First Numeric Distribution
        ax = axes[0, 2]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            vals = self.data[col].dropna()
            ax.hist(vals, bins=30, color=COLORS["primary"], edgecolor="white", alpha=0.7)
            ax.axvline(vals.mean(), color=COLORS["critical"], linestyle="--", label=f"Mean: {vals.mean():.1f}")
            ax.axvline(vals.median(), color=COLORS["success"], linestyle="--", label=f"Median: {vals.median():.1f}")
            ax.legend(fontsize=8)
            ax.set_title(f"Distribution: {col[:20]}", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No Numeric Data", ha="center", va="center")
            ax.axis("off")

        # Correlation
        ax = axes[1, 0]
        num_cols = self.data.select_dtypes(include=[np.number]).columns[:8]
        if len(num_cols) >= 2:
            corr = self.data[num_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                       annot=True, fmt=".2f", ax=ax, cbar_kws={"shrink": 0.5})
            ax.set_title("Correlation Matrix", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Insufficient numeric columns", ha="center", va="center")
            ax.axis("off")

        # Top Categorical
        ax = axes[1, 1]
        cat_cols = self.data.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            col = cat_cols[0]
            top = self.data[col].value_counts().head(8)
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top)))
            ax.barh(range(len(top)), top.values[::-1], color=colors)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([str(v)[:15] for v in top.index[::-1]])
            ax.set_title(f"Top Values: {col[:20]}", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No Categorical Data", ha="center", va="center")
            ax.axis("off")

        # Quality Metrics
        ax = axes[1, 2]
        ax.axis("off")
        total_cells = self.data.size
        missing_pct_overall = self.data.isnull().sum().sum() / total_cells * 100 if total_cells > 0 else 0
        dup_pct = self.data.duplicated().mean() * 100
        complete_pct = (len(self.data) - self.data.isnull().any(axis=1).sum()) / len(self.data) * 100

        metrics = f"""DATA QUALITY METRICS
━━━━━━━━━━━━━━━━━━━━
📊 Rows:            {len(self.data):,}
📋 Columns:         {len(self.data.columns)}
📦 Total Cells:     {total_cells:,}

🕳️ Missing:         {missing_pct_overall:.1f}%
📑 Duplicates:      {dup_pct:.1f}%
✅ Complete Rows:   {complete_pct:.1f}%"""

        ax.text(0.1, 0.9, metrics, transform=ax.transAxes, fontsize=11,
               va="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    def _create_findings_summary(self, findings: List[Finding], path: str):
        """Create a findings severity summary chart."""
        severity_counts = {}
        for f in findings:
            sev = f.severity.value if isinstance(f.severity, Severity) else f.severity
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("FINDINGS SUMMARY", fontsize=14, fontweight="bold")

        # Severity distribution
        severities = ["critical", "high", "medium", "low", "info"]
        counts = [severity_counts.get(s, 0) for s in severities]
        colors = [SEVERITY_COLORS[s] for s in severities]

        ax1.barh(severities, counts, color=colors)
        ax1.set_xlabel("Count")
        ax1.set_title("Findings by Severity", fontweight="bold")

        for i, c in enumerate(counts):
            if c > 0:
                ax1.text(c + 0.1, i, str(c), va="center", fontweight="bold")

        # Category distribution
        cat_counts = {}
        for f in findings:
            cat_counts[f.category] = cat_counts.get(f.category, 0) + 1

        if cat_counts:
            cats = sorted(cat_counts.keys())
            ax2.barh(cats, [cat_counts[c] for c in cats], color=COLORS["primary"])
            ax2.set_xlabel("Count")
            ax2.set_title("Findings by Category", fontweight="bold")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    def _create_health_dashboard(self, health_score: HealthScore, path: str):
        """Create health score dashboard with gauges."""
        components = health_score.components
        n = len(components)
        cols = min(4, n + 1)
        rows = (n + 1 + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle("DATASET HEALTH SCORE", fontsize=16, fontweight="bold")

        if n == 0:
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            return

        flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Overall gauge
        self._draw_gauge(flat_axes[0], health_score.overall, "Overall Health", health_score.verdict)

        # Component gauges
        for i, (name, comp) in enumerate(components.items()):
            if i + 1 < len(flat_axes):
                self._draw_gauge(flat_axes[i + 1], comp.score, comp.name, f"{comp.findings_count} findings")

        # Hide unused axes
        for j in range(len(components) + 1, len(flat_axes)):
            flat_axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    def _create_missing_chart(self, path: str):
        """Create missing data heatmap."""
        missing_matrix = self.data.isnull()
        cols_with_missing = [c for c in self.data.columns if missing_matrix[c].any()]

        if not cols_with_missing:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("MISSING DATA PATTERN", fontsize=14, fontweight="bold")

        sample_size = min(200, len(self.data))
        indices = np.linspace(0, len(self.data) - 1, sample_size, dtype=int)

        sns.heatmap(
            missing_matrix.iloc[indices][cols_with_missing[:20]],
            cbar=True, cmap="YlOrRd", ax=ax, yticklabels=False,
        )
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows (sampled)")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    @staticmethod
    def _draw_gauge(ax, value: float, title: str, subtitle: str = ""):
        """Draw a gauge chart."""
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        if value >= 70:
            color = COLORS["success"]
        elif value >= 40:
            color = COLORS["medium"]
        else:
            color = COLORS["critical"]

        theta = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", linewidth=20, solid_capstyle="round")

        value_angle = np.pi * (min(value, 100) / 100)
        theta_val = np.linspace(0, value_angle, 50)
        ax.plot(np.cos(np.pi - theta_val), np.sin(np.pi - theta_val),
               color=color, linewidth=20, solid_capstyle="round")

        ax.text(0, 0.4, f"{value:.0f}", fontsize=24, fontweight="bold",
               ha="center", va="center", color=color)
        ax.text(0, -0.1, title, fontsize=10, fontweight="bold",
               ha="center", va="center", color=COLORS["primary"])
        if subtitle:
            ax.text(0, -0.25, subtitle, fontsize=8, ha="center", va="center", color=COLORS["info"])
