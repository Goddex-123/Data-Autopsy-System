"""
HTML Report Renderer

Professional HTML report with modern CSS, responsive layout,
and evidence-based findings presentation.
"""

from datetime import datetime
from typing import Dict, List

from ..core.models import Finding, HealthScore, DatasetProfile, Severity
from ..core.scoring import ScoringEngine


def render_html_report(
    findings: List[Finding],
    health_score: HealthScore,
    profile: DatasetProfile,
    fingerprint: dict,
    source_path: str,
    timestamp: datetime,
) -> str:
    """Render the complete HTML report."""
    severity_counts = ScoringEngine.count_by_severity(findings)

    score_cards = _render_score_cards(health_score)
    findings_html = _render_findings(findings)
    profile_html = _render_profile(profile)
    fingerprint_html = _render_fingerprint(fingerprint)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Autopsy Report</title>
    <style>
        :root {{
            --primary: #1a1a2e;
            --secondary: #16213e;
            --accent: #0f3460;
            --highlight: #e94560;
            --success: #27AE60;
            --warning: #F39C12;
            --danger: #E74C3C;
            --info: #3498DB;
            --text: #2c3e50;
            --bg: #f8f9fa;
            --card: #ffffff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white; padding: 32px; border-radius: 12px;
            margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        header h1 {{ font-size: 2em; margin-bottom: 8px; }}
        .meta {{ opacity: 0.85; font-size: 0.9em; }}
        .score-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px; margin-bottom: 24px;
        }}
        .score-card {{
            background: var(--card); border-radius: 10px; padding: 20px;
            text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-top: 4px solid var(--info); transition: transform 0.2s;
        }}
        .score-card:hover {{ transform: translateY(-3px); }}
        .score-val {{ font-size: 2.2em; font-weight: bold; margin: 8px 0; }}
        .score-label {{ font-size: 0.8em; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
        .s-good {{ color: var(--success); border-color: var(--success); }}
        .s-warn {{ color: var(--warning); border-color: var(--warning); }}
        .s-bad {{ color: var(--danger); border-color: var(--danger); }}
        section {{
            background: var(--card); border-radius: 10px; padding: 24px;
            margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        section h2 {{
            color: var(--primary); border-bottom: 2px solid var(--info);
            padding-bottom: 8px; margin-bottom: 16px; font-size: 1.3em;
        }}
        .finding {{
            background: #f8f9fa; border-left: 4px solid var(--info);
            padding: 16px; margin: 12px 0; border-radius: 0 6px 6px 0;
        }}
        .finding.sev-critical {{ border-color: var(--danger); background: #fdf3f2; }}
        .finding.sev-high {{ border-color: #E67E22; background: #fef6f0; }}
        .finding.sev-medium {{ border-color: var(--warning); background: #fefcf4; }}
        .finding.sev-low {{ border-color: var(--info); }}
        .finding h3 {{ margin-bottom: 6px; font-size: 1em; }}
        .finding .desc {{ margin: 8px 0; }}
        .finding .rec {{ color: #555; font-style: italic; margin-top: 8px; }}
        .finding .lim {{ color: #888; font-size: 0.85em; margin-top: 4px; }}
        .badge {{
            display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.75em; font-weight: 600; text-transform: uppercase;
        }}
        .badge-critical {{ background: #fde8e8; color: var(--danger); }}
        .badge-high {{ background: #fef0e4; color: #E67E22; }}
        .badge-medium {{ background: #fef9e4; color: #856404; }}
        .badge-low {{ background: #e3f2fd; color: #1565c0; }}
        .badge-info {{ background: #f0f0f0; color: #666; }}
        .summary-box {{
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            padding: 20px; border-radius: 8px; text-align: center;
            margin: 16px 0;
        }}
        .evidence {{ background: #2C3E50; color: #ecf0f1; padding: 12px;
            border-radius: 6px; font-family: monospace; font-size: 0.85em;
            overflow-x: auto; margin: 8px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        th, td {{ padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }}
        th {{ background: var(--primary); color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Data Autopsy Report</h1>
            <div class="meta">
                <p><strong>Source:</strong> {source_path}</p>
                <p><strong>Analysis Date:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Rows:</strong> {profile.row_count:,} | <strong>Columns:</strong> {profile.column_count}</p>
            </div>
        </header>

        {score_cards}

        <section>
            <h2>📊 Executive Summary</h2>
            <div class="summary-box">
                <h2>{health_score.verdict_emoji} {health_score.verdict}</h2>
                <p>Overall Health Score: <strong>{health_score.overall:.0f}/100</strong></p>
                <p>Total Findings: <strong>{len(findings)}</strong>
                   (🔴 {severity_counts.get('critical',0)} |
                    🟠 {severity_counts.get('high',0)} |
                    🟡 {severity_counts.get('medium',0)} |
                    🔵 {severity_counts.get('low',0)} |
                    ℹ️ {severity_counts.get('info',0)})</p>
            </div>
        </section>

        {profile_html}
        {findings_html}
        {fingerprint_html}

        <footer>
            <p>Generated by Data Autopsy System v2.0</p>
            <p>Report ID: {timestamp.strftime('%Y%m%d%H%M%S')}</p>
        </footer>
    </div>
</body>
</html>"""


def _render_score_cards(health_score: HealthScore) -> str:
    cards = []
    for name, comp in health_score.components.items():
        cls = "s-good" if comp.score >= 70 else "s-warn" if comp.score >= 40 else "s-bad"
        cards.append(f"""
        <div class="score-card {cls}">
            <div class="score-label">{comp.name}</div>
            <div class="score-val">{comp.score:.0f}</div>
            <div class="score-label">{comp.findings_count} findings</div>
        </div>""")
    return f'<div class="score-grid">{"".join(cards)}</div>'


def _render_findings(findings: List[Finding]) -> str:
    categories = sorted(set(f.category for f in findings))
    sections = []
    cat_emojis = {
        "data_quality": "📋", "missing_data": "🕳️", "anomaly": "🚨",
        "bias": "⚖️", "privacy": "🔒", "ml_audit": "🤖",
        "drift": "📈", "robustness": "🧪", "provenance": "📑",
    }

    for cat in categories:
        cat_findings = sorted(
            [f for f in findings if f.category == cat],
            key=lambda x: x.severity.numeric if hasattr(x.severity, 'numeric') else 0,
            reverse=True,
        )
        emoji = cat_emojis.get(cat, "📌")
        items = []
        for f in cat_findings:
            sev_val = f.severity.value if isinstance(f.severity, Severity) else f.severity
            badge_cls = f"badge-{sev_val}"
            rec_html = f'<div class="rec">💡 {f.recommendation}</div>' if f.recommendation else ""
            lim_html = f'<div class="lim">⚠️ {f.limitations}</div>' if f.limitations else ""

            items.append(f"""
            <div class="finding sev-{sev_val}">
                <h3><span class="badge {badge_cls}">{sev_val}</span> {f.title}</h3>
                <div class="desc">{f.description}</div>
                {rec_html}
                {lim_html}
            </div>""")

        sections.append(f"""
        <section>
            <h2>{emoji} {cat.replace('_', ' ').title()}</h2>
            {"".join(items)}
        </section>""")

    return "".join(sections)


def _render_profile(profile: DatasetProfile) -> str:
    return f"""
    <section>
        <h2>📊 Dataset Profile</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Rows</td><td>{profile.row_count:,}</td></tr>
            <tr><td>Columns</td><td>{profile.column_count}</td></tr>
            <tr><td>Total Cells</td><td>{profile.total_cells:,}</td></tr>
            <tr><td>Missing Cells</td><td>{profile.missing_cells:,} ({profile.missing_percentage:.1f}%)</td></tr>
            <tr><td>Duplicate Rows</td><td>{profile.duplicate_rows:,} ({profile.duplicate_percentage:.1f}%)</td></tr>
            <tr><td>Memory Usage</td><td>{profile.memory_usage_mb:.2f} MB</td></tr>
        </table>
    </section>"""


def _render_fingerprint(fingerprint: dict) -> str:
    system = fingerprint.get("system", {})
    return f"""
    <section>
        <h2>🔐 Provenance</h2>
        <div class="evidence">
Content SHA-256: {fingerprint.get('content_sha256', 'N/A')}
Schema SHA-256:  {fingerprint.get('schema_sha256', 'N/A')}
Rows: {fingerprint.get('row_count', 'N/A')}  |  Columns: {fingerprint.get('column_count', 'N/A')}
Autopsy Version: {system.get('autopsy_version', 'N/A')}
Python: {system.get('python_version', 'N/A')[:20]}
        </div>
    </section>"""
