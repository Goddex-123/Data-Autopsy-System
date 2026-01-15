"""
Forensic Report Generator

Generates executive-style forensic reports with evidence and recommendations.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import json


class ForensicReport:
    """
    Executive-style forensic report generator.
    """
    
    def __init__(self, findings: dict, source_path: str, output_dir: str = "output"):
        """
        Initialize report generator.
        
        Args:
            findings: All forensic findings from analyzers
            source_path: Original data source path
            output_dir: Directory for output files
        """
        self.findings = findings
        self.source_path = source_path
        self.output_dir = output_dir
        self.timestamp = datetime.now()
        
    def save(self, filename: str = "forensic_report.html"):
        """
        Save the complete forensic report.
        
        Args:
            filename: Output filename (supports .html, .md, .json)
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if filename.endswith('.html'):
            content = self._generate_html_report()
        elif filename.endswith('.md'):
            content = self._generate_markdown_report()
        elif filename.endswith('.json'):
            content = self._generate_json_report()
        else:
            content = self._generate_markdown_report()
            filepath += '.md'
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"📋 Report saved to: {filepath}")
        return filepath
    
    def _generate_html_report(self) -> str:
        """Generate HTML formatted report."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Analysis Report</title>
    <style>
        :root {{
            --primary: #2C3E50;
            --warning: #E74C3C;
            --success: #27AE60;
            --info: #3498DB;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg);
            color: var(--primary);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary), #34495e);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .meta {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        
        .score-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .score-card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .score-card:hover {{
            transform: translateY(-5px);
        }}
        
        .score-value {{
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .score-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .score-good {{ color: var(--success); }}
        .score-warning {{ color: #f39c12; }}
        .score-danger {{ color: var(--warning); }}
        
        section {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        section h2 {{
            color: var(--primary);
            border-bottom: 2px solid var(--info);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .finding {{
            background: #f8f9fa;
            border-left: 4px solid var(--info);
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}
        
        .finding.warning {{
            border-left-color: var(--warning);
            background: #fdf3f2;
        }}
        
        .finding.success {{
            border-left-color: var(--success);
            background: #f0fdf4;
        }}
        
        .finding h3 {{
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .evidence {{
            background: #2C3E50;
            color: #fff;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            margin: 10px 0;
        }}
        
        .recommendations {{
            background: linear-gradient(135deg, #e8f4f8, #f0f8ff);
            padding: 20px;
            border-radius: 5px;
        }}
        
        .recommendations h3 {{
            color: var(--info);
            margin-bottom: 15px;
        }}
        
        .recommendations ul {{
            padding-left: 20px;
        }}
        
        .recommendations li {{
            margin-bottom: 10px;
        }}
        
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .visualization img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .badge-danger {{ background: #fde8e8; color: var(--warning); }}
        .badge-warning {{ background: #fef3cd; color: #856404; }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-info {{ background: #cce5ff; color: #004085; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Forensic Analysis Report</h1>
            <div class="meta">
                <p><strong>Source:</strong> {self.source_path}</p>
                <p><strong>Analysis Date:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </header>
        
        {self._generate_score_cards_html()}
        
        {self._generate_executive_summary_html()}
        
        {self._generate_provenance_section_html()}
        
        {self._generate_bias_section_html()}
        
        {self._generate_anomaly_section_html()}
        
        {self._generate_missing_section_html()}
        
        {self._generate_robustness_section_html()}
        
        {self._generate_recommendations_html()}
        
        <footer>
            <p>Generated by Data Autopsy System | Forensic Dataset Analysis</p>
            <p>Report ID: {self.timestamp.strftime('%Y%m%d%H%M%S')}</p>
        </footer>
    </div>
</body>
</html>"""
    
    def _generate_score_cards_html(self) -> str:
        """Generate score cards HTML."""
        provenance = self.findings.get('provenance', {})
        bias = self.findings.get('bias', {})
        anomalies = self.findings.get('anomalies', {})
        missing = self.findings.get('missing', {})
        robustness = self.findings.get('robustness', {})
        
        scores = [
            ('Credibility', provenance.get('credibility_score', 0), True),
            ('Bias Score', bias.get('overall_bias_score', 0), False),
            ('Anomaly Score', anomalies.get('overall_anomaly_score', 0), False),
            ('Missing Score', missing.get('overall_missing_score', 0), False),
            ('Robustness', robustness.get('overall_robustness_score', 100), True),
        ]
        
        cards = []
        for label, score, higher_is_better in scores:
            if higher_is_better:
                css_class = 'score-good' if score >= 70 else 'score-warning' if score >= 40 else 'score-danger'
            else:
                css_class = 'score-good' if score <= 30 else 'score-warning' if score <= 60 else 'score-danger'
                
            cards.append(f"""
            <div class="score-card">
                <div class="score-label">{label}</div>
                <div class="score-value {css_class}">{score:.0f}</div>
                <div class="score-label">/ 100</div>
            </div>
            """)
            
        return f'<div class="score-cards">{"".join(cards)}</div>'
    
    def _generate_executive_summary_html(self) -> str:
        """Generate executive summary section."""
        provenance = self.findings.get('provenance', {})
        bias = self.findings.get('bias', {})
        anomalies = self.findings.get('anomalies', {})
        missing = self.findings.get('missing', {})
        robustness = self.findings.get('robustness', {})
        
        # Calculate overall health
        overall = (
            provenance.get('credibility_score', 0) + 
            robustness.get('overall_robustness_score', 100) + 
            (100 - bias.get('overall_bias_score', 0)) + 
            (100 - anomalies.get('overall_anomaly_score', 0)) + 
            (100 - missing.get('overall_missing_score', 0))
        ) / 5
        
        if overall >= 70:
            verdict = "HEALTHY"
            verdict_class = "success"
            verdict_emoji = "✅"
        elif overall >= 50:
            verdict = "MODERATE CONCERNS"
            verdict_class = "warning"
            verdict_emoji = "⚠️"
        else:
            verdict = "SIGNIFICANT ISSUES"
            verdict_class = "danger"
            verdict_emoji = "🚨"
            
        # Collect all warnings
        all_warnings = []
        all_warnings.extend(provenance.get('concerns', []))
        all_warnings.extend(bias.get('critical_warnings', []))
        all_warnings.extend(anomalies.get('red_flags', []))
        all_warnings.extend(missing.get('concerns', []))
        all_warnings.extend(robustness.get('fragility_warnings', []))
        
        warnings_html = ""
        if all_warnings:
            warnings_html = "<h3>Key Findings</h3><ul>"
            for w in all_warnings[:10]:
                warnings_html += f"<li>{w}</li>"
            warnings_html += "</ul>"
        else:
            warnings_html = "<div class='finding success'><h3>✅ No Critical Issues</h3><p>The dataset passed all major forensic checks.</p></div>"
            
        return f"""
        <section>
            <h2>Executive Summary</h2>
            <div class="finding {verdict_class}">
                <h3>{verdict_emoji} Overall Assessment: {verdict}</h3>
                <p>Overall Data Health Score: <strong>{overall:.0f}/100</strong></p>
            </div>
            {warnings_html}
        </section>
        """
    
    def _generate_provenance_section_html(self) -> str:
        """Generate provenance section."""
        prov = self.findings.get('provenance', {})
        meta = prov.get('metadata', {})
        
        return f"""
        <section>
            <h2>📋 Data Provenance</h2>
            <div class="finding">
                <h3>Dataset Overview</h3>
                <div class="evidence">
Source: {meta.get('source', 'Unknown')}
Rows: {meta.get('rows', 'N/A'):,}
Columns: {meta.get('columns', 'N/A')}
Memory Usage: {meta.get('memory_usage_mb', 0):.2f} MB
                </div>
            </div>
            <div class="finding">
                <h3>Credibility Assessment</h3>
                <p>Credibility Score: <strong>{prov.get('credibility_score', 0):.0f}/100</strong></p>
                <ul>
                    {''.join(f"<li>{c}</li>" for c in prov.get('concerns', ['No concerns']))}
                </ul>
            </div>
        </section>
        """
    
    def _generate_bias_section_html(self) -> str:
        """Generate bias section."""
        bias = self.findings.get('bias', {})
        score = bias.get('overall_bias_score', 0)
        
        class_txt = 'success' if score <= 30 else 'warning' if score <= 60 else 'danger'
        
        dist = bias.get('distribution_bias', {})
        sampling = bias.get('sampling_bias', {})
        
        findings_html = ""
        
        skewed = dist.get('skewed_columns', [])
        if skewed:
            findings_html += f"<div class='finding warning'><h3>Skewed Distributions</h3><p>{len(skewed)} column(s) show significant skewness</p></div>"
            
        imbalance = sampling.get('class_imbalance', [])
        if imbalance:
            findings_html += f"<div class='finding warning'><h3>Class Imbalance</h3><p>{len(imbalance)} categorical variable(s) show imbalance</p></div>"
            
        if not skewed and not imbalance:
            findings_html = "<div class='finding success'><h3>✅ No Major Bias Detected</h3></div>"
            
        return f"""
        <section>
            <h2>⚖️ Bias Detection</h2>
            <div class="finding">
                <p>Overall Bias Score: <span class="badge badge-{class_txt}">{score:.0f}/100</span></p>
            </div>
            {findings_html}
        </section>
        """
    
    def _generate_anomaly_section_html(self) -> str:
        """Generate anomaly section."""
        anom = self.findings.get('anomalies', {})
        score = anom.get('overall_anomaly_score', 0)
        
        class_txt = 'success' if score <= 30 else 'warning' if score <= 60 else 'danger'
        
        red_flags = anom.get('red_flags', [])
        
        if red_flags:
            flags_html = "<ul>" + "".join(f"<li>🚨 {f}</li>" for f in red_flags) + "</ul>"
        else:
            flags_html = "<div class='finding success'><h3>✅ No Major Anomalies</h3></div>"
            
        return f"""
        <section>
            <h2>🚨 Anomaly Detection</h2>
            <div class="finding">
                <p>Overall Anomaly Score: <span class="badge badge-{class_txt}">{score:.0f}/100</span></p>
            </div>
            {flags_html}
        </section>
        """
    
    def _generate_missing_section_html(self) -> str:
        """Generate missing data section."""
        missing = self.findings.get('missing', {})
        overview = missing.get('overview', {})
        score = missing.get('overall_missing_score', 0)
        
        class_txt = 'success' if score <= 30 else 'warning' if score <= 60 else 'danger'
        
        return f"""
        <section>
            <h2>🕳️ Missing Data Analysis</h2>
            <div class="finding">
                <p>Missing Data Score: <span class="badge badge-{class_txt}">{score:.0f}/100</span></p>
                <p>Overall Missing: {overview.get('missing_percentage', 0):.1f}%</p>
                <p>Complete Rows: {overview.get('complete_percentage', 100):.1f}%</p>
            </div>
        </section>
        """
    
    def _generate_robustness_section_html(self) -> str:
        """Generate robustness section."""
        robust = self.findings.get('robustness', {})
        score = robust.get('overall_robustness_score', 100)
        
        class_txt = 'success' if score >= 70 else 'warning' if score >= 40 else 'danger'
        
        warnings = robust.get('fragility_warnings', [])
        
        if warnings:
            warn_html = "<ul>" + "".join(f"<li>⚠️ {w}</li>" for w in warnings) + "</ul>"
        else:
            warn_html = "<div class='finding success'><h3>✅ Conclusions Appear Robust</h3></div>"
            
        return f"""
        <section>
            <h2>🧪 Robustness Analysis</h2>
            <div class="finding">
                <p>Robustness Score: <span class="badge badge-{class_txt}">{score:.0f}/100</span></p>
            </div>
            {warn_html}
        </section>
        """
    
    def _generate_recommendations_html(self) -> str:
        """Generate recommendations section."""
        recommendations = self._generate_recommendations()
        
        items = "".join(f"<li>{r}</li>" for r in recommendations)
        
        return f"""
        <section class="recommendations">
            <h2>📌 Recommendations</h2>
            <ul>{items}</ul>
        </section>
        """
    
    def _generate_recommendations(self) -> List[str]:
        """Generate list of recommendations based on findings."""
        recommendations = []
        
        bias = self.findings.get('bias', {})
        if bias.get('overall_bias_score', 0) > 50:
            recommendations.append("Address class imbalance through resampling or stratified analysis")
            recommendations.append("Document known biases and their potential impact on conclusions")
            
        anomalies = self.findings.get('anomalies', {})
        if anomalies.get('overall_anomaly_score', 0) > 50:
            recommendations.append("Investigate flagged anomalies to determine if data cleaning is needed")
            recommendations.append("Verify data collection methodology for columns failing Benford's Law")
            
        missing = self.findings.get('missing', {})
        if missing.get('overall_missing_score', 0) > 30:
            recommendations.append("Develop imputation strategy for missing data or document exclusions")
            recommendations.append("Investigate whether missing data is random or systematic")
            
        robustness = self.findings.get('robustness', {})
        if robustness.get('overall_robustness_score', 100) < 70:
            recommendations.append("Use robust statistical methods less sensitive to outliers")
            recommendations.append("Report subgroup analyses alongside overall findings")
            
        if not recommendations:
            recommendations.append("Dataset appears suitable for analysis with standard precautions")
            recommendations.append("Document analysis methodology for reproducibility")
            
        return recommendations
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown formatted report."""
        provenance = self.findings.get('provenance', {})
        bias = self.findings.get('bias', {})
        anomalies = self.findings.get('anomalies', {})
        missing = self.findings.get('missing', {})
        robustness = self.findings.get('robustness', {})
        
        recommendations = self._generate_recommendations()
        
        return f"""# 🔬 Forensic Analysis Report

**Source:** {self.source_path}  
**Analysis Date:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Score | Status |
|--------|-------|--------|
| Credibility | {provenance.get('credibility_score', 0):.0f}/100 | {'✅' if provenance.get('credibility_score', 0) >= 70 else '⚠️'} |
| Bias | {bias.get('overall_bias_score', 0):.0f}/100 | {'✅' if bias.get('overall_bias_score', 0) <= 30 else '⚠️'} |
| Anomalies | {anomalies.get('overall_anomaly_score', 0):.0f}/100 | {'✅' if anomalies.get('overall_anomaly_score', 0) <= 30 else '⚠️'} |
| Missing Data | {missing.get('overall_missing_score', 0):.0f}/100 | {'✅' if missing.get('overall_missing_score', 0) <= 30 else '⚠️'} |
| Robustness | {robustness.get('overall_robustness_score', 100):.0f}/100 | {'✅' if robustness.get('overall_robustness_score', 100) >= 70 else '⚠️'} |

---

## 📋 Data Provenance

- **Rows:** {provenance.get('metadata', {}).get('rows', 'N/A'):,}
- **Columns:** {provenance.get('metadata', {}).get('columns', 'N/A')}
- **Credibility Score:** {provenance.get('credibility_score', 0):.0f}/100

### Concerns
{chr(10).join('- ' + c for c in provenance.get('concerns', ['None']))}

---

## ⚖️ Bias Detection

**Bias Score:** {bias.get('overall_bias_score', 0):.0f}/100

### Critical Warnings
{chr(10).join('- ⚠️ ' + w for w in bias.get('critical_warnings', ['None']))}

---

## 🚨 Anomaly Detection

**Anomaly Score:** {anomalies.get('overall_anomaly_score', 0):.0f}/100

### Red Flags
{chr(10).join('- 🚨 ' + f for f in anomalies.get('red_flags', ['None']))}

---

## 🕳️ Missing Data

**Missing Score:** {missing.get('overall_missing_score', 0):.0f}/100

- Overall Missing: {missing.get('overview', {}).get('missing_percentage', 0):.1f}%
- Complete Rows: {missing.get('overview', {}).get('complete_percentage', 100):.1f}%

### Concerns
{chr(10).join('- ' + c for c in missing.get('concerns', ['None']))}

---

## 🧪 Robustness

**Robustness Score:** {robustness.get('overall_robustness_score', 100):.0f}/100

### Warnings
{chr(10).join('- ' + w for w in robustness.get('fragility_warnings', ['None']))}

---

## 📌 Recommendations

{chr(10).join('1. ' + r for i, r in enumerate(recommendations))}

---

*Generated by Data Autopsy System*
"""
    
    def _generate_json_report(self) -> str:
        """Generate JSON formatted report."""
        report = {
            'metadata': {
                'source': self.source_path,
                'analysis_date': self.timestamp.isoformat(),
                'generator': 'Data Autopsy System'
            },
            'findings': self.findings,
            'recommendations': self._generate_recommendations()
        }
        return json.dumps(report, indent=2, default=str)
    
    def print_summary(self):
        """Print a quick summary to console."""
        provenance = self.findings.get('provenance', {})
        bias = self.findings.get('bias', {})
        anomalies = self.findings.get('anomalies', {})
        missing = self.findings.get('missing', {})
        robustness = self.findings.get('robustness', {})
        
        print("\n" + "=" * 50)
        print("🔬 FORENSIC ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"\nSource: {self.source_path}")
        print(f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "-" * 50)
        print("SCORES:")
        print(f"  Credibility:  {provenance.get('credibility_score', 0):>5.0f}/100")
        print(f"  Bias:         {bias.get('overall_bias_score', 0):>5.0f}/100")
        print(f"  Anomalies:    {anomalies.get('overall_anomaly_score', 0):>5.0f}/100")
        print(f"  Missing:      {missing.get('overall_missing_score', 0):>5.0f}/100")
        print(f"  Robustness:   {robustness.get('overall_robustness_score', 100):>5.0f}/100")
        print("-" * 50 + "\n")
