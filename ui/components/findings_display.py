"""UI Findings Display Component"""

import streamlit as st
from typing import List


def render_findings_by_category(findings: List[dict]):
    """Render findings organized by category in tabs."""
    categories = sorted(set(f.get("category", "other") for f in findings))

    cat_emojis = {
        "data_quality": "📋", "missing_data": "🕳️", "anomaly": "🚨",
        "bias": "⚖️", "privacy": "🔒", "ml_audit": "🤖",
        "drift": "📈", "robustness": "🧪", "provenance": "📑",
    }

    cat_labels = [
        f"{cat_emojis.get(c, '📌')} {c.replace('_', ' ').title()}"
        for c in categories
    ]

    if not categories:
        st.info("No findings to display.")
        return

    tabs = st.tabs(cat_labels)

    for tab, cat in zip(tabs, categories):
        cat_findings = sorted(
            [f for f in findings if f.get("category") == cat],
            key=lambda x: {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}.get(
                x.get("severity", "info"), 0
            ),
            reverse=True,
        )

        with tab:
            if not cat_findings:
                st.success("✅ No issues detected")
                continue

            for f in cat_findings:
                _render_single_finding(f)


def _render_single_finding(finding: dict):
    """Render a single finding card."""
    severity = finding.get("severity", "info")
    title = finding.get("title", "")
    description = finding.get("description", "")
    evidence_score = finding.get("evidence_score", finding.get("confidence", 0))
    recommendation = finding.get("recommendation", "")
    limitations = finding.get("limitations", "")
    evidence = finding.get("evidence", {})

    severity_colors = {
        "critical": "#E74C3C",
        "high": "#E67E22",
        "medium": "#F39C12",
        "low": "#3498DB",
        "info": "#8a8f98",
    }
    
    emoji = {
        "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "ℹ️",
    }.get(severity, "📌")
    
    color = severity_colors.get(severity, "#8a8f98")

    st.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.02); backdrop-filter: blur(10px); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid rgba(255, 255, 255, 0.05); border-left: 4px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <h3 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1.3rem; font-weight: 500;">{emoji} {title}</h3>
        <div style="font-size: 0.85rem; color: #8a8f98; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">
            SEVERITY: <strong style="color: {color}">{severity.upper()}</strong> &nbsp;|&nbsp; EVIDENCE SCORE: <strong>{evidence_score:.2f}</strong>
        </div>
        <p style="color: #ededed; font-size: 1rem; line-height: 1.5;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

    if recommendation:
        st.markdown(f"💡 **Recommendation:** {recommendation}")

    if limitations:
        st.caption(f"⚠️ *{limitations}*")

    if evidence:
        with st.expander("📊 Evidence Details"):
            for key, value in evidence.items():
                if isinstance(value, dict):
                    st.json(value)
                elif isinstance(value, list):
                    st.json(value)
                else:
                    st.text(f"{key}: {value}")

    st.markdown("---")


def render_severity_summary(severity_counts: dict):
    """Render a quick severity summary."""
    cols = st.columns(5)
    labels = [
        ("🔴 Critical", "critical"),
        ("🟠 High", "high"),
        ("🟡 Medium", "medium"),
        ("🔵 Low", "low"),
        ("ℹ️ Info", "info"),
    ]

    for col, (label, sev) in zip(cols, labels):
        with col:
            count = severity_counts.get(sev, 0)
            st.metric(label, count)
