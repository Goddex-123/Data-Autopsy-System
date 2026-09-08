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
    confidence = finding.get("confidence", 0)
    recommendation = finding.get("recommendation", "")
    limitations = finding.get("limitations", "")
    evidence = finding.get("evidence", {})

    severity_config = {
        "critical": ("🔴", "error"),
        "high": ("🟠", "warning"),
        "medium": ("🟡", "warning"),
        "low": ("🔵", "info"),
        "info": ("ℹ️", "info"),
    }

    emoji, st_type = severity_config.get(severity, ("📌", "info"))

    # Use appropriate streamlit element
    if severity in ("critical", "high"):
        st.error(f"{emoji} **{title}**")
    elif severity == "medium":
        st.warning(f"{emoji} **{title}**")
    else:
        st.info(f"{emoji} **{title}**")

    st.markdown(f"*Severity: {severity.upper()} | Confidence: {confidence:.0%}*")
    st.markdown(description)

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
