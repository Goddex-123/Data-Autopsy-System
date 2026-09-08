#!/usr/bin/env python3
"""
Data Autopsy System — Streamlit Web Interface (v2.0)

Thin orchestrator that delegates to UI components and the analysis engine.
"""

import streamlit as st
import pandas as pd
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autopsy import DataAutopsy
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.components.score_cards import render_health_overview
from ui.components.findings_display import render_findings_by_category, render_severity_summary

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Data Autopsy System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .main-header h1 { color: #00d4ff; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .main-header p { color: #a0a0a0; font-size: 1.1rem; }
    .score-card {
        background: linear-gradient(135deg, #2C3E50 0%, #3d5a73 100%);
        border-radius: 10px; padding: 1.2rem; text-align: center;
        color: white; margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .score-card h3 { font-size: 2.2rem; margin: 0; }
    .score-card p { margin: 0.3rem 0 0 0; opacity: 0.8; font-size: 0.85rem; }
    .score-good { border-left: 5px solid #27AE60; }
    .score-warning { border-left: 5px solid #F39C12; }
    .score-danger { border-left: 5px solid #E74C3C; }
    .stProgress > div > div > div > div { background-color: #00d4ff; }
</style>
""", unsafe_allow_html=True)


# ── Main Application ─────────────────────────────────────────────────
def main():
    render_header()

    uploaded_file, options = render_sidebar()

    if uploaded_file is None:
        _render_welcome()
        return

    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{uploaded_file.name}** — {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return

    # Data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # Run Analysis
    if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
        output_dir = tempfile.mkdtemp()
        progress = st.progress(0)
        status = st.empty()

        try:
            status.text("🔬 Initializing analysis engine...")
            progress.progress(10)

            autopsy = DataAutopsy(df)

            if options["quick_scan"]:
                status.text("🔍 Running quick scan...")
                progress.progress(50)
                results = autopsy.quick_scan()
                progress.progress(100)
                status.text("✅ Quick scan complete!")
                _render_quick_results(results)
            else:
                status.text("🔬 Running full investigation...")
                progress.progress(30)

                report = autopsy.investigate(
                    output_dir=output_dir,
                    target_column=options.get("target_column"),
                    generate_visuals=options["generate_visuals"],
                )

                progress.progress(90)
                status.text("📊 Building reports...")

                # Get results
                findings_data = autopsy.findings
                all_findings = findings_data.get("all_findings", [])
                health = findings_data.get("health_score", {})

                progress.progress(100)
                status.text("✅ Investigation complete!")

                # Render results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                # Health overview
                render_health_overview(health)

                # Severity summary
                severity_counts = {}
                for f in all_findings:
                    sev = f.get("severity", "info")
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                render_severity_summary(severity_counts)

                # Detailed findings
                st.markdown("---")
                st.markdown("## 🔍 Detailed Findings")
                render_findings_by_category(all_findings)

                # Visualizations
                if options["generate_visuals"]:
                    viz = findings_data.get("visualizations", {})
                    _render_visualizations(viz)

                # Download reports
                _render_downloads(report, output_dir)

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_welcome():
    """Render the welcome screen."""
    st.markdown("### 👋 Welcome to Data Autopsy v2.0")
    st.markdown("Upload a CSV file using the sidebar to begin analysis.")

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("📋 Data Quality", "Completeness, validity, consistency, duplicates"),
        ("🚨 Anomalies", "Benford's Law, outliers, Isolation Forest"),
        ("⚖️ Bias Detection", "Imbalance, population comparison, skewness"),
        ("🔒 Privacy", "PII detection, email, phone, IP patterns"),
    ]
    for col, (title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"#### {title}")
            st.markdown(desc)

    col5, col6, col7, col8 = st.columns(4)
    features2 = [
        ("🕳️ Missing Data", "Patterns, mechanisms, sentinel values"),
        ("🤖 ML Audit", "Target leakage, feature redundancy"),
        ("🧪 Robustness", "Subgroups, Simpson's Paradox"),
        ("📊 Reports", "HTML, JSON, Markdown exports"),
    ]
    for col, (title, desc) in zip([col5, col6, col7, col8], features2):
        with col:
            st.markdown(f"#### {title}")
            st.markdown(desc)


def _render_quick_results(results: dict):
    """Render quick scan results."""
    st.markdown("### 📋 Quick Scan Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{results['rows']:,}")
        st.metric("Missing %", f"{results['missing_percentage']:.1f}%")
    with col2:
        st.metric("Columns", results['columns'])
        st.metric("Duplicates", results['duplicate_rows'])
    with col3:
        st.metric("Memory", f"{results['memory_usage_mb']:.1f} MB")
        st.metric("Numeric Cols", results['numeric_columns'])


def _render_visualizations(viz: dict):
    """Render generated visualizations."""
    viz_files = {k: v for k, v in viz.items() if v and os.path.exists(str(v))}
    if not viz_files:
        return

    st.markdown("---")
    st.markdown("## 📊 Visual Evidence")

    cols = st.columns(2)
    for i, (name, path) in enumerate(viz_files.items()):
        with cols[i % 2]:
            st.image(path, caption=name.replace("_", " ").title())


def _render_downloads(report, output_dir: str):
    """Render report download buttons."""
    st.markdown("---")
    st.markdown("## 📥 Download Reports")

    col1, col2, col3 = st.columns(3)

    try:
        html_path = report.save("report.html")
        md_path = report.save("report.md")
        json_path = report.save("report.json")

        with col1:
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    st.download_button("📄 HTML Report", f.read(),
                                      file_name="autopsy_report.html", mime="text/html")

        with col2:
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    st.download_button("📝 Markdown Report", f.read(),
                                      file_name="autopsy_report.md", mime="text/markdown")

        with col3:
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    st.download_button("📊 JSON Report", f.read(),
                                      file_name="autopsy_report.json", mime="application/json")
    except Exception as e:
        st.warning(f"Could not generate some reports: {e}")


if __name__ == "__main__":
    main()
