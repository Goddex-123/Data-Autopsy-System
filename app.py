#!/usr/bin/env python3
"""
Data Autopsy System - Web Interface

Streamlit-based web application for forensic dataset analysis.
Users can upload datasets and receive comprehensive forensic reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import base64
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autopsy import DataAutopsy

# Page configuration
st.set_page_config(
    page_title="Data Autopsy System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark forensic theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #00d4ff;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #a0a0a0;
        font-size: 1.1rem;
    }
    .score-card {
        background: linear-gradient(135deg, #2C3E50 0%, #3d5a73 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .score-card h3 {
        font-size: 2.5rem;
        margin: 0;
    }
    .score-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
    }
    .score-good { border-left: 5px solid #27AE60; }
    .score-warning { border-left: 5px solid #F39C12; }
    .score-danger { border-left: 5px solid #E74C3C; }
    
    .finding-card {
        background: #f8f9fa;
        border-left: 4px solid #3498DB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .finding-card.warning {
        border-left-color: #E74C3C;
        background: #fdf3f2;
    }
    .finding-card.success {
        border-left-color: #27AE60;
        background: #f0fdf4;
    }
    
    .stProgress > div > div > div > div {
        background-color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)


def get_score_class(score, higher_is_better=True):
    """Get CSS class based on score."""
    if higher_is_better:
        if score >= 70:
            return "score-good"
        elif score >= 40:
            return "score-warning"
        return "score-danger"
    else:
        if score <= 30:
            return "score-good"
        elif score <= 60:
            return "score-warning"
        return "score-danger"


def render_score_card(title, score, higher_is_better=True):
    """Render a score card."""
    css_class = get_score_class(score, higher_is_better)
    st.markdown(f"""
    <div class="score-card {css_class}">
        <h3>{score:.0f}</h3>
        <p>{title}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Data Autopsy System</h1>
        <p>Forensic Analysis of Datasets | Uncover Bias, Manipulation, and Hidden Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to perform forensic analysis"
        )
        
        st.markdown("---")
        
        st.header("⚙️ Analysis Options")
        run_quick_scan = st.checkbox("Quick Scan Only", value=False, 
                                     help="Run a quick preliminary scan instead of full analysis")
        
        generate_visuals = st.checkbox("Generate Visualizations", value=True,
                                       help="Create visual evidence plots")
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        **Data Autopsy** treats your datasets like crime scenes:
        - 🔍 Who generated it?
        - ⚖️ What bias exists?
        - 🕳️ What is missing on purpose?
        - 🚨 What conclusions are dangerous?
        
        Used in **journalism**, **courts**, and **intelligence**.
        """)
    
    # Main content
    if uploaded_file is None:
        # Welcome screen
        st.markdown("### 👋 Welcome to Data Autopsy")
        st.markdown("""
        Upload a CSV file using the sidebar to begin forensic analysis.
        
        **What we analyze:**
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📋 Provenance
            - Data origin tracking
            - Metadata validation
            - Credibility scoring
            """)
            
        with col2:
            st.markdown("""
            #### ⚖️ Bias Detection
            - Sampling bias
            - Class imbalance
            - Representation issues
            """)
            
        with col3:
            st.markdown("""
            #### 🚨 Anomalies
            - Benford's Law
            - Outlier detection
            - Fabrication signatures
            """)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("""
            #### 🕳️ Missing Data
            - Pattern analysis
            - Sentinel values
            - Coverage gaps
            """)
            
        with col5:
            st.markdown("""
            #### 🧪 Robustness
            - Sensitivity testing
            - Simpson's Paradox
            - Subgroup analysis
            """)
            
        with col6:
            st.markdown("""
            #### 📊 Reports
            - Visual evidence
            - Executive summary
            - Recommendations
            """)
            
    else:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded **{uploaded_file.name}** - {len(df):,} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
        
        # Run Analysis
        if st.button("🔬 Run Forensic Analysis", type="primary", use_container_width=True):
            
            # Create temp output directory
            output_dir = tempfile.mkdtemp()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize autopsy
                status_text.text("🔬 Initializing forensic investigation...")
                progress_bar.progress(10)
                autopsy = DataAutopsy(df)
                
                if run_quick_scan:
                    # Quick scan only
                    status_text.text("🔍 Running quick scan...")
                    progress_bar.progress(50)
                    quick_results = autopsy.quick_scan()
                    progress_bar.progress(100)
                    status_text.text("✅ Quick scan complete!")
                    
                    st.markdown("### 📋 Quick Scan Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", f"{quick_results['rows']:,}")
                        st.metric("Missing %", f"{quick_results['missing_percentage']:.1f}%")
                        st.metric("Numeric Cols", quick_results['numeric_columns'])
                    with col2:
                        st.metric("Columns", quick_results['columns'])
                        st.metric("Duplicate Rows", quick_results['duplicate_rows'])
                        st.metric("Text Cols", quick_results['categorical_columns'])
                        
                else:
                    # Full investigation
                    status_text.text("📋 Analyzing data provenance...")
                    progress_bar.progress(20)
                    
                    # Run investigation
                    report = autopsy.investigate(output_dir=output_dir)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Investigation complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Forensic Analysis Results")
                    
                    # Score cards
                    findings = autopsy.findings
                    provenance = findings.get('provenance', {})
                    bias = findings.get('bias', {})
                    anomalies = findings.get('anomalies', {})
                    missing = findings.get('missing', {})
                    robustness = findings.get('robustness', {})
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        render_score_card("Credibility", provenance.get('credibility_score', 0), True)
                    with col2:
                        render_score_card("Bias Score", bias.get('overall_bias_score', 0), False)
                    with col3:
                        render_score_card("Anomaly Score", anomalies.get('overall_anomaly_score', 0), False)
                    with col4:
                        render_score_card("Missing Score", missing.get('overall_missing_score', 0), False)
                    with col5:
                        render_score_card("Robustness", robustness.get('overall_robustness_score', 100), True)
                    
                    # Overall health
                    overall = (
                        provenance.get('credibility_score', 0) + 
                        robustness.get('overall_robustness_score', 100) + 
                        (100 - bias.get('overall_bias_score', 0)) + 
                        (100 - anomalies.get('overall_anomaly_score', 0)) + 
                        (100 - missing.get('overall_missing_score', 0))
                    ) / 5
                    
                    if overall >= 70:
                        verdict = "✅ HEALTHY"
                        color = "green"
                    elif overall >= 50:
                        verdict = "⚠️ MODERATE CONCERNS"
                        color = "orange"
                    else:
                        verdict = "🚨 SIGNIFICANT ISSUES"
                        color = "red"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f0f0f0; border-radius: 10px; margin: 1rem 0;">
                        <h2 style="color: {color};">{verdict}</h2>
                        <p>Overall Data Health Score: <strong>{overall:.0f}/100</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed findings tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📋 Provenance", "⚖️ Bias", "🚨 Anomalies", "🕳️ Missing", "🧪 Robustness"
                    ])
                    
                    with tab1:
                        st.markdown("### Data Provenance Analysis")
                        meta = provenance.get('metadata', {})
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rows", f"{meta.get('rows', 0):,}")
                            st.metric("Memory Usage", f"{meta.get('memory_usage_mb', 0):.2f} MB")
                        with col2:
                            st.metric("Columns", meta.get('columns', 0))
                            st.metric("Credibility", f"{provenance.get('credibility_score', 0):.0f}/100")
                        
                        concerns = provenance.get('concerns', [])
                        if concerns:
                            st.markdown("**Concerns:**")
                            for c in concerns:
                                st.warning(c)
                    
                    with tab2:
                        st.markdown("### Bias Detection Results")
                        
                        warnings = bias.get('critical_warnings', [])
                        if warnings:
                            for w in warnings:
                                st.warning(f"⚠️ {w}")
                        else:
                            st.success("✅ No critical bias warnings detected")
                        
                        # Show class imbalance
                        sampling = bias.get('sampling_bias', {})
                        if sampling.get('class_imbalance'):
                            st.markdown("**Class Imbalance Detected:**")
                            for item in sampling['class_imbalance'][:5]:
                                st.info(f"Column `{item['column']}` - Imbalance ratio: {item['imbalance_ratio']:.1f}x")
                    
                    with tab3:
                        st.markdown("### Anomaly Detection Results")
                        
                        red_flags = anomalies.get('red_flags', [])
                        if red_flags:
                            for flag in red_flags:
                                st.error(f"🚨 {flag}")
                        else:
                            st.success("✅ No major anomalies detected")
                        
                        # Benford's Law
                        benford = anomalies.get('benfords_law', {})
                        violations = benford.get('violations', [])
                        if violations:
                            st.markdown("**Benford's Law Violations:**")
                            for v in violations[:5]:
                                st.warning(f"Column `{v['column']}` - p-value: {v['p_value']:.6f}")
                    
                    with tab4:
                        st.markdown("### Missing Data Analysis")
                        
                        overview = missing.get('overview', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Missing Cells", f"{overview.get('missing_cells', 0):,}")
                        with col2:
                            st.metric("Missing %", f"{overview.get('missing_percentage', 0):.1f}%")
                        with col3:
                            st.metric("Complete Rows", f"{overview.get('complete_percentage', 100):.1f}%")
                        
                        concerns = missing.get('concerns', [])
                        if concerns:
                            for c in concerns:
                                st.warning(f"🕳️ {c}")
                        
                        # Silent assumptions
                        silent = missing.get('silent_assumptions', {})
                        if silent.get('sentinel_values'):
                            st.markdown("**Sentinel Values Detected (Hidden Missing Data):**")
                            for s in silent['sentinel_values'][:5]:
                                st.info(f"Column `{s['column']}` - Value `{s['value']}` appears {s['count']} times")
                    
                    with tab5:
                        st.markdown("### Robustness Analysis")
                        
                        warnings = robustness.get('fragility_warnings', [])
                        if warnings:
                            for w in warnings:
                                st.warning(f"⚠️ {w}")
                        else:
                            st.success("✅ Conclusions appear robust")
                        
                        # Simpson's Paradox
                        subgroups = robustness.get('subgroup_analysis', {})
                        paradox = subgroups.get('simpson_paradox_candidates', [])
                        if paradox:
                            st.error("🚨 **Simpson's Paradox Detected!** Conclusions may reverse within subgroups.")
                            for p in paradox[:3]:
                                st.info(f"Variables: {p['variables']} | Overall: {p['overall_correlation']:.3f} | Group '{p['group']}': {p['group_correlation']:.3f}")
                    
                    # Visualizations
                    if generate_visuals:
                        st.markdown("---")
                        st.markdown("## 📊 Visual Evidence")
                        
                        viz_files = findings.get('visualizations', {})
                        
                        col1, col2 = st.columns(2)
                        
                        if viz_files.get('data_overview') and os.path.exists(viz_files['data_overview']):
                            with col1:
                                st.image(viz_files['data_overview'], caption="Data Overview Dashboard")
                        
                        if viz_files.get('executive_summary') and os.path.exists(viz_files['executive_summary']):
                            with col2:
                                st.image(viz_files['executive_summary'], caption="Executive Summary")
                        
                        col3, col4 = st.columns(2)
                        
                        if viz_files.get('bias_analysis') and os.path.exists(viz_files['bias_analysis']):
                            with col3:
                                st.image(viz_files['bias_analysis'], caption="Bias Analysis")
                        
                        if viz_files.get('anomaly_detection') and os.path.exists(viz_files['anomaly_detection']):
                            with col4:
                                st.image(viz_files['anomaly_detection'], caption="Anomaly Detection")
                    
                    # Download report
                    st.markdown("---")
                    st.markdown("## 📥 Download Report")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Generate reports
                    html_report = os.path.join(output_dir, 'report.html')
                    md_report = os.path.join(output_dir, 'report.md')
                    json_report = os.path.join(output_dir, 'report.json')
                    
                    report.save('report.html')
                    report.save('report.md')
                    report.save('report.json')
                    
                    with col1:
                        if os.path.exists(html_report):
                            with open(html_report, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.download_button(
                                "📄 Download HTML Report",
                                html_content,
                                file_name="forensic_report.html",
                                mime="text/html"
                            )
                    
                    with col2:
                        if os.path.exists(md_report):
                            with open(md_report, 'r', encoding='utf-8') as f:
                                md_content = f.read()
                            st.download_button(
                                "📝 Download Markdown Report",
                                md_content,
                                file_name="forensic_report.md",
                                mime="text/markdown"
                            )
                    
                    with col3:
                        if os.path.exists(json_report):
                            with open(json_report, 'r', encoding='utf-8') as f:
                                json_content = f.read()
                            st.download_button(
                                "📊 Download JSON Report",
                                json_content,
                                file_name="forensic_report.json",
                                mime="application/json"
                            )
                            
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
