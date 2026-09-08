"""UI Header Component"""

import streamlit as st


def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Data Autopsy System</h1>
        <p>Data Quality · Statistical Forensics · ML Dataset Auditing</p>
    </div>
    """, unsafe_allow_html=True)
