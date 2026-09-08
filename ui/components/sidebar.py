"""UI Sidebar Component"""

import streamlit as st


def render_sidebar():
    """
    Render the sidebar with upload, options, and about section.

    Returns:
        tuple: (uploaded_file, options_dict)
    """
    with st.sidebar:
        st.header("📂 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file to perform forensic analysis",
        )

        st.markdown("---")

        st.header("⚙️ Analysis Options")
        run_quick = st.checkbox(
            "Quick Scan Only", value=False,
            help="Run a quick scan instead of full analysis",
        )
        generate_visuals = st.checkbox(
            "Generate Visualizations", value=True,
            help="Create visual evidence plots",
        )

        st.markdown("---")

        st.header("🎯 ML Audit (Optional)")
        target_column = st.text_input(
            "Target Column",
            value="target",
            help="Specify the target column for leakage detection and ML analysis",
        )

        st.markdown("---")

        st.header("🧪 Demo")
        if st.button("Load Massive Sample Dataset", use_container_width=True, type="primary"):
            st.session_state["use_sample"] = True
            
        use_sample = st.session_state.get("use_sample", False)
        if use_sample:
            if st.button("🗑️ Clear Sample", use_container_width=True):
                st.session_state["use_sample"] = False
                use_sample = False

        st.markdown("---")

        st.header("ℹ️ About")
        st.markdown("""
        **Data Autopsy v2.0** audits your datasets:
        - 📋 Data Quality & Consistency
        - 🕳️ Missing Data Patterns
        - 🚨 Anomaly Detection (Ensemble)
        - ⚖️ Bias & Representation
        - 🔒 PII / Privacy Risks
        - 🤖 ML Target Leakage
        - 🧪 Robustness Testing
        """)

    options = {
        "quick_scan": run_quick,
        "generate_visuals": generate_visuals,
        "target_column": target_column if target_column.strip() else None,
        "use_sample": use_sample,
    }

    return uploaded_file, options
