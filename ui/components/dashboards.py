import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_data_dashboard(df: pd.DataFrame):
    """
    Renders an interactive EDA dashboard for the raw dataset.
    """
    st.markdown("---")
    st.markdown("## 📊 Dataset Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data types pie chart
        dtypes_counts = df.dtypes.astype(str).value_counts().reset_index()
        dtypes_counts.columns = ['Data Type', 'Count']
        fig_dtypes = px.pie(
            dtypes_counts, 
            values='Count', 
            names='Data Type', 
            title="Column Data Types",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_dtypes.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ededed')
        )
        st.plotly_chart(fig_dtypes, use_container_width=True)
        
    with col2:
        # Missing values bar chart
        missing_counts = df.isnull().sum()
        missing_df = missing_counts[missing_counts > 0].reset_index()
        if not missing_df.empty:
            missing_df.columns = ['Column', 'Missing Values']
            missing_df = missing_df.sort_values('Missing Values', ascending=True)
            fig_missing = px.bar(
                missing_df, 
                x='Missing Values', 
                y='Column', 
                orientation='h',
                title="Missing Values by Column",
                color='Missing Values',
                color_continuous_scale='Reds'
            )
            fig_missing.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ededed')
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.info("✅ No missing values detected in the dataset.")

    # Correlation Matrix (if applicable)
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 1:
        with st.expander("🔗 Correlation Matrix Heatmap"):
            corr = df[num_cols].corr()
            fig_corr = px.imshow(
                corr, 
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Numeric Feature Correlation"
            )
            fig_corr.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ededed')
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # Distribution Explorer
    with st.expander("📈 Interactive Distribution Explorer"):
        if len(num_cols) > 0:
            selected_col = st.selectbox("Select a numeric column to view distribution", num_cols)
            fig_dist = px.histogram(
                df, 
                x=selected_col, 
                marginal="box", 
                title=f"Distribution of {selected_col}",
                color_discrete_sequence=['#5e6ad2']
            )
            fig_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ededed')
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("No numeric columns available for distribution analysis.")
