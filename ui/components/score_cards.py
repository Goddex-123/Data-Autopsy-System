"""UI Score Cards Component"""

import streamlit as st


def render_score_card(title: str, score: float, higher_is_better: bool = True):
    """Render a single score card."""
    if higher_is_better:
        css = "score-good" if score >= 70 else "score-warning" if score >= 40 else "score-danger"
    else:
        css = "score-good" if score <= 30 else "score-warning" if score <= 60 else "score-danger"

    st.markdown(f"""
    <div class="score-card {css}">
        <h3>{score:.0f}</h3>
        <p>{title}</p>
    </div>
    """, unsafe_allow_html=True)


def render_health_overview(health_score: dict):
    """Render the overall health score overview."""
    overall = health_score.get("overall", 0)
    verdict = health_score.get("verdict", "UNKNOWN")

    if overall >= 80:
        color = "green"
        emoji = "✅"
    elif overall >= 60:
        color = "orange"
        emoji = "⚠️"
    elif overall >= 40:
        color = "#E67E22"
        emoji = "🟠"
    else:
        color = "red"
        emoji = "🔴"

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 10px; margin: 1rem 0; border: 2px solid {color};">
        <h2 style="color: {color};">{emoji} {verdict}</h2>
        <p>Overall Health Score: <strong>{overall:.0f}/100</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Component scores
    components = health_score.get("components", {})
    if components:
        cols = st.columns(min(len(components), 5))
        for i, (name, comp) in enumerate(components.items()):
            with cols[i % len(cols)]:
                render_score_card(comp.get("name", name), comp.get("score", 0))
