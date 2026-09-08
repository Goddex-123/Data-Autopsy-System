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
        color = "#27AE60"
        emoji = "✅"
    elif overall >= 60:
        color = "#F39C12"
        emoji = "⚠️"
    elif overall >= 40:
        color = "#E67E22"
        emoji = "🟠"
    else:
        color = "#E74C3C"
        emoji = "🔴"

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border-radius: 16px; margin: 1.5rem 0; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid {color}; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <h2 style="color: {color}; margin-top: 0; font-size: 2.2rem; letter-spacing: -0.02em;">{emoji} {verdict}</h2>
        <p style="color: #ededed; font-size: 1.1rem; margin-bottom: 0;">Overall Health Score: <strong style="font-size: 1.5rem; color: #ffffff;">{overall:.0f}</strong><span style="color: #8a8f98;"> / 100</span></p>
    </div>
    """, unsafe_allow_html=True)

    # Component scores
    components = health_score.get("components", {})
    if components:
        cols = st.columns(min(len(components), 5))
        for i, (name, comp) in enumerate(components.items()):
            with cols[i % len(cols)]:
                render_score_card(comp.get("name", name), comp.get("score", 0))
