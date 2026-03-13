import streamlit as st
from streamlit_option_menu import option_menu

from dashboard.components.assistant_panel import render_assistant_panel


def render_sidebar() -> str:
    """Render the branded sidebar navigation and return the selected page."""
    page_options = [
        "Channel Analysis",
        "Recommendations",
        "Ytuber",
        "Channel Insights",
        "Outlier Finder",
        "Tools",
        "Deployment",
    ]
    current_page = st.session_state.get("app_page", page_options[0])
    if current_page not in page_options:
        current_page = page_options[0]

    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.35rem;">
                <div style="width:28px;height:20px;border-radius:6px;background:linear-gradient(135deg,#8B5CF6,#A855F7);display:flex;align-items:center;justify-content:center;box-shadow:0 10px 20px rgba(54,20,122,0.35);">
                    <span style="font-size:14px;font-weight:800;color:#FFFFFF;">▶</span>
                </div>
                <div>
                    <div style="font-weight:700;font-size:14px;letter-spacing:0.08em;text-transform:uppercase;color:#FFFFFF;">YouTube IP V4</div>
                    <div style="font-size:11px;color:#B8C1DA;">Creator Intelligence Suite</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin:0.15rem 0 0.5rem;font-size:11px;color:#8993B2;'>Navigate</div>", unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=page_options,
            icons=["bar-chart-fill", "bullseye", "rocket-takeoff-fill", "graph-up-arrow", "search", "tools", "gear"],
            default_index=page_options.index(current_page),
            styles={
                "container": {
                    "padding": "0.2rem 0 0.5rem",
                    "background": "transparent",
                },
                "icon": {
                    "color": "#C4B5FD",
                    "font-size": "16px",
                },
                "nav-link": {
                    "font-size": "13px",
                    "padding": "0.4rem 0.85rem",
                    "border-radius": "12px",
                    "color": "#B8C1DA",
                    "margin": "0.08rem 0",
                    "background": "rgba(255,255,255,0.02)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(90deg,rgba(139,92,246,0.95),rgba(168,85,247,0.78))",
                    "color": "#FFFFFF",
                    "box-shadow": "0 12px 24px rgba(31, 18, 71, 0.42)",
                },
            },
        )

        st.session_state["app_page"] = selected

        st.markdown("<hr style='border-color:rgba(255,255,255,0.10);margin:0.4rem 0 0.6rem;' />", unsafe_allow_html=True)

        render_assistant_panel(selected)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.10);margin:0.6rem 0 0.6rem;' />", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size:11px;color:#B8C1DA;margin-bottom:0.4rem;">
                Use <code>.env</code> locally or Streamlit secrets in deployment for <code>YOUTUBE_API_KEY</code>, <code>GEMINI_API_KEY</code>, and <code>OPENAI_API_KEY</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size:10px;color:#8993B2;margin-top:0.6rem;line-height:1.4;">
                <strong>Streamlit-ready deployment</strong><br/>
                Repo: royayushkr/Youtube-IP-V4
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected
