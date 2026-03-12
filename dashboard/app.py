import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.sidebar import render_sidebar
from dashboard.views import channel_analysis, outlier_finder, recommendations, ytuber


st.set_page_config(
    page_title="YouTube IP V3",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_global_css() -> None:
    css = """
    <style>
    :root {
        --yt-red: #FF0000;
        --yt-red-dark: #CC0000;
        --yt-bg: #0F0F23;
        --yt-bg-alt: #1A1A2E;
        --yt-surface: #16213E;
        --yt-accent: #00D4FF;
        --yt-success: #00E676;
        --yt-warning: #FFB300;
        --yt-text: #FFFFFF;
        --yt-text-muted: #B0B0B0;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1A1A2E 0%, #0F0F23 40%, #000000 100%) !important;
        color: var(--yt-text);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0F23 0%, #1A1A2E 50%, #0F0F23 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 999px !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.22) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.16);
        border-radius: 4px;
    }

    /* Headings */
    .yt-page-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #FFFFFF 0%, #FF0000 50%, #00D4FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .yt-page-subtitle {
        color: var(--yt-text-muted);
        font-size: 15px;
        margin-bottom: 1.5rem;
    }

    .yt-section-header {
        font-size: 22px;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .yt-section-underline {
        width: 72px;
        height: 3px;
        border-radius: 999px;
        background: linear-gradient(90deg, #FF0000, #00D4FF);
        margin-bottom: 1.25rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        flex: 1 1 160px;
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: radial-gradient(circle at top left, rgba(255,255,255,0.08) 0%, rgba(22,33,62,0.9) 35%, rgba(10,10,25,0.95) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 14px 35px rgba(0,0,0,0.55);
        backdrop-filter: blur(10px);
        transition: transform 0.15s ease-out, box-shadow 0.15s ease-out, border-color 0.15s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 45px rgba(0,0,0,0.75);
        border-color: rgba(255,0,0,0.55);
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--yt-text-muted);
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-icon {
        font-size: 18px;
        margin-right: 0.5rem;
    }
    .metric-delta {
        font-size: 12px;
        margin-top: 0.15rem;
    }
    .metric-delta.positive {
        color: var(--yt-success);
    }
    .metric-delta.negative {
        color: #FF6090;
    }

    /* Tables */
    .styled-dataframe table {
        border-radius: 10px;
        overflow: hidden;
    }
    .styled-dataframe thead tr th {
        background: linear-gradient(90deg, rgba(255,0,0,0.85), rgba(12,12,32,0.95));
        color: #FFFFFF !important;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }
    .styled-dataframe tbody tr:nth-child(odd) {
        background-color: rgba(255,255,255,0.015);
    }
    .styled-dataframe tbody tr:nth-child(even) {
        background-color: rgba(255,255,255,0.03);
    }

    /* Buttons */
    button[kind="primary"], .stButton > button {
        background: linear-gradient(90deg, #FF0000, #CC0000);
        color: #FFFFFF;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.16);
        padding: 0.4rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(0,0,0,0.6);
        transition: transform 0.12s ease-out, box-shadow 0.12s ease-out, background 0.15s;
    }
    button[kind="primary"]:hover, .stButton > button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 14px 30px rgba(0,0,0,0.8);
        background: linear-gradient(90deg, #FF3333, #FF0000);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div,
    .stDateInput > div > div,
    .stSlider > div > div {
        background-color: rgba(15,15,35,0.95) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        color: var(--yt-text) !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        outline: none !important;
        border-color: var(--yt-accent) !important;
        box-shadow: 0 0 0 1px rgba(0,212,255,0.55);
    }

    /* Tabs */
    [data-testid="stHorizontalBlock"] .stTabs {
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.12);
    }
    .stTabs [data-baseweb="tab"] {
        padding-bottom: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 14px;
        font-weight: 500;
    }

    /* Cards / containers */
    .yt-card {
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        background: radial-gradient(circle at top left, rgba(255,255,255,0.06) 0%, rgba(22,33,62,0.96) 40%, rgba(6,6,20,0.98) 100%);
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 16px 40px rgba(0,0,0,0.7);
        backdrop-filter: blur(12px);
        margin-bottom: 1.25rem;
        animation: fadeIn 0.35s ease-out;
    }

    /* Keyword chips */
    .keyword-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.1rem 0.6rem;
        border-radius: 999px;
        margin: 0.12rem;
        font-size: 12px;
        background: linear-gradient(90deg, rgba(255,0,0,0.25), rgba(0,212,255,0.25));
        border: 1px solid rgba(255,255,255,0.16);
        color: var(--yt-text);
        white-space: nowrap;
    }

    /* Thumbnail cards */
    .thumb-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 0.75rem;
    }
    .thumb-card {
        border-radius: 16px;
        overflow: hidden;
        background: #050511;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 12px 32px rgba(0,0,0,0.85);
        transition: transform 0.16s ease-out, box-shadow 0.16s ease-out;
    }
    .thumb-card:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 18px 42px rgba(0,0,0,0.95);
    }
    .thumb-card img {
        width: 100%;
        display: block;
    }
    .thumb-card-footer {
        padding: 0.4rem 0.6rem 0.6rem;
        font-size: 12px;
        color: var(--yt-text-muted);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.4s ease-out;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _render_hero() -> None:
    st.markdown(
        """
        <div class="fade-in" style="margin-bottom: 1.0rem;">
            <div class="yt-page-title">YouTube IP V3</div>
            <div class="yt-page-subtitle">
                Cross-channel analytics, benchmarking, and AI-assisted planning for YouTube creators.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_inject_global_css()

page = render_sidebar()

if page in {"Channel Analysis", "Recommendations"}:
    _render_hero()

if page == "Channel Analysis":
    channel_analysis.render()
elif page == "Recommendations":
    recommendations.render()
elif page == "Ytuber":
    ytuber.render()
elif page == "Outlier Finder":
    outlier_finder.render()
else:
    st.title("Deployment")
    st.markdown(
        """
        Deploy this repo directly from GitHub to Streamlit Community Cloud.

        ### Local Run
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        streamlit run streamlit_app.py
        ```

        ### Streamlit Cloud Settings
        - Repo: `royayushkr/Youtube-IP-V3`
        - Branch: `main`
        - Main file path: `streamlit_app.py`

        ### Secrets
        ```toml
        YOUTUBE_API_KEYS = ["youtube_key_1", "youtube_key_2"]
        GEMINI_API_KEYS = ["gemini_key_1", "gemini_key_2"]
        OPENAI_API_KEYS = ["openai_key_1", "openai_key_2"]
        ```

        Single-key fallbacks also work with `YOUTUBE_API_KEY`, `GEMINI_API_KEY`, and `OPENAI_API_KEY`.

        ### Notes
        - `dashboard/app.py` remains the main application module.
        - `streamlit_app.py` is the root-level deployment entrypoint for Streamlit Cloud.
        - Channel Analysis and Recommendations use the committed CSV datasets under `data/youtube api data/`.
        - The Ytuber suite uses live API calls and rotates across the configured key pools.
        - Outlier Finder is a standalone sidebar feature that scans the public YouTube search cohort, applies heuristic language filtering, and caches query results for one hour.

        ### Outlier Finder Methodology Summary
        - **Outlier Score:** weighted mix of baseline lift, peer percentile, engagement percentile, and recency boost.
        - **Views Per Day:** views divided by video age in days.
        - **Language Confidence:** metadata plus title-script heuristic, not a guaranteed classifier.
        - **Important Caveat:** YouTube search is ranked and sampled, so Outlier Finder is not an exhaustive index of every relevant video on YouTube.
        - For the full explanation of filters, metrics, and caveats, open **Outlier Finder -> Methodology** in the sidebar.

        ### Alternate Entrypoint
        ```bash
        streamlit run dashboard/app.py
        ```
        """
    )
