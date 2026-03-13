import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.sidebar import render_sidebar
from dashboard.components.theme import inject_shared_theme
from dashboard.views import channel_analysis, channel_insights, outlier_finder, recommendations, tools, ytuber


st.set_page_config(
    page_title="YouTube IP V4",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def _render_hero() -> None:
    st.markdown(
        """
            <div class="fade-in" style="margin-bottom: 1.0rem;">
            <div class="yt-page-title">YouTube IP V4</div>
            <div class="yt-page-subtitle">
                Cross-channel analytics, benchmarking, and AI-assisted planning for YouTube creators.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_shared_theme()

page = render_sidebar()

if page in {"Channel Analysis", "Recommendations"}:
    _render_hero()

if page == "Channel Analysis":
    channel_analysis.render()
elif page == "Recommendations":
    recommendations.render()
elif page == "Ytuber":
    ytuber.render()
elif page == "Channel Insights":
    channel_insights.render()
elif page == "Outlier Finder":
    outlier_finder.render()
elif page == "Tools":
    tools.render()
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
        - Repo: `royayushkr/Youtube-IP-V4`
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
        - Channel Insights stores dated SQLite snapshots on manual refresh and can optionally blend in owner-only YouTube Analytics metrics when Google OAuth is configured for the current session.
        - Outlier Finder is a standalone sidebar feature with a results-first flow, breakout snapshot, structured AI research, heuristic language filtering, and one-hour query caching.
        - Tools is a standalone sidebar utility page for metadata preview, thumbnails, transcripts, audio, and video downloads.
        - `packages.txt` installs `ffmpeg` for merged video downloads and MP3 conversion in the Tools page.

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
