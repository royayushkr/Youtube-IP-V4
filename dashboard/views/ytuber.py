import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:
    build = None
    HttpError = Exception

from dashboard.components.visualizations import (
    kpi_row,
    plotly_bar_chart,
    plotly_heatmap,
    plotly_line_chart,
    plotly_radar_chart,
    plotly_scatter,
    plotly_treemap,
    section_header,
    styled_dataframe,
    styled_keyword_chips,
)
from src.llm_integration.thumbnail_generator import ThumbnailGenerator
from src.services.public_channel_service import load_public_channel_workspace
from src.services.outliers_finder import (
    SUBSCRIBER_BUCKETS,
    OutlierSearchRequest,
    search_outlier_videos,
)
from src.utils.api_keys import get_provider_key_count, run_with_provider_keys


DATASET_PATH = os.path.join("data", "youtube api data", "research_science_channels_videos.csv")
DEFAULT_CATEGORY = "research_science"
THUMB_KEYS = ["default", "medium", "high", "standard", "maxres"]
STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "for", "with", "on", "and", "or", "at", "is", "are", "was", "were",
    "this", "that", "how", "why", "what", "when", "from", "your", "you", "my", "we", "our", "it", "vs", "into",
}
POWER_WORDS = {
    "secret", "ultimate", "proven", "easy", "fast", "best", "shocking", "truth", "mistake", "science",
    "future", "breakthrough", "insane", "new", "critical", "warning", "guide", "explained", "hidden", "top",
}
WORKSPACE_MODULES = [
    "AI Studio",
    "Overview",
    "Channel Audit",
    "Keyword Intel",
    "Outliers Finder",
    "Title & SEO Lab",
    "Competitor Benchmark",
    "Content Planner",
]
AI_STUDIO_TASKS = [
    "Full Pack (titles + descriptions + scripts + thumbnail concepts)",
    "Video Ideas",
    "Niche Expansion",
    "Titles Only",
    "Descriptions Only",
    "Scripts Only",
    "Hooks + CTAs",
    "Shorts Ideas",
    "Thumbnail Concepts",
]
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openai": "OpenAI / ChatGPT",
}
TEXT_MODEL_CATALOG = {
    "gemini": [
        {
            "id": "gemini-2.5-flash-lite",
            "label": "Gemini 2.5 Flash-Lite",
            "summary": "Fastest and lowest-cost Gemini text model",
            "input_per_million": 0.10,
            "output_per_million": 0.40,
        },
        {
            "id": "gemini-2.5-flash",
            "label": "Gemini 2.5 Flash",
            "summary": "Balanced reasoning, speed, and quality",
            "input_per_million": 0.30,
            "output_per_million": 2.50,
        },
    ],
    "openai": [
        {
            "id": "gpt-4o-mini",
            "label": "GPT-4o mini",
            "summary": "Fast, affordable model for production tasks",
            "input_per_million": 0.15,
            "output_per_million": 0.60,
        },
        {
            "id": "gpt-4.1-mini",
            "label": "GPT-4.1 mini",
            "summary": "Stronger instruction following at moderate cost",
            "input_per_million": 0.40,
            "output_per_million": 1.60,
        },
        {
            "id": "gpt-4.1",
            "label": "GPT-4.1",
            "summary": "Higher-quality non-reasoning model",
            "input_per_million": 2.00,
            "output_per_million": 8.00,
        },
        {
            "id": "gpt-4o",
            "label": "GPT-4o",
            "summary": "Flagship general-purpose model",
            "input_per_million": 2.50,
            "output_per_million": 10.00,
        },
    ],
}
IMAGE_MODEL_CATALOG = {
    "gemini": [
        {
            "id": "gemini-2.5-flash-image",
            "label": "Gemini 2.5 Flash Image",
            "summary": "Native Gemini image generation optimized for speed",
            "per_image": 0.039,
            "input_per_million": 0.30,
            "size_options": ["1024x1024"],
            "quality_options": ["standard"],
        },
    ],
    "openai": [
        {
            "id": "gpt-image-1.5",
            "label": "GPT Image 1.5",
            "summary": "Latest OpenAI image model with strongest prompt adherence",
            "pricing": {
                "low": {"1024x1024": 0.009, "1024x1536": 0.013, "1536x1024": 0.013},
                "medium": {"1024x1024": 0.034, "1024x1536": 0.050, "1536x1024": 0.050},
                "high": {"1024x1024": 0.133, "1024x1536": 0.200, "1536x1024": 0.200},
            },
            "size_options": ["1024x1024", "1024x1536", "1536x1024"],
            "quality_options": ["low", "medium", "high"],
            "background_options": ["opaque", "transparent"],
            "format_options": ["png", "webp", "jpeg"],
        },
        {
            "id": "gpt-image-1-mini",
            "label": "GPT Image 1 Mini",
            "summary": "Lower-cost thumbnail generation",
            "pricing": {
                "low": {"1024x1024": 0.005, "1024x1536": 0.006, "1536x1024": 0.006},
                "medium": {"1024x1024": 0.011, "1024x1536": 0.015, "1536x1024": 0.015},
                "high": {"1024x1024": 0.036, "1024x1536": 0.052, "1536x1024": 0.052},
            },
            "size_options": ["1024x1024", "1024x1536", "1536x1024"],
            "quality_options": ["low", "medium", "high"],
            "background_options": ["opaque", "transparent"],
            "format_options": ["png", "webp", "jpeg"],
        },
        {
            "id": "gpt-image-1",
            "label": "GPT Image 1",
            "summary": "Higher-fidelity image generation",
            "pricing": {
                "low": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016},
                "medium": {"1024x1024": 0.042, "1024x1536": 0.063, "1536x1024": 0.063},
                "high": {"1024x1024": 0.167, "1024x1536": 0.250, "1536x1024": 0.250},
            },
            "size_options": ["1024x1024", "1024x1536", "1536x1024"],
            "quality_options": ["low", "medium", "high"],
            "background_options": ["opaque", "transparent"],
            "format_options": ["png", "webp", "jpeg"],
        },
    ],
}
AUDIENCE_OPTIONS = ["Broad", "Beginner", "Intermediate", "Advanced"]
FORMAT_OPTIONS = ["Long-form", "Shorts", "Mixed"]
STRATEGY_FILTERS = [
    "Evergreen",
    "Search-first",
    "Trend-reactive",
    "Authority-building",
    "High CTR",
    "Sponsor-safe",
]
OUTLIER_TIMEFRAME_OPTIONS = ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"]
OUTLIER_REGION_OPTIONS = ["Any", "US", "IN", "GB", "CA", "AU", "DE", "FR", "BR", "JP"]
OUTLIER_LANGUAGE_OPTIONS = ["Any", "en", "es", "hi", "pt", "de", "fr", "ja"]
OUTLIER_SORT_OPTIONS = {
    "Outlier Score": "outlier_score",
    "Views / Day": "views_per_day",
    "Views": "views",
    "Engagement": "engagement_rate",
    "Views / Subscriber": "views_per_subscriber",
}


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _join_list(x: Optional[List[str]]) -> str:
    if not x:
        return ""
    return "|".join([str(i) for i in x])


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_channel_query(prompt_text: str) -> str:
    text = prompt_text.strip()
    if not text:
        return ""

    match = re.search(r"(UC[\w-]{20,}|@[A-Za-z0-9._-]+)", text)
    if match:
        return match.group(1)
    return text


def _goal_from_prompt(prompt_text: str) -> str:
    cleaned = re.sub(r"(UC[\w-]{20,}|@[A-Za-z0-9._-]+)", "", prompt_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned


def _inject_ytuber_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top center, rgba(63, 85, 127, 0.42) 0%, transparent 24%),
                radial-gradient(circle at top left, rgba(139, 92, 246, 0.16) 0%, transparent 18%),
                linear-gradient(180deg, #2A3446 0%, #1C2433 44%, #121A27 100%) !important;
        }
        [data-testid="stAppViewBlockContainer"] {
            max-width: 1240px !important;
            padding-top: 1.35rem !important;
        }
        .ytuber-hero {
            max-width: 860px;
            margin: 0 auto 1rem;
            text-align: center;
        }
        .ytuber-brand-row {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #FFFFFF;
            font-size: 12px;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .ytuber-brand-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, #A855F7, #8B5CF6);
            box-shadow: 0 0 16px rgba(139, 92, 246, 0.55);
        }
        .ytuber-kicker {
            font-size: 12px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #94A3C7;
            margin-bottom: 0.5rem;
        }
        .ytuber-title {
            font-size: clamp(32px, 4vw, 50px);
            line-height: 1.02;
            font-weight: 800;
            color: #FFFFFF;
            max-width: 760px;
            margin: 0 auto 0.7rem;
        }
        .ytuber-subtitle {
            font-size: 15px;
            color: #D4DBEE;
            max-width: 680px;
            margin: 0 auto;
        }
        .ytuber-search-meta {
            margin-top: 0.9rem;
            font-size: 13px;
            color: #B8C2DE;
        }
        .ytuber-command-title {
            font-size: 18px;
            color: #FFFFFF;
            font-weight: 700;
            margin-bottom: 0.2rem;
            text-align: center;
        }
        .ytuber-command-subtitle {
            font-size: 13px;
            color: #AEB8D6;
            margin-bottom: 0.85rem;
            text-align: center;
        }
        .ytuber-toolbar-note {
            font-size: 12px;
            color: #AEB8D6;
            margin-top: 0.45rem;
            text-align: center;
        }
        .ytuber-empty-copy {
            text-align: center;
            color: #AEB8D6;
            font-size: 13px;
            margin: 0.6rem 0 0.2rem;
        }
        .ytuber-banner {
            border-radius: 20px;
            padding: 0.95rem 1rem;
            margin-bottom: 1rem;
            background: linear-gradient(180deg, rgba(45,58,86,0.92) 0%, rgba(29,38,57,0.96) 100%);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 14px 32px rgba(7, 12, 21, 0.25);
        }
        .ytuber-banner-title {
            font-size: 18px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.25rem;
        }
        .ytuber-banner-meta {
            font-size: 13px;
            color: #B7C1DD;
        }
        .ytuber-footer-card {
            padding: 0.48rem 0.62rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .ytuber-footer-label {
            font-size: 10px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8D9ABF;
        }
        .ytuber-footer-value {
            font-size: 15px;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.2;
            margin-top: 0.1rem;
        }
        .ytuber-footer-detail {
            font-size: 11px;
            color: #B4BED9;
            margin-top: 0.08rem;
        }
        .outlier-card {
            border-radius: 20px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(40, 50, 70, 0.96) 0%, rgba(28, 37, 56, 0.99) 100%);
            border: 1px solid rgba(255,255,255,0.09);
            box-shadow: 0 14px 30px rgba(8, 12, 20, 0.24);
            margin-bottom: 1rem;
        }
        .outlier-card img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            display: block;
            background: rgba(255,255,255,0.04);
        }
        .outlier-card-body {
            padding: 0.9rem 1rem 1rem;
        }
        .outlier-card-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.9rem;
            margin-bottom: 0.55rem;
        }
        .outlier-card-title {
            font-size: 16px;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.35;
            margin-bottom: 0.2rem;
        }
        .outlier-card-channel {
            font-size: 12px;
            color: #AFBAD8;
        }
        .outlier-score-badge {
            flex-shrink: 0;
            min-width: 70px;
            text-align: center;
            padding: 0.4rem 0.6rem;
            border-radius: 16px;
            background: rgba(139, 92, 246, 0.16);
            border: 1px solid rgba(196, 181, 253, 0.22);
            color: #FFFFFF;
        }
        .outlier-score-value {
            font-size: 20px;
            line-height: 1;
            font-weight: 800;
        }
        .outlier-score-label {
            font-size: 10px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #DCCFFF;
            margin-top: 0.15rem;
        }
        .outlier-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.55rem 0 0.7rem;
        }
        .outlier-metric-pill {
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 11px;
            color: #D3DBEF;
        }
        .outlier-reasons {
            margin: 0;
            padding-left: 1rem;
            color: #E6ECFA;
            font-size: 12px;
            line-height: 1.5;
        }
        .outlier-reasons li {
            margin-bottom: 0.15rem;
        }
        .outlier-link {
            margin-top: 0.7rem;
            display: inline-block;
            color: #D8C7FF !important;
            font-size: 12px;
            font-weight: 600;
            text-decoration: none;
        }
        .ytuber-score-card {
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(40, 50, 70, 0.96) 0%, rgba(28, 37, 56, 0.99) 100%);
            border: 1px solid rgba(255,255,255,0.09);
            box-shadow: 0 14px 30px rgba(8, 12, 20, 0.24);
            margin-bottom: 0.8rem;
        }
        .ytuber-score-label {
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #9BA7CB;
            margin-bottom: 0.3rem;
        }
        .ytuber-score-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .ytuber-score-value {
            font-size: 38px;
            line-height: 1;
            font-weight: 800;
            color: #FFFFFF;
        }
        .ytuber-score-pill {
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .ytuber-score-bar {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,0.07);
            margin: 0.8rem 0 0.45rem;
        }
        .ytuber-score-bar span {
            display: block;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #8B5CF6 0%, #A855F7 55%, #60A5FA 100%);
        }
        .ytuber-score-note {
            font-size: 12px;
            color: #B9C4DE;
        }
        .ytuber-part-card {
            padding: 0.68rem 0.78rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.07);
            margin-bottom: 0.55rem;
        }
        .ytuber-part-label {
            font-size: 11px;
            color: #9BA7CB;
            margin-bottom: 0.1rem;
        }
        .ytuber-part-value {
            font-size: 18px;
            font-weight: 700;
            color: #FFFFFF;
        }
        .ytuber-section-surface {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1rem 1rem 0.5rem;
            margin-bottom: 1rem;
        }
        .yt-section-header {
            font-size: 20px !important;
            margin-top: 1.1rem !important;
            margin-bottom: 0.45rem !important;
        }
        .yt-section-underline {
            display: none !important;
        }
        .yt-card,
        .metric-card {
            background: linear-gradient(180deg, rgba(41, 52, 73, 0.95) 0%, rgba(28, 37, 55, 0.98) 100%) !important;
            border: 1px solid rgba(255,255,255,0.09) !important;
            box-shadow: 0 14px 30px rgba(8, 12, 20, 0.30) !important;
        }
        .keyword-chip {
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.22), rgba(96, 165, 250, 0.14)) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: none !important;
            background: rgba(255,255,255,0.04);
            padding: 0.35rem;
            border-radius: 999px;
            margin-bottom: 0.85rem;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto;
            border-radius: 999px;
            padding: 0.42rem 0.95rem;
            color: #C5CEE6;
            border: 1px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(139, 92, 246, 0.16) !important;
            border-color: rgba(196, 181, 253, 0.28) !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        .stButton > button {
            background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03)) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            color: #FFFFFF !important;
            box-shadow: none !important;
        }
        .stButton > button:hover {
            background: linear-gradient(180deg, rgba(255,255,255,0.11), rgba(255,255,255,0.05)) !important;
            border-color: rgba(255,255,255,0.24) !important;
            transform: translateY(-1px);
        }
        button[kind="primary"] {
            background: linear-gradient(90deg, #8B5CF6, #A855F7) !important;
            border-color: rgba(196, 181, 253, 0.28) !important;
            box-shadow: 0 12px 24px rgba(83, 44, 184, 0.28) !important;
        }
        button[kind="primary"]:hover {
            background: linear-gradient(90deg, #9C74F8, #B566FA) !important;
            border-color: rgba(196, 181, 253, 0.38) !important;
        }
        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stSelectbox > div > div,
        .stDateInput > div > div,
        .stMultiSelect > div > div {
            background-color: rgba(38, 48, 71, 0.92) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }
        .stTextInput > div > div > input {
            min-height: 3.05rem !important;
            border-radius: 14px !important;
        }
        div[data-testid="stTextInput"]:has(input[aria-label="Search Channel"]) [data-baseweb="input"] {
            min-height: 3.5rem !important;
            border-radius: 999px !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            background: linear-gradient(180deg, rgba(46, 57, 81, 0.96) 0%, rgba(37, 47, 69, 0.99) 100%) !important;
            box-shadow: 0 12px 26px rgba(10, 15, 24, 0.16) !important;
        }
        div[data-testid="stTextInput"]:has(input[aria-label="Search Channel"]) [data-baseweb="input"]:focus-within {
            border-color: rgba(139, 92, 246, 0.45) !important;
            box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.24), 0 16px 34px rgba(10, 15, 24, 0.20) !important;
        }
        div[data-testid="stTextInput"]:has(input[aria-label="Search Channel"]) input {
            font-size: 16px !important;
            color: #FFFFFF !important;
        }
        .stButton > button,
        .stFormSubmitButton > button {
            min-height: 3.25rem;
        }
        [data-testid="stToggle"] label p,
        [data-testid="stToggle"] small {
            color: #C5CEE6 !important;
        }
        @media (max-width: 900px) {
            .ytuber-title {
                font-size: 34px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_connection_cards(provider_counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    if provider_counts is None:
        provider_counts = {
            "youtube": get_provider_key_count("youtube"),
            "gemini": get_provider_key_count("gemini"),
            "openai": get_provider_key_count("openai"),
        }

    cards = [
        ("YouTube Pool", provider_counts["youtube"], "Live channel fetch and benchmarking"),
        ("Gemini Pool", provider_counts["gemini"], "Text and image generation"),
        ("OpenAI Pool", provider_counts["openai"], "ChatGPT text and image generation"),
    ]

    cols = st.columns(3)
    for col, (label, count, detail) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="yt-card ytuber-status-card">
                    <div class="ytuber-status-label">{escape(label)}</div>
                    <div class="ytuber-status-value">{count}</div>
                    <div class="ytuber-status-detail">{escape(detail)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return provider_counts


def _render_pool_footer(provider_counts: Dict[str, int], workspace_meta: Optional[Dict[str, Any]] = None) -> None:
    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
    footer_cards: List[Tuple[str, str, str]] = []
    if workspace_meta:
        footer_cards.extend(
            [
                ("Data Source", workspace_meta.get("source_label", "Unavailable"), "How this channel was loaded"),
                ("Channel ID", workspace_meta.get("channel_id", "Unavailable"), "Reference identifier"),
                ("Videos In View", f"{workspace_meta.get('video_count', 0):,}", "Posts included in the workspace"),
            ]
        )

    footer_cards.extend(
        [
            ("YouTube Pool", str(provider_counts["youtube"]), "Live fetch"),
            ("Gemini Pool", str(provider_counts["gemini"]), "AI text/image"),
            ("OpenAI Pool", str(provider_counts["openai"]), "ChatGPT/image"),
        ]
    )

    for row_start in range(0, len(footer_cards), 3):
        row_cards = footer_cards[row_start:row_start + 3]
        cols = st.columns(len(row_cards))
        for col, (label, value, detail) in zip(cols, row_cards):
            with col:
                st.markdown(
                    f"""
                    <div class="ytuber-footer-card">
                        <div class="ytuber-footer-label">{escape(label)}</div>
                        <div class="ytuber-footer-value">{escape(value)}</div>
                        <div class="ytuber-footer-detail">{escape(detail)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _catalog_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in items}


def _format_text_model_option(provider: str, model_id: str) -> str:
    meta = _catalog_map(TEXT_MODEL_CATALOG[provider])[model_id]
    return f"{meta['label']}  •  ${meta['input_per_million']}/${meta['output_per_million']} per 1M in/out"


def _format_image_model_option(provider: str, model_id: str) -> str:
    meta = _catalog_map(IMAGE_MODEL_CATALOG[provider])[model_id]
    if provider == "gemini":
        return f"{meta['label']}  •  ~${meta['per_image']:.3f} per image"
    default_quality = meta["quality_options"][1] if len(meta["quality_options"]) > 1 else meta["quality_options"][0]
    default_size = meta["size_options"][0]
    default_cost = meta["pricing"][default_quality][default_size]
    return f"{meta['label']}  •  from ${default_cost:.3f} per image"


def _estimate_text_cost(
    provider: str,
    model_id: str,
    task: str,
    idea_count: int,
    script_count: int,
    thumbnail_count: int,
) -> Tuple[int, int, float]:
    meta = _catalog_map(TEXT_MODEL_CATALOG[provider])[model_id]

    base_input_tokens = 1800
    output_tokens = 0
    if task == "Full Pack (titles + descriptions + scripts + thumbnail concepts)":
        output_tokens = idea_count * 240 + script_count * 1100 + thumbnail_count * 80
    elif task == "Video Ideas":
        output_tokens = idea_count * 180
    elif task == "Niche Expansion":
        output_tokens = idea_count * 220
    elif task == "Titles Only":
        output_tokens = max(idea_count, 1) * 90
    elif task == "Descriptions Only":
        output_tokens = max(idea_count, 1) * 180
    elif task == "Scripts Only":
        output_tokens = script_count * 1400
    elif task == "Hooks + CTAs":
        output_tokens = max(idea_count, 1) * 120
    elif task == "Shorts Ideas":
        output_tokens = idea_count * 130
    elif task == "Thumbnail Concepts":
        output_tokens = thumbnail_count * 75

    estimated_cost = (
        (base_input_tokens / 1_000_000) * meta["input_per_million"]
        + (output_tokens / 1_000_000) * meta["output_per_million"]
    )
    return base_input_tokens, output_tokens, estimated_cost


def _estimate_image_cost(
    provider: str,
    model_id: str,
    image_count: int,
    size: str,
    quality: str,
) -> Tuple[float, float]:
    meta = _catalog_map(IMAGE_MODEL_CATALOG[provider])[model_id]
    if provider == "gemini":
        per_image = meta["per_image"]
    else:
        per_image = meta["pricing"][quality][size]
    return per_image, per_image * image_count


def _build_trend_radar_df(channel_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"video_publishedAt", "video_title"}
    if channel_df.empty or not required_cols.issubset(channel_df.columns):
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    recent_60 = channel_df[channel_df["video_publishedAt"] >= (now - timedelta(days=60))]
    prev_60 = channel_df[
        (channel_df["video_publishedAt"] < (now - timedelta(days=60)))
        & (channel_df["video_publishedAt"] >= (now - timedelta(days=120)))
    ]

    def keyword_counter(frame: pd.DataFrame) -> Counter:
        c = Counter()
        if frame.empty:
            return c
        for title in frame["video_title"].fillna("").astype(str):
            c.update(set(_tokenize(title)))
        return c

    c_recent = keyword_counter(recent_60)
    c_prev = keyword_counter(prev_60)

    rows = []
    for kw, recent_count in c_recent.items():
        prev_count = c_prev.get(kw, 0)
        rows.append(
            {
                "keyword": kw,
                "recent_mentions": recent_count,
                "previous_mentions": prev_count,
                "momentum_delta": recent_count - prev_count,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["momentum_delta", "recent_mentions"], ascending=[False, False])
        .head(25)
    )


def _generate_competitor_recommendations(
    channel_title: str,
    own_stats: Dict[str, Any],
    benchmark_df: pd.DataFrame,
    competitor_trend_df: pd.DataFrame,
    keyword_gap_df: pd.DataFrame,
) -> str:
    if get_provider_key_count("gemini") <= 0:
        return "Add a Gemini API key pool to generate benchmark recommendations."

    top_competitors = benchmark_df.head(5).to_dict(orient="records")
    trend_rows = competitor_trend_df.head(8).to_dict(orient="records")
    gap_rows = keyword_gap_df.head(8).to_dict(orient="records")

    prompt = (
        "You are a senior YouTube growth strategist. "
        "Compare the user's channel against competitor benchmark data and explain what the channel should do next. "
        "Keep the tone polished, concise, and specific.\n\n"
        f"User channel: {channel_title}\n"
        f"User stats: {own_stats}\n"
        f"Top competitor rows: {top_competitors}\n"
        f"Competitor trend radar: {trend_rows}\n"
        f"Keyword gaps vs competitors: {gap_rows}\n\n"
        "Return markdown with these sections:\n"
        "1. What competitors are doing differently\n"
        "2. Immediate next steps\n"
        "3. Content angles to test next\n"
        "4. Risks to avoid\n"
        "Use short bullets and plain language."
    )
    return run_with_provider_keys(
        "gemini",
        lambda key: _gemini_generate_text(key, "gemini-2.5-flash", prompt),
        retryable_error=_is_ai_retryable_error,
    )


def _is_youtube_retryable_error(exc: Exception) -> bool:
    status = getattr(getattr(exc, "resp", None), "status", None)
    if status in (400, 401, 403, 429, 500, 503):
        return True

    message = str(exc).lower()
    retry_tokens = (
        "quota",
        "rate limit",
        "resource exhausted",
        "api key",
        "403",
        "429",
        "500",
        "503",
        "backenderror",
        "service unavailable",
        "daily limit",
        "forbidden",
        "access not configured",
    )
    return any(token in message for token in retry_tokens)


def _is_ai_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_tokens = (
        "quota",
        "rate limit",
        "resource exhausted",
        "too many requests",
        "insufficient_quota",
        "api key",
        "401",
        "403",
        "429",
        "500",
        "503",
        "overloaded",
    )
    return any(token in message for token in retry_tokens)


def _generate_text_with_provider_pool(provider: str, model: str, prompt: str) -> str:
    provider_name = provider.lower().strip()
    if provider_name == "gemini":
        return run_with_provider_keys(
            "gemini",
            lambda key: _gemini_generate_text(key, model, prompt),
            retryable_error=_is_ai_retryable_error,
        )
    if provider_name == "openai":
        return run_with_provider_keys(
            "openai",
            lambda key: _openai_generate_text(key, model, prompt),
            retryable_error=_is_ai_retryable_error,
        )
    raise RuntimeError(f"Unsupported text provider: {provider}")


def _generate_images_with_provider_pool(
    provider: str,
    model: str,
    *,
    title: str,
    context: str,
    style: str,
    negative_prompt: str,
    count: int,
    size: str,
    quality: str,
    output_format: str,
    background: str,
) -> List[Any]:
    provider_name = provider.lower().strip()
    def _run_generate(key: str) -> List[Any]:
        generator = ThumbnailGenerator(
            provider=provider_name,
            api_key=key,
            model=model,
        )
        try:
            return generator.generate(
                title=title,
                context=context,
                style=style,
                negative_prompt=negative_prompt,
                count=count,
                size=size,
                quality=quality,
                output_format=output_format,
                background=background,
            )
        except TypeError as exc:
            text = str(exc)
            if "unexpected keyword argument 'quality'" not in text and "unexpected keyword argument 'output_format'" not in text and "unexpected keyword argument 'background'" not in text:
                raise
            return generator.generate(
                title=title,
                context=context,
                style=style,
                negative_prompt=negative_prompt,
                count=count,
                size=size,
            )

    return run_with_provider_keys(
        provider_name,
        _run_generate,
        retryable_error=_is_ai_retryable_error,
    )


def _api_call_with_backoff(fn, max_retries: int = 7):
    delay = 1.0
    last_exc = None
    for _ in range(max_retries):
        try:
            return fn()
        except HttpError as e:
            last_exc = e
            status = getattr(e.resp, "status", None)
            if status in (403, 429, 500, 503):
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
    raise RuntimeError(f"API call failed after retries: {last_exc}")


def _yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def _resolve_channel_id(youtube, handle_or_query: str) -> str:
    q = handle_or_query.strip()
    if q.startswith("UC") and len(q) >= 20:
        return q

    req = youtube.search().list(part="snippet", q=q, type="channel", maxResults=1)
    resp = _api_call_with_backoff(req.execute)
    items = resp.get("items", [])

    if not items and q.startswith("@"):
        q2 = q[1:]
        req2 = youtube.search().list(part="snippet", q=q2, type="channel", maxResults=1)
        resp2 = _api_call_with_backoff(req2.execute)
        items = resp2.get("items", [])

    if not items:
        raise RuntimeError(f"No channel found for: {handle_or_query}")

    channel_id = _safe_get(items[0], ["snippet", "channelId"])
    if not channel_id:
        raise RuntimeError(f"Search returned item without channelId for: {handle_or_query}")
    return channel_id


def _fetch_channel_details(youtube, channel_id: str) -> Dict[str, Any]:
    req = youtube.channels().list(
        part="snippet,contentDetails,statistics,brandingSettings,status,topicDetails",
        id=channel_id,
        maxResults=1,
    )
    resp = _api_call_with_backoff(req.execute)
    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"No channel details returned for channelId: {channel_id}")
    return items[0]


def _fetch_recent_video_ids(
    youtube,
    uploads_playlist_id: str,
    published_after_utc: datetime,
    max_videos: int = 600,
) -> List[str]:
    video_ids: List[str] = []
    page_token: Optional[str] = None
    stop = False

    while len(video_ids) < max_videos and not stop:
        req = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist_id,
            maxResults=min(50, max_videos - len(video_ids)),
            pageToken=page_token,
        )
        resp = _api_call_with_backoff(req.execute)

        for it in resp.get("items", []):
            vid = _safe_get(it, ["contentDetails", "videoId"])
            published_at = _safe_get(it, ["snippet", "publishedAt"])
            if not vid or not published_at:
                continue
            dt = pd.to_datetime(published_at, errors="coerce", utc=True)
            if pd.isna(dt):
                continue
            if dt.to_pydatetime() < published_after_utc:
                stop = True
                break
            video_ids.append(vid)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return video_ids


def _fetch_videos_details(youtube, video_ids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        if not chunk:
            continue
        req = youtube.videos().list(
            part="snippet,contentDetails,statistics,status,topicDetails",
            id=",".join(chunk),
            maxResults=50,
        )
        resp = _api_call_with_backoff(req.execute)
        out.extend(resp.get("items", []))
    return out


def _extract_thumbnails(thumbnails: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(thumbnails, dict):
        thumbnails = {}

    for k in THUMB_KEYS:
        v = thumbnails.get(k, {}) if isinstance(thumbnails.get(k, {}), dict) else {}
        out[f"thumb_{k}_url"] = v.get("url", "")
        out[f"thumb_{k}_width"] = v.get("width", "")
        out[f"thumb_{k}_height"] = v.get("height", "")
    return out


def _channel_fields(channel: Dict[str, Any], handle: str) -> Dict[str, Any]:
    snippet = channel.get("snippet", {}) or {}
    stats = channel.get("statistics", {}) or {}
    branding = _safe_get(channel, ["brandingSettings", "channel"], {}) or {}
    status = channel.get("status", {}) or {}
    topic = channel.get("topicDetails", {}) or {}

    uploads_pid = _safe_get(channel, ["contentDetails", "relatedPlaylists", "uploads"], "")

    return {
        "snapshot_utc": _iso_now(),
        "category_name": DEFAULT_CATEGORY,
        "channel_handle_used": handle,
        "channel_id": channel.get("id", ""),
        "channel_title": snippet.get("title", ""),
        "channel_description": snippet.get("description", ""),
        "channel_publishedAt": snippet.get("publishedAt", ""),
        "uploads_playlist_id": uploads_pid,
        "channel_country": branding.get("country", ""),
        "channel_keywords": branding.get("keywords", ""),
        "channel_defaultLanguage": branding.get("defaultLanguage", ""),
        "channel_madeForKids": status.get("madeForKids", ""),
        "channel_isLinked": status.get("isLinked", ""),
        "channel_subscriberCount": stats.get("subscriberCount", ""),
        "channel_viewCount": stats.get("viewCount", ""),
        "channel_videoCount": stats.get("videoCount", ""),
        "channel_topicCategories": _join_list(topic.get("topicCategories")),
        "channel_topicIds": _join_list(topic.get("topicIds")),
    }


def _video_row(video: Dict[str, Any], ch: Dict[str, Any]) -> Dict[str, Any]:
    sn = video.get("snippet", {}) or {}
    cd = video.get("contentDetails", {}) or {}
    stx = video.get("statistics", {}) or {}
    vs = video.get("status", {}) or {}
    tp = video.get("topicDetails", {}) or {}

    thumbs = sn.get("thumbnails", {}) or {}
    thumb_cols = _extract_thumbnails(thumbs)

    return {
        **ch,
        "video_id": video.get("id", ""),
        "video_title": sn.get("title", ""),
        "video_description": sn.get("description", ""),
        "video_publishedAt": sn.get("publishedAt", ""),
        "video_channelId": sn.get("channelId", ""),
        "video_categoryId": sn.get("categoryId", ""),
        "video_tags": _join_list(sn.get("tags")),
        "video_defaultLanguage": sn.get("defaultLanguage", ""),
        "video_defaultAudioLanguage": sn.get("defaultAudioLanguage", ""),
        **thumb_cols,
        "views": stx.get("viewCount", ""),
        "likes": stx.get("likeCount", ""),
        "comments": stx.get("commentCount", ""),
        "duration": cd.get("duration", ""),
        "caption": cd.get("caption", ""),
        "licensedContent": cd.get("licensedContent", ""),
        "definition": cd.get("definition", ""),
        "projection": cd.get("projection", ""),
        "madeForKids": vs.get("madeForKids", ""),
        "embeddable": vs.get("embeddable", ""),
        "video_topicCategories": _join_list(tp.get("topicCategories")),
        "video_topicIds": _join_list(tp.get("topicIds")),
    }


def _load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH)


def _append_rows_to_dataset(new_rows: pd.DataFrame, existing_df: pd.DataFrame) -> None:
    if new_rows.empty:
        return

    if existing_df.empty:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        new_rows.to_csv(DATASET_PATH, index=False)
        return

    existing_cols = existing_df.columns.tolist()
    for c in existing_cols:
        if c not in new_rows.columns:
            new_rows[c] = ""

    for c in new_rows.columns:
        if c not in existing_cols:
            existing_df[c] = ""
            existing_cols.append(c)

    new_rows = new_rows[existing_cols]
    new_rows.to_csv(DATASET_PATH, mode="a", header=False, index=False)


def _parse_iso_duration_seconds(duration: str) -> int:
    if not isinstance(duration, str):
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _ensure_numeric_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["views", "likes", "comments"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "video_publishedAt" in out.columns:
        out["video_publishedAt"] = pd.to_datetime(out["video_publishedAt"], errors="coerce", utc=True)
    out["engagement_rate"] = (out["likes"].fillna(0) + out["comments"].fillna(0)) / out["views"].clip(lower=1)
    out["publish_month"] = out["video_publishedAt"].dt.to_period("M").astype(str)
    out["publish_day"] = out["video_publishedAt"].dt.day_name()
    out["publish_hour"] = out["video_publishedAt"].dt.hour
    out["duration_seconds"] = out["duration"].fillna("").astype(str).map(_parse_iso_duration_seconds)
    out["is_short"] = out["duration_seconds"] <= 60
    return out


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def _top_keywords(df: pd.DataFrame, top_n: int = 30) -> List[str]:
    counter: Counter = Counter()
    for title in df["video_title"].fillna("").astype(str):
        counter.update(_tokenize(title))
    return [k for k, _ in counter.most_common(top_n)]


def _keyword_intel(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for _, row in df.iterrows():
        title = str(row.get("video_title", ""))
        views = float(row.get("views") or 0)
        eng = float(row.get("engagement_rate") or 0)
        published = row.get("video_publishedAt")
        recency_weight = 1.0
        if pd.notna(published):
            days = (now - published.to_pydatetime()).days
            recency_weight = max(0.1, 1 - min(days / 365, 0.9))

        seen = set(_tokenize(title))
        for token in seen:
            rows.append(
                {
                    "keyword": token,
                    "views": views,
                    "engagement": eng,
                    "recency_weight": recency_weight,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["keyword", "videos", "avg_views", "avg_engagement", "momentum", "score"])

    kdf = pd.DataFrame(rows)
    out = (
        kdf.groupby("keyword", dropna=False)
        .agg(
            videos=("keyword", "count"),
            avg_views=("views", "mean"),
            avg_engagement=("engagement", "mean"),
            momentum=("recency_weight", "mean"),
        )
        .reset_index()
    )

    if out.empty:
        return out

    max_views = max(out["avg_views"].max(), 1)
    max_eng = max(out["avg_engagement"].max(), 0.0001)
    max_momentum = max(out["momentum"].max(), 0.0001)
    competition_proxy = out["videos"] / max(out["videos"].max(), 1)

    out["score"] = (
        (out["avg_views"] / max_views) * 40
        + (out["avg_engagement"] / max_eng) * 30
        + (out["momentum"] / max_momentum) * 20
        + (1 - competition_proxy) * 10
    )

    return out.sort_values("score", ascending=False).head(top_n)


def _title_score(title: str, keyword_hints: List[str]) -> Tuple[int, Dict[str, int], List[str]]:
    text = title.strip()
    lower = text.lower()

    length = len(text)
    word_count = len(text.split())

    length_score = max(0, 30 - int(abs(length - 55) * 0.8))
    clarity_score = 20 if 6 <= word_count <= 12 else max(6, 20 - abs(word_count - 9) * 2)
    number_score = 12 if re.search(r"\d", text) else 0
    curiosity_score = 10 if any(p in text for p in ["?", "!", ":"]) else 0
    power_score = min(15, sum(1 for w in POWER_WORDS if w in lower) * 5)
    keyword_matches = sum(1 for k in keyword_hints[:12] if k and k in lower)
    keyword_score = min(13, keyword_matches * 3)

    total = max(0, min(100, length_score + clarity_score + number_score + curiosity_score + power_score + keyword_score))

    suggestions: List[str] = []
    if length < 40:
        suggestions.append("Make title more specific; 45-65 chars usually performs better.")
    if length > 70:
        suggestions.append("Shorten title for stronger mobile readability.")
    if number_score == 0:
        suggestions.append("Consider adding a number or quantified claim.")
    if curiosity_score == 0:
        suggestions.append("Try adding a curiosity trigger such as '?' or a strong contrast.")
    if power_score == 0:
        suggestions.append("Use at least one strong power word (e.g., proven, hidden, breakthrough).")
    if keyword_score == 0 and keyword_hints:
        suggestions.append(f"Include one of your high-opportunity keywords: {', '.join(keyword_hints[:5])}.")

    parts = {
        "Length": int(length_score),
        "Clarity": int(clarity_score),
        "Numbers": int(number_score),
        "Curiosity": int(curiosity_score),
        "Power Words": int(power_score),
        "Keyword Match": int(keyword_score),
    }
    return total, parts, suggestions


def _description_score(description: str, keyword_hints: List[str]) -> Tuple[int, Dict[str, int], List[str]]:
    text = description.strip()
    lower = text.lower()
    length = len(text)

    length_score = 30 if 400 <= length <= 1800 else max(6, 30 - int(abs(length - 900) / 80))
    cta_score = 20 if any(x in lower for x in ["subscribe", "comment", "watch", "join", "follow", "link"]) else 5
    keyword_matches = sum(1 for k in keyword_hints[:15] if k in lower)
    keyword_score = min(25, keyword_matches * 4)
    structure_score = 15 if "\n" in text else 8
    hashtags = re.findall(r"#\w+", text)
    hashtag_score = 10 if 1 <= len(hashtags) <= 3 else 4

    total = max(0, min(100, length_score + cta_score + keyword_score + structure_score + hashtag_score))

    tips = []
    if length < 300:
        tips.append("Description is short; add context, value bullets, and timestamps if possible.")
    if cta_score < 20:
        tips.append("Add a clear CTA (subscribe, watch next, comment).")
    if keyword_score < 8 and keyword_hints:
        tips.append(f"Add relevant keywords: {', '.join(keyword_hints[:5])}.")
    if hashtag_score < 10:
        tips.append("Use 1-3 relevant hashtags, not more.")

    parts = {
        "Length": int(length_score),
        "CTA": int(cta_score),
        "Keywords": int(keyword_score),
        "Structure": int(structure_score),
        "Hashtags": int(hashtag_score),
    }
    return total, parts, tips


def _score_status(score: int) -> Tuple[str, str, str]:
    if score >= 80:
        return "Strong", "rgba(66, 200, 121, 0.16)", "#42C879"
    if score >= 60:
        return "Competitive", "rgba(255, 170, 60, 0.14)", "#FFB34D"
    return "Needs Work", "rgba(168, 85, 247, 0.16)", "#C4B5FD"


def _render_score_card(label: str, score: int, note: str) -> None:
    status_label, status_bg, status_color = _score_status(score)
    st.markdown(
        f"""
        <div class="ytuber-score-card">
            <div class="ytuber-score-label">{escape(label)}</div>
            <div class="ytuber-score-head">
                <div class="ytuber-score-value">{score}</div>
                <div class="ytuber-score-pill" style="background:{status_bg};color:{status_color};">{status_label}</div>
            </div>
            <div class="ytuber-score-bar"><span style="width:{max(0, min(score, 100))}%;"></span></div>
            <div class="ytuber-score-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_score_parts(parts: Dict[str, int]) -> None:
    entries = list(parts.items())
    cols = st.columns(3)
    for idx, (label, value) in enumerate(entries):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="ytuber-part-card">
                    <div class="ytuber-part-label">{escape(label)}</div>
                    <div class="ytuber-part-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _queue_ai_studio_prefill(task: str, brief: str) -> None:
    st.session_state["ytuber_ai_task_pending"] = task
    st.session_state["ytuber_ai_brief_pending"] = brief
    st.session_state["ytuber_active_module_pending"] = "AI Studio"
    st.session_state["ytuber_ai_notice"] = f"AI Studio is prefilled for {task.lower()}. Open the AI Studio tab."
    st.rerun()


def _queue_outlier_finder_page(prefill_query: str, note: str) -> None:
    st.session_state["app_page"] = "Outlier Finder"
    st.session_state["outlier_page_query"] = prefill_query.strip()
    st.session_state["outlier_page_prefill_note"] = note
    st.rerun()


def _render_outliers_shortcut(channel_df: pd.DataFrame, channel_title: str) -> None:
    section_header("Outlier Finder", icon="🚀")
    keyword_hints = _top_keywords(channel_df, 8)
    suggested_query = " ".join(keyword_hints[:3]).strip() or channel_title.strip()

    st.markdown(
        """
        <div class="yt-card" style="padding:1.15rem 1.2rem;">
            <div style="font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#B8C1DA;margin-bottom:0.3rem;">Standalone Research Tool</div>
            <div style="font-size:24px;font-weight:800;color:#FFFFFF;line-height:1.15;margin-bottom:0.45rem;">
                Discover Breakout Videos In A Dedicated Research Workspace.
            </div>
            <div style="font-size:14px;color:#C8D1EA;max-width:760px;">
                Search any niche, tighten the filters, and move into a standalone page built for result-first scanning, breakout snapshots, structured AI research, and methodology notes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if keyword_hints:
        st.markdown("**Suggested starter query from this channel**")
        styled_keyword_chips(keyword_hints[:6])
        st.caption(f"Recommended niche seed: `{suggested_query}`")

    action_cols = st.columns([1.3, 1.1])
    with action_cols[0]:
        if st.button("Open Outlier Finder", type="primary", use_container_width=True):
            _queue_outlier_finder_page(
                prefill_query=suggested_query,
                note=f"Prefilled from {channel_title}'s strongest recurring keywords. Adjust the niche query before running the scan.",
            )
    with action_cols[1]:
        if st.button("Open With Current Channel Name", use_container_width=True):
            _queue_outlier_finder_page(
                prefill_query=channel_title,
                note=f"Prefilled with the current channel name: {channel_title}. Replace it with a niche phrase if you want broader research.",
            )

    st.markdown(
        """
        <div class="yt-card" style="padding:0.9rem 1rem;">
            <div style="font-size:13px;color:#FFFFFF;font-weight:700;margin-bottom:0.35rem;">Why the standalone page is better for this job</div>
            <div style="font-size:13px;color:#C4CEE6;line-height:1.55;">
                It puts results first, then the breakout pattern snapshot, then structured AI research. It also supports stricter language filtering, exact versus broad matching, subscriber ranges, and cleaner niche-scanning controls.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _compute_channel_audit(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ordered = df.sort_values("video_publishedAt").copy()

    out["videos"] = len(ordered)
    out["median_views"] = float(ordered["views"].median()) if not ordered.empty else 0
    out["avg_engagement"] = float(ordered["engagement_rate"].mean()) if not ordered.empty else 0
    out["shorts_ratio"] = float(ordered["is_short"].mean()) if not ordered.empty else 0

    if len(ordered) > 2:
        gaps = ordered["video_publishedAt"].diff().dt.total_seconds().dropna() / 86400
        mean_gap = float(gaps.mean()) if not gaps.empty else 0
        std_gap = float(gaps.std()) if not gaps.empty else 0
        consistency = max(0.0, 100 - (std_gap / max(mean_gap, 1)) * 40)
    else:
        mean_gap = 0
        consistency = 0

    out["avg_upload_gap_days"] = mean_gap
    out["consistency_score"] = float(consistency)

    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    previous_cutoff = datetime.now(timezone.utc) - timedelta(days=180)
    recent = ordered[ordered["video_publishedAt"] >= recent_cutoff]
    previous = ordered[(ordered["video_publishedAt"] < recent_cutoff) & (ordered["video_publishedAt"] >= previous_cutoff)]

    recent_avg = float(recent["views"].mean()) if not recent.empty else 0
    previous_avg = float(previous["views"].mean()) if not previous.empty else 0
    growth = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
    out["view_growth_90d_pct"] = growth

    threshold = max(float(ordered["views"].median()) * 2.0, 1.0)
    out["outlier_rate"] = float((ordered["views"] >= threshold).mean()) if not ordered.empty else 0

    return out


def _fetch_or_get_cached_channel(
    channel_query: str,
    force_refresh: bool,
    youtube_api_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str, str]:
    workspace = load_public_channel_workspace(
        channel_query=channel_query,
        force_refresh=force_refresh,
        youtube_api_key=youtube_api_key,
    )
    return workspace.channel_df, workspace.source, workspace.channel_id, workspace.channel_title


def _gemini_generate_text(gemini_key: str, model: str, prompt: str) -> str:
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={gemini_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, json=payload, timeout=90)
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini text API error ({response.status_code}): {response.text[:500]}")

    body = response.json()
    texts: List[str] = []
    for candidate in body.get("candidates", []):
        for part in _safe_get(candidate, ["content", "parts"], []) or []:
            txt = part.get("text")
            if txt:
                texts.append(txt)

    if not texts:
        raise RuntimeError("Gemini did not return text output.")
    return "\n\n".join(texts)


def _openai_generate_text(openai_key: str, model: str, prompt: str) -> str:
    """Call OpenAI chat completions to generate strategy text."""
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an advanced YouTube strategist supporting the "
                    "YouTube IP V4 creator analytics platform. "
                    "Keep outputs concise, structured, and actionable."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI text API error ({response.status_code}): {response.text[:500]}"
        )
    body = response.json()
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI did not return any choices.")
    message = choices[0].get("message", {}) or {}
    content = message.get("content") or ""
    if not content:
        raise RuntimeError("OpenAI returned an empty message content.")
    return content


def _render_overview(channel_df: pd.DataFrame) -> None:
    section_header("Channel Overview", icon="📈")

    total_videos = len(channel_df)
    total_views = int(channel_df["views"].fillna(0).sum())
    total_likes = int(channel_df["likes"].fillna(0).sum())
    total_comments = int(channel_df["comments"].fillna(0).sum())
    avg_views = int(channel_df["views"].fillna(0).mean())
    med_eng = channel_df["engagement_rate"].median() * 100

    kpi_row(
        [
            {"label": "Videos (1Y)", "value": f"{total_videos:,}", "icon": "🎬"},
            {"label": "Total Views", "value": f"{total_views:,}", "icon": "👁️"},
            {"label": "Total Likes", "value": f"{total_likes:,}", "icon": "❤️"},
            {"label": "Total Comments", "value": f"{total_comments:,}", "icon": "💬"},
            {"label": "Avg Views / Video", "value": f"{avg_views:,}", "icon": "📊"},
            {"label": "Median Engagement", "value": f"{med_eng:.2f} %", "icon": "💡"},
        ]
    )

    left, right = st.columns(2)

    with left:
        section_header("Monthly Video + Views Trend", icon="📆")
        trend = (
            channel_df.groupby("publish_month")
            .agg(videos=("video_id", "count"), views=("views", "sum"))
            .reset_index()
            .sort_values("publish_month")
        )
        fig = plotly_line_chart(
            trend,
            x="publish_month",
            y_cols=["videos", "views"],
            title="Uploads and Views",
            secondary_y=["views"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_header("Top 12 Videos", icon="⭐")
        top_videos = channel_df[
            [
                "video_title",
                "views",
                "likes",
                "comments",
                "engagement_rate",
                "video_publishedAt",
                "video_id",
            ]
        ].sort_values("views", ascending=False).head(12)
        styled_dataframe(top_videos, title=None, precision=2)


def _render_channel_audit(channel_df: pd.DataFrame) -> None:
    section_header("Channel Audit", icon="🧪")
    audit = _compute_channel_audit(channel_df)

    kpi_row(
        [
            {
                "label": "Consistency Score",
                "value": f"{audit['consistency_score']:.1f}/100",
                "icon": "📆",
            },
            {
                "label": "Avg Upload Gap",
                "value": f"{audit['avg_upload_gap_days']:.1f} days",
                "icon": "⏱️",
            },
            {
                "label": "90d View Growth",
                "value": f"{audit['view_growth_90d_pct']:.1f}%",
                "icon": "📈",
            },
            {
                "label": "Outlier Rate",
                "value": f"{audit['outlier_rate'] * 100:.1f}%",
                "icon": "✨",
            },
        ]
    )

    st.markdown("**Audit Notes**")
    notes = []
    if audit["consistency_score"] < 45:
        notes.append("Upload cadence is inconsistent. Use a fixed weekly posting pattern.")
    if audit["view_growth_90d_pct"] < 0:
        notes.append("Views are down vs previous 90 days. Test stronger hooks and tighter titles.")
    if audit["outlier_rate"] < 0.08:
        notes.append("Few breakout videos detected. Increase experimentation with bold concepts.")
    if audit["shorts_ratio"] > 0.7:
        notes.append("Channel is heavily shorts-weighted; blend long-form to deepen session time.")
    if not notes:
        notes.append("Performance is stable. Focus on scaling repeatable winning formats.")

    for item in notes:
        st.markdown(
            f"""
            <div class="yt-card" style="padding:0.5rem 0.75rem;margin-bottom:0.35rem;">
                <span style="font-size:12px;color:#FFFFFF;">{item}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    trend_df = _build_trend_radar_df(channel_df)
    if not trend_df.empty:
        st.markdown("**Momentum Radar**")
        trend_cols = st.columns(2)
        with trend_cols[0]:
            styled_dataframe(trend_df.head(10), title=None, precision=1)
        with trend_cols[1]:
            rising = trend_df[trend_df["momentum_delta"] > 0].head(12)
            if not rising.empty:
                rising_fig = plotly_bar_chart(
                    rising.sort_values("momentum_delta", ascending=True),
                    x="keyword",
                    y="momentum_delta",
                    title="Rising Topics in the Last 60 Days",
                    horizontal=True,
                )
                st.plotly_chart(rising_fig, use_container_width=True)


def _render_keyword_intel(channel_df: pd.DataFrame) -> List[str]:
    section_header("Keyword Intelligence", icon="🔑")
    intel = _keyword_intel(channel_df)
    if intel.empty:
        st.info("Not enough text data to compute keyword insights.")
        return []

    styled_dataframe(intel, title=None, precision=2)

    top10 = intel.head(10)["keyword"].tolist()
    st.markdown("**High-opportunity keywords:**")
    styled_keyword_chips(top10)

    # Treemap and bar chart views
    if "score" in intel.columns:
        tree_fig = plotly_treemap(
            intel.head(40),
            path=["keyword"],
            values="score",
            title="Keyword Opportunity Treemap",
        )
        st.plotly_chart(tree_fig, use_container_width=True)

        bar_fig = plotly_bar_chart(
            intel.head(20).sort_values("score", ascending=False),
            x="keyword",
            y="score",
            title="Top Keyword Opportunities",
            horizontal=True,
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    return intel["keyword"].tolist()


def _timeframe_to_window(
    timeframe_label: str,
    custom_dates: Optional[Tuple[datetime, datetime]] = None,
) -> Tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if timeframe_label == "Last 7 Days":
        return now - timedelta(days=7), now
    if timeframe_label == "Last 30 Days":
        return now - timedelta(days=30), now
    if timeframe_label == "Last 90 Days":
        return now - timedelta(days=90), now
    if not custom_dates or len(custom_dates) != 2:
        raise ValueError("Choose both start and end dates for a custom timeframe.")
    start_date, end_date = custom_dates
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
    return start_dt, end_dt


def _format_int_label(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}"


def _format_subscriber_label(
    subscriber_count: Optional[float],
    hidden_subscriber_count: bool,
) -> str:
    if hidden_subscriber_count or subscriber_count is None or pd.isna(subscriber_count):
        return "Hidden"
    value = int(float(subscriber_count))
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value}"


def _render_outlier_cards(result_frame: pd.DataFrame) -> None:
    if result_frame.empty:
        return

    cols = st.columns(3)
    for idx, row in result_frame.head(6).iterrows():
        reasons = [
            escape(str(row.get("explanation_1", ""))),
            escape(str(row.get("explanation_2", ""))),
            escape(str(row.get("explanation_3", ""))),
        ]
        reasons_html = "".join(
            f"<li>{reason}</li>" for reason in reasons if reason.strip()
        )
        thumb_html = (
            f'<img src="{escape(str(row.get("thumbnail_url", "")))}" alt="{escape(str(row.get("video_title", "")))}" />'
            if str(row.get("thumbnail_url", "")).strip()
            else ""
        )
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="outlier-card">
                    {thumb_html}
                    <div class="outlier-card-body">
                        <div class="outlier-card-top">
                            <div>
                                <div class="outlier-card-title">{escape(str(row.get("video_title", "")))}</div>
                                <div class="outlier-card-channel">{escape(str(row.get("channel_title", "")))}</div>
                            </div>
                            <div class="outlier-score-badge">
                                <div class="outlier-score-value">{float(row.get("outlier_score", 0)):.1f}</div>
                                <div class="outlier-score-label">Score</div>
                            </div>
                        </div>
                        <div class="outlier-metrics">
                            <span class="outlier-metric-pill">{_format_int_label(row.get("views"))} views</span>
                            <span class="outlier-metric-pill">{_format_int_label(row.get("views_per_day"))} / day</span>
                            <span class="outlier-metric-pill">{float(row.get("engagement_rate", 0)) * 100:.2f}% engagement</span>
                            <span class="outlier-metric-pill">{_format_subscriber_label(row.get("channel_subscriber_count"), bool(row.get("hidden_subscriber_count")))} subs</span>
                            <span class="outlier-metric-pill">{float(row.get("age_days", 0)):.1f} days old</span>
                        </div>
                        <ul class="outlier-reasons">{reasons_html}</ul>
                        <a class="outlier-link" href="{escape(str(row.get("video_url", "")))}" target="_blank">Open on YouTube</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_outlier_prompt_rows(result_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    columns = [
        "video_title",
        "channel_title",
        "outlier_score",
        "views",
        "views_per_day",
        "engagement_rate",
        "size_bucket",
        "age_days",
        "explanation_text",
    ]
    return result_frame.head(12)[columns].to_dict(orient="records")


def _render_outliers_ai_panel(result_frame: pd.DataFrame, current_channel_title: str) -> None:
    available_text_providers = [
        provider for provider in ["gemini", "openai"] if get_provider_key_count(provider) > 0
    ]
    if result_frame.empty:
        return
    if not available_text_providers:
        st.caption("Add `GEMINI_API_KEYS` and/or `OPENAI_API_KEYS` to unlock AI summaries for Outliers Finder.")
        return

    if (
        "ytuber_outliers_ai_provider" not in st.session_state
        or st.session_state["ytuber_outliers_ai_provider"] not in available_text_providers
    ):
        st.session_state["ytuber_outliers_ai_provider"] = (
            "gemini" if "gemini" in available_text_providers else available_text_providers[0]
        )

    provider = st.selectbox(
        "AI provider",
        available_text_providers,
        key="ytuber_outliers_ai_provider",
        format_func=lambda value: PROVIDER_LABELS.get(value, value.title()),
    )
    model_options = [item["id"] for item in TEXT_MODEL_CATALOG[provider]]
    model_key = "ytuber_outliers_ai_model"
    if model_key not in st.session_state or st.session_state[model_key] not in model_options:
        st.session_state[model_key] = model_options[0]
    model = st.selectbox(
        "Summary model",
        model_options,
        key=model_key,
        format_func=lambda value: _format_text_model_option(provider, value),
    )

    source_rows = _build_outlier_prompt_rows(result_frame)
    button_cols = st.columns(2)
    if button_cols[0].button("Summarize Outlier Patterns", use_container_width=True):
        prompt = (
            "You are a YouTube strategist analyzing a scanned cohort of public outlier videos.\n\n"
            f"Creator context: {current_channel_title or 'General creator'}\n"
            f"Outlier rows: {source_rows}\n\n"
            "Return:\n"
            "1. What patterns repeat across the strongest outliers\n"
            "2. Which hooks, formats, or packaging choices appear most often\n"
            "3. What the creator should pay attention to next"
        )
        with st.spinner("Generating outlier summary..."):
            try:
                st.session_state["ytuber_outliers_summary_output"] = _generate_text_with_provider_pool(
                    provider,
                    model,
                    prompt,
                )
            except Exception as exc:
                st.error(f"AI summary failed: {exc}")

    if button_cols[1].button("Generate Content Angles", use_container_width=True):
        prompt = (
            "You are a YouTube strategist turning public outlier findings into actionable content ideas.\n\n"
            f"Creator context: {current_channel_title or 'General creator'}\n"
            f"Outlier rows: {source_rows}\n\n"
            "Return:\n"
            "1. 8 content angles inspired by the outliers but not copied\n"
            "2. 10 title seeds\n"
            "3. A short note on how to differentiate from the observed winners"
        )
        with st.spinner("Generating content angles..."):
            try:
                st.session_state["ytuber_outliers_angles_output"] = _generate_text_with_provider_pool(
                    provider,
                    model,
                    prompt,
                )
            except Exception as exc:
                st.error(f"AI angle generation failed: {exc}")

    summary_output = st.session_state.get("ytuber_outliers_summary_output", "")
    if summary_output:
        st.markdown("**AI Pattern Summary**")
        st.markdown(summary_output)

    angle_output = st.session_state.get("ytuber_outliers_angles_output", "")
    if angle_output:
        st.markdown("**AI Content Angles**")
        st.markdown(angle_output)


def _render_outliers_finder(current_channel_title: str) -> None:
    section_header("Outliers Finder", icon="🚀")
    st.caption(
        "Search a niche cohort with the official YouTube API and rank videos by public overperformance versus channel baseline and peer context."
    )

    with st.form("ytuber_outliers_form"):
        niche_query = st.text_input(
            "Niche or keyword",
            key="ytuber_outliers_query",
            placeholder="AI automation, fitness for busy professionals, tech explainer, science shorts...",
        )
        primary_cols = st.columns(4)
        with primary_cols[0]:
            timeframe = st.selectbox(
                "Timeframe",
                OUTLIER_TIMEFRAME_OPTIONS,
                index=1,
                key="ytuber_outliers_timeframe",
            )
        with primary_cols[1]:
            region_code = st.selectbox(
                "Geography",
                OUTLIER_REGION_OPTIONS,
                index=0,
                key="ytuber_outliers_region",
            )
        with primary_cols[2]:
            language_code = st.selectbox(
                "Language",
                OUTLIER_LANGUAGE_OPTIONS,
                index=0,
                key="ytuber_outliers_language",
            )
        with primary_cols[3]:
            subscriber_bucket = st.selectbox(
                "Channel size",
                list(SUBSCRIBER_BUCKETS.keys()),
                index=0,
                key="ytuber_outliers_subscriber_bucket",
            )

        custom_dates = None
        if timeframe == "Custom":
            default_end = datetime.now(timezone.utc).date()
            default_start = default_end - timedelta(days=30)
            custom_dates = st.date_input(
                "Custom range",
                value=(default_start, default_end),
                max_value=default_end,
                key="ytuber_outliers_custom_dates",
            )

        secondary_cols = st.columns([1.1, 1.4, 1.1])
        with secondary_cols[0]:
            include_hidden_subscribers = st.toggle(
                "Include hidden subscriber counts",
                value=True,
                key="ytuber_outliers_include_hidden",
            )
        with secondary_cols[1]:
            st.markdown(
                "<div class='ytuber-toolbar-note'>Results are outliers inside the scanned cohort returned by the YouTube API, not exhaustive rankings across all of YouTube.</div>",
                unsafe_allow_html=True,
            )
        with secondary_cols[2]:
            submitted = st.form_submit_button(
                "Run Outlier Scan",
                type="primary",
                use_container_width=True,
                disabled=get_provider_key_count("youtube") <= 0,
            )

        with st.expander("Advanced Search Settings", expanded=False):
            advanced_cols = st.columns(3)
            with advanced_cols[0]:
                search_pages = st.slider(
                    "Search pages",
                    min_value=2,
                    max_value=4,
                    value=2,
                    step=1,
                    key="ytuber_outliers_search_pages",
                    help="Each extra page can add about 100 YouTube quota units.",
                )
            with advanced_cols[1]:
                baseline_channel_limit = st.slider(
                    "Baseline channels",
                    min_value=10,
                    max_value=20,
                    value=15,
                    step=5,
                    key="ytuber_outliers_baseline_channels",
                )
            with advanced_cols[2]:
                baseline_video_cap = st.slider(
                    "Baseline uploads per channel",
                    min_value=10,
                    max_value=30,
                    value=20,
                    step=5,
                    key="ytuber_outliers_baseline_videos",
                )

    if submitted:
        if not niche_query.strip():
            st.session_state["ytuber_outliers_error"] = "Enter a niche, topic, or keyword before running the scan."
            st.session_state.pop("ytuber_outliers_result", None)
        else:
            try:
                published_after, published_before = _timeframe_to_window(
                    timeframe,
                    custom_dates=tuple(custom_dates) if timeframe == "Custom" and custom_dates else None,
                )
                if (published_before - published_after).days > 180:
                    raise ValueError("Custom timeframe cannot exceed 180 days in V1.")
                request = OutlierSearchRequest(
                    niche_query=niche_query.strip(),
                    published_after_iso=published_after.isoformat(),
                    published_before_iso=published_before.isoformat(),
                    region_code="" if region_code == "Any" else region_code,
                    relevance_language="" if language_code == "Any" else language_code,
                    subscriber_bucket=subscriber_bucket,
                    include_hidden_subscribers=include_hidden_subscribers,
                    max_results=int(search_pages * 50),
                    baseline_channel_limit=baseline_channel_limit,
                    baseline_video_cap=baseline_video_cap,
                )
                with st.spinner("Scanning the niche cohort and scoring outliers..."):
                    result = search_outlier_videos(request)
                st.session_state["ytuber_outliers_result"] = result
                st.session_state.pop("ytuber_outliers_error", None)
                st.session_state.pop("ytuber_outliers_summary_output", None)
                st.session_state.pop("ytuber_outliers_angles_output", None)
            except Exception as exc:
                st.session_state["ytuber_outliers_error"] = str(exc)
                st.session_state.pop("ytuber_outliers_result", None)

    error_message = st.session_state.get("ytuber_outliers_error", "")
    if error_message:
        st.error(error_message)

    result = st.session_state.get("ytuber_outliers_result")
    if not result:
        st.info("Run a niche scan to surface overperforming videos in the selected cohort.")
        return

    for warning in result.warnings:
        st.warning(warning)

    result_frame = result.to_frame()
    if result_frame.empty:
        st.info("No strong matches were found in the scanned cohort. Broaden the timeframe or loosen the filters.")
        return

    result_frame["published_at"] = pd.to_datetime(result_frame["published_at_iso"], errors="coerce", utc=True)
    result_frame["engagement_pct"] = result_frame["engagement_rate"] * 100
    result_frame["log10_subscribers"] = result_frame["channel_subscriber_count"].fillna(0).apply(
        lambda value: math.log10(float(value) + 1)
    )

    kpi_row(
        [
            {"label": "Videos Scanned", "value": f"{result.scanned_videos:,}", "icon": "🎬"},
            {"label": "Channels Scanned", "value": f"{result.scanned_channels:,}", "icon": "📺"},
            {"label": "Channel Baselines", "value": f"{result.baseline_channels:,}", "icon": "📉"},
            {"label": "Cache Policy", "value": result.cache_policy, "icon": "🧠"},
            {"label": "Quota Mode", "value": result.quota_profile, "icon": "⚡"},
        ]
    )

    sort_label = st.selectbox(
        "Sort results by",
        list(OUTLIER_SORT_OPTIONS.keys()),
        index=0,
        key="ytuber_outliers_sort",
    )
    sort_column = OUTLIER_SORT_OPTIONS[sort_label]
    sorted_frame = result_frame.sort_values(sort_column, ascending=False).reset_index(drop=True)

    st.markdown("**Top Outlier Videos**")
    _render_outlier_cards(sorted_frame)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        scatter_fig = plotly_scatter(
            sorted_frame,
            x="log10_subscribers",
            y="outlier_score",
            size="views",
            color="age_bucket",
            title="Outlier Score vs Channel Size (log10 subscribers + 1)",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
    with chart_cols[1]:
        channel_breakout = (
            sorted_frame.groupby("channel_title", dropna=False)
            .agg(outliers=("video_id", "count"))
            .reset_index()
            .sort_values("outliers", ascending=False)
            .head(12)
        )
        channel_fig = plotly_bar_chart(
            channel_breakout.sort_values("outliers", ascending=True),
            x="channel_title",
            y="outliers",
            title="Channels Producing the Most Outliers",
            horizontal=True,
        )
        st.plotly_chart(channel_fig, use_container_width=True)

    keyword_counter: Counter = Counter()
    for title in sorted_frame.head(20)["video_title"].tolist():
        keyword_counter.update(_tokenize(title))
    if keyword_counter:
        st.markdown("**Repeated Title Keywords Across The Outliers**")
        styled_keyword_chips([keyword for keyword, _ in keyword_counter.most_common(10)])

    table_df = sorted_frame[
        [
            "thumbnail_url",
            "video_title",
            "channel_title",
            "outlier_score",
            "views",
            "views_per_day",
            "engagement_pct",
            "channel_subscriber_count",
            "age_days",
            "explanation_text",
        ]
    ].copy()
    table_df.rename(
        columns={
            "thumbnail_url": "Thumbnail",
            "video_title": "Title",
            "channel_title": "Channel",
            "outlier_score": "Outlier Score",
            "views": "Views",
            "views_per_day": "Views / Day",
            "engagement_pct": "Engagement %",
            "channel_subscriber_count": "Subscribers",
            "age_days": "Age (Days)",
            "explanation_text": "Why It Is An Outlier",
        },
        inplace=True,
    )
    styled_dataframe(
        table_df,
        title="Scanned Cohort Results",
        precision=2,
        image_columns=["Thumbnail"],
    )

    st.markdown("**AI Layer**")
    _render_outliers_ai_panel(sorted_frame, current_channel_title)


def _render_title_seo_lab(keyword_hints: List[str]) -> None:
    section_header("Title & SEO Lab", icon="🧪")
    st.caption("Scores update as the title or description changes. Use this editor to pressure-test hooks before you generate more ideas.")
    test_title = st.text_input("Test title", value="The Hidden Physics Trick That Changes Everything")
    test_desc = st.text_area(
        "Test description",
        value="In this video we break down the science, show real examples, and explain how to apply this idea. Subscribe for more! #science #learning",
        height=120,
    )

    title_score, parts, tips = _title_score(test_title, keyword_hints)
    desc_score, desc_parts, desc_tips = _description_score(test_desc, keyword_hints)

    summary_cols = st.columns(2)
    with summary_cols[0]:
        _render_score_card(
            "Title Score",
            title_score,
            "Higher scores usually mean stronger hook clarity, keyword fit, and click potential.",
        )
    with summary_cols[1]:
        _render_score_card(
            "Description Score",
            desc_score,
            "Descriptions should support search relevance, next-step CTAs, and on-page structure.",
        )

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Title Breakdown**")
        _render_score_parts(parts)
        if tips:
            st.markdown("**Title Recommendations**")
            for tip in tips:
                st.markdown(f"- {tip}")
    with detail_cols[1]:
        st.markdown("**Description Breakdown**")
        _render_score_parts(desc_parts)
        if desc_tips:
            st.markdown("**Description Recommendations**")
            for tip in desc_tips:
                st.markdown(f"- {tip}")

    if keyword_hints:
        st.markdown("**Priority Keywords To Weave In**")
        styled_keyword_chips(keyword_hints[:8])

    action_cols = st.columns([1.2, 1.2, 1.6])
    with action_cols[0]:
        if st.button("Prepare Better Titles In AI Studio", use_container_width=True, key="ytuber_title_to_ai"):
            _queue_ai_studio_prefill(
                "Titles Only",
                f"Improve this title direction for stronger CTR and search intent: {test_title}. "
                f"Description context: {test_desc[:400]}",
            )
    with action_cols[1]:
        if st.button("Prepare Video Ideas In AI Studio", use_container_width=True, key="ytuber_ideas_to_ai"):
            _queue_ai_studio_prefill(
                "Video Ideas",
                f"Use this as the seed concept and expand into new video ideas: {test_title}. "
                f"Description context: {test_desc[:400]}",
            )
    with action_cols[2]:
        st.markdown(
            "<div style='font-size:12px;color:#AEB8D6;padding-top:0.5rem;'>Use AI Studio when the title direction is weak, the channel needs alternate angles, or you want batch ideas from the same concept.</div>",
            unsafe_allow_html=True,
        )


def _render_competitor_benchmark(
    channel_df: pd.DataFrame,
    channel_title: str,
    channel_id: str,
) -> None:
    section_header("Competitor Benchmark", icon="📊")
    handles = st.text_area(
        "Competitor handles (comma separated)",
        value="@3blue1brown,@veritasium,@smartereveryday",
        height=90,
    )

    run = st.button("Run Competitor Benchmark", use_container_width=True)
    state_key = f"ytuber_competitor_state_{channel_id}"

    if run:
        if get_provider_key_count("youtube") <= 0:
            st.error("No YouTube API keys are configured for competitor benchmarking.")
            return

        competitors = [h.strip() for h in handles.split(",") if h.strip()]
        rows = []
        competitor_frames: List[pd.DataFrame] = []
        own_keywords = set(_top_keywords(channel_df, 20))
        keyword_gap_counter: Counter = Counter()

        with st.spinner("Loading competitor channels..."):
            for handle in competitors:
                try:
                    cdf, source, cid, title = _fetch_or_get_cached_channel(
                        handle,
                        force_refresh=False,
                    )
                    cdf = _ensure_numeric_and_dates(cdf)
                    competitor_frames.append(cdf.assign(_benchmark_channel=title))
                    competitor_keywords = _top_keywords(cdf, 12)
                    for kw in competitor_keywords:
                        if kw not in own_keywords:
                            keyword_gap_counter[kw] += 1

                    best_day = "N/A"
                    if not cdf.empty and cdf["publish_day"].notna().any():
                        best_day = (
                            cdf.groupby("publish_day")["views"]
                            .mean()
                            .sort_values(ascending=False)
                            .index[0]
                        )

                    rows.append(
                        {
                            "handle": handle,
                            "channel_title": title,
                            "channel_id": cid,
                            "source": source,
                            "videos_1y": len(cdf),
                            "total_views": int(cdf["views"].fillna(0).sum()) if not cdf.empty else 0,
                            "avg_views": int(cdf["views"].fillna(0).mean()) if not cdf.empty else 0,
                            "median_engagement": float(cdf["engagement_rate"].median()) if not cdf.empty else 0.0,
                            "best_day": best_day,
                            "top_topics": ", ".join(competitor_keywords[:4]),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "handle": handle,
                            "channel_title": "ERROR",
                            "channel_id": "",
                            "source": "error",
                            "videos_1y": 0,
                            "total_views": 0,
                            "avg_views": 0,
                            "median_engagement": 0.0,
                            "best_day": "N/A",
                            "top_topics": "",
                            "error": str(exc),
                        }
                    )

        if not rows:
            st.warning("No competitor data produced.")
            return

        bdf = pd.DataFrame(rows).sort_values("total_views", ascending=False)
        combined_competitors = (
            pd.concat(competitor_frames, ignore_index=True)
            if competitor_frames
            else pd.DataFrame()
        )
        competitor_trend_df = _build_trend_radar_df(combined_competitors)
        keyword_gap_df = pd.DataFrame(
            [
                {"keyword": keyword, "competitor_count": count}
                for keyword, count in keyword_gap_counter.most_common(15)
            ]
        )
        own_stats = {
            "videos_1y": len(channel_df),
            "total_views": int(channel_df["views"].fillna(0).sum()),
            "avg_views": int(channel_df["views"].fillna(0).mean()),
            "median_engagement": float(channel_df["engagement_rate"].median()),
        }

        insights = _generate_competitor_recommendations(
            channel_title,
            own_stats,
            bdf,
            competitor_trend_df,
            keyword_gap_df,
        )

        st.session_state[state_key] = {
            "benchmark_rows": bdf.to_dict(orient="records"),
            "trend_rows": competitor_trend_df.to_dict(orient="records"),
            "gap_rows": keyword_gap_df.to_dict(orient="records"),
            "insights": insights,
        }

    if state_key not in st.session_state:
        st.caption("Enter competitor handles and run benchmark.")
        return

    state = st.session_state[state_key]
    bdf = pd.DataFrame(state.get("benchmark_rows", []))
    competitor_trend_df = pd.DataFrame(state.get("trend_rows", []))
    keyword_gap_df = pd.DataFrame(state.get("gap_rows", []))
    styled_dataframe(bdf, title=None, precision=1)

    if not bdf.empty:
        radar_series = {
            row["channel_title"]: [
                row["videos_1y"],
                row["total_views"],
                row["avg_views"],
                row["median_engagement"],
            ]
            for _, row in bdf.iterrows()
        }
        cats = ["Videos", "Total Views", "Avg Views", "Median Engagement"]
        radar_fig = plotly_radar_chart(cats, radar_series, "Competitor Shape")
        st.plotly_chart(radar_fig, use_container_width=True)

        bar_fig = plotly_bar_chart(
            bdf.head(10),
            x="channel_title",
            y="total_views",
            title="Total Views by Competitor",
            horizontal=True,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    if not competitor_trend_df.empty:
        st.markdown("**Competitor Trend Radar**")
        trend_cols = st.columns(2)
        with trend_cols[0]:
            styled_dataframe(competitor_trend_df.head(12), title=None, precision=1)
        with trend_cols[1]:
            positive = competitor_trend_df[competitor_trend_df["momentum_delta"] > 0].head(12)
            if not positive.empty:
                comp_trend_fig = plotly_bar_chart(
                    positive.sort_values("momentum_delta", ascending=True),
                    x="keyword",
                    y="momentum_delta",
                    title="Rising Competitor Topics",
                    horizontal=True,
                )
                st.plotly_chart(comp_trend_fig, use_container_width=True)

    if not keyword_gap_df.empty:
        st.markdown("**Keyword gaps vs competitors**")
        gap_fig = plotly_bar_chart(
            keyword_gap_df.sort_values("competitor_count", ascending=True).head(12),
            x="keyword",
            y="competitor_count",
            title="Keywords Competitors Use More Often",
            horizontal=True,
        )
        st.plotly_chart(gap_fig, use_container_width=True)
        styled_keyword_chips(keyword_gap_df["keyword"].head(8).tolist())

    insights = state.get("insights", "")
    if insights:
        st.markdown("**Competitive Recommendations**")
        st.markdown(insights)


def _render_trend_radar(channel_df: pd.DataFrame) -> None:
    section_header("Trend Radar", icon="📡")
    tdf = _build_trend_radar_df(channel_df)
    if tdf.empty:
        st.info("Not enough recent data for trend radar.")
        return
    styled_dataframe(tdf, title=None, precision=1)

    rising = tdf[tdf["momentum_delta"] > 0].copy()
    falling = tdf[tdf["momentum_delta"] <= 0].copy()

    if not rising.empty:
        rising_fig = plotly_bar_chart(
            rising.sort_values("momentum_delta", ascending=True),
            x="keyword",
            y="momentum_delta",
            title="Rising Keywords",
            horizontal=True,
        )
        st.plotly_chart(rising_fig, use_container_width=True)

    if not falling.empty:
        falling_fig = plotly_bar_chart(
            falling.sort_values("momentum_delta", ascending=True),
            x="keyword",
            y="momentum_delta",
            title="Falling Keywords",
            horizontal=True,
        )
        st.plotly_chart(falling_fig, use_container_width=True)


def _render_content_planner(channel_df: pd.DataFrame) -> None:
    section_header("Content Planner", icon="🗓️")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_perf = (
        channel_df.groupby("publish_day", dropna=False)
        .agg(
            avg_views=("views", "mean"),
            median_engagement=("engagement_rate", "median"),
            videos=("video_id", "count"),
            total_views=("views", "sum"),
        )
        .reset_index()
        .set_index("publish_day")
        .reindex(weekday_order)
        .dropna(how="all")
        .reset_index()
    )

    hour_perf = (
        channel_df.groupby("publish_hour", dropna=False)
        .agg(
            avg_views=("views", "mean"),
            median_engagement=("engagement_rate", "median"),
            videos=("video_id", "count"),
            total_views=("views", "sum"),
        )
        .reset_index()
        .sort_values("publish_hour")
    )

    best_day = (
        day_perf.sort_values("avg_views", ascending=False).iloc[0]["publish_day"]
        if not day_perf.empty
        else "Wednesday"
    )
    best_hour = (
        int(hour_perf.sort_values("avg_views", ascending=False).iloc[0]["publish_hour"])
        if not hour_perf.empty
        else 15
    )

    kpi_row(
        [
            {"label": "Best Publishing Day", "value": str(best_day), "icon": "📅"},
            {
                "label": "Best Publishing Hour (UTC)",
                "value": f"{best_hour:02d}:00",
                "icon": "⏰",
            },
        ]
    )

    day_cols = st.columns(2)
    with day_cols[0]:
        fig_day_views = plotly_bar_chart(
            day_perf,
            x="publish_day",
            y="avg_views",
            title="Average Views by Day",
        )
        st.plotly_chart(fig_day_views, use_container_width=True)
    with day_cols[1]:
        fig_day_uploads = plotly_bar_chart(
            day_perf,
            x="publish_day",
            y="videos",
            title="Uploads by Day",
        )
        st.plotly_chart(fig_day_uploads, use_container_width=True)

    hour_cols = st.columns(2)
    with hour_cols[0]:
        fig_hour_views = plotly_bar_chart(
            hour_perf,
            x="publish_hour",
            y="avg_views",
            title="Average Views by Hour (UTC)",
        )
        st.plotly_chart(fig_hour_views, use_container_width=True)
    with hour_cols[1]:
        fig_hour_uploads = plotly_bar_chart(
            hour_perf,
            x="publish_hour",
            y="videos",
            title="Uploads by Hour (UTC)",
        )
        st.plotly_chart(fig_hour_uploads, use_container_width=True)

    heatmap_source = (
        channel_df.groupby(["publish_day", "publish_hour"], dropna=False)
        .agg(avg_views=("views", "mean"), videos=("video_id", "count"))
        .reset_index()
    )
    heatmap_source["publish_day"] = pd.Categorical(
        heatmap_source["publish_day"],
        categories=weekday_order,
        ordered=True,
    )
    heatmap_source = heatmap_source.sort_values(["publish_day", "publish_hour"])

    heat_cols = st.columns(2)
    with heat_cols[0]:
        heat_views = plotly_heatmap(
            heatmap_source,
            x="publish_hour",
            y="publish_day",
            z="avg_views",
            title="Day x Hour Average Views",
        )
        st.plotly_chart(heat_views, use_container_width=True)
    with heat_cols[1]:
        heat_uploads = plotly_heatmap(
            heatmap_source,
            x="publish_hour",
            y="publish_day",
            z="videos",
            title="Day x Hour Upload Density",
        )
        st.plotly_chart(heat_uploads, use_container_width=True)

    top_topics = _top_keywords(channel_df, top_n=12)
    if top_topics:
        st.markdown("**Suggested next content angles**")
        styled_keyword_chips(top_topics[:8])

    weekday_map = {d: i for i, d in enumerate(weekday_order)}
    target_weekday = weekday_map.get(best_day, 2)

    start = datetime.now(timezone.utc)
    plan_rows = []
    cursor = start
    for i in range(1, 5):
        while cursor.weekday() != target_weekday:
            cursor += timedelta(days=1)
        plan_rows.append(
            {
                "week": f"Week {i}",
                "publish_date_utc": cursor.date().isoformat(),
                "publish_time_utc": f"{best_hour:02d}:00",
                "topic_hint": top_topics[(i - 1) % max(len(top_topics), 1)] if top_topics else "core topic",
            }
        )
        cursor += timedelta(days=7)

    st.markdown("**4-Week Suggested Calendar**")
    calendar_df = pd.DataFrame(plan_rows)
    cols = st.columns(4)
    for idx, row in calendar_df.iterrows():
        col = cols[idx % 4]
        with col:
            st.markdown(
                f"""
                <div class="yt-card" style="padding:0.6rem 0.75rem;margin-bottom:0.6rem;">
                    <div style="font-size:11px;color:#B0B0B0;">{row['week']}</div>
                    <div style="font-size:16px;font-weight:600;color:#FFFFFF;">{row['publish_date_utc']}</div>
                    <div style="font-size:12px;color:#B0B0B0;margin-bottom:0.15rem;">{row['publish_time_utc']} UTC</div>
                    <div style="font-size:12px;color:#FFFFFF;">
                        <span class="keyword-chip">{row['topic_hint']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_ai_studio(
    channel_df: pd.DataFrame,
    channel_title: str,
    channel_id: str,
    keyword_hints: List[str],
) -> None:
    section_header("AI Studio", icon="🤖")

    pending_task = st.session_state.pop("ytuber_ai_task_pending", None)
    if pending_task in AI_STUDIO_TASKS:
        st.session_state["ytuber_ai_task"] = pending_task

    pending_brief = st.session_state.pop("ytuber_ai_brief_pending", None)
    if pending_brief is not None:
        st.session_state["ytuber_creative_brief"] = pending_brief

    available_text_providers = [
        provider for provider in ["gemini", "openai"] if get_provider_key_count(provider) > 0
    ]
    available_image_providers = available_text_providers[:]

    st.caption("Choose the model mix, output counts, and creative filters before generating text or thumbnails.")
    notice = st.session_state.pop("ytuber_ai_notice", None)
    if notice:
        st.success(notice)

    if not available_text_providers and not available_image_providers:
        st.info("Add `GEMINI_API_KEYS` and/or `OPENAI_API_KEYS` in secrets to unlock AI Studio.")
        return

    if "ytuber_ai_task" not in st.session_state or st.session_state["ytuber_ai_task"] not in AI_STUDIO_TASKS:
        st.session_state["ytuber_ai_task"] = AI_STUDIO_TASKS[0]

    default_text_provider = "gemini" if "gemini" in available_text_providers else available_text_providers[0]
    default_image_provider = "gemini" if "gemini" in available_image_providers else available_image_providers[0]
    if "ytuber_text_provider" not in st.session_state or st.session_state["ytuber_text_provider"] not in available_text_providers:
        st.session_state["ytuber_text_provider"] = default_text_provider
    if "ytuber_image_provider" not in st.session_state or st.session_state["ytuber_image_provider"] not in available_image_providers:
        st.session_state["ytuber_image_provider"] = default_image_provider

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        text_provider = st.selectbox(
            "Text provider",
            available_text_providers,
            key="ytuber_text_provider",
            format_func=lambda value: PROVIDER_LABELS.get(value, value.title()),
        )
        text_model_options = [item["id"] for item in TEXT_MODEL_CATALOG[text_provider]]
        if (
            "ytuber_text_model_selection" not in st.session_state
            or st.session_state["ytuber_text_model_selection"] not in text_model_options
        ):
            st.session_state["ytuber_text_model_selection"] = text_model_options[0]
        text_model = st.selectbox(
            "Text model",
            text_model_options,
            key="ytuber_text_model_selection",
            format_func=lambda value: _format_text_model_option(text_provider, value),
        )
        text_model_meta = _catalog_map(TEXT_MODEL_CATALOG[text_provider])[text_model]
        st.caption(
            f"{text_model_meta['summary']}  •  "
            f"${text_model_meta['input_per_million']}/1M input tokens  •  "
            f"${text_model_meta['output_per_million']}/1M output tokens"
        )

    with control_col2:
        image_provider = st.selectbox(
            "Image provider",
            available_image_providers,
            key="ytuber_image_provider",
            format_func=lambda value: PROVIDER_LABELS.get(value, value.title()),
        )
        image_model_options = [item["id"] for item in IMAGE_MODEL_CATALOG[image_provider]]
        if (
            "ytuber_image_model_selection" not in st.session_state
            or st.session_state["ytuber_image_model_selection"] not in image_model_options
        ):
            st.session_state["ytuber_image_model_selection"] = image_model_options[0]
        image_model = st.selectbox(
            "Thumbnail model",
            image_model_options,
            key="ytuber_image_model_selection",
            format_func=lambda value: _format_image_model_option(image_provider, value),
        )
        image_model_meta = _catalog_map(IMAGE_MODEL_CATALOG[image_provider])[image_model]
        st.caption(image_model_meta["summary"])

    if (
        "ytuber_image_size" not in st.session_state
        or st.session_state["ytuber_image_size"] not in image_model_meta["size_options"]
    ):
        st.session_state["ytuber_image_size"] = image_model_meta["size_options"][0]
    if (
        "ytuber_image_quality" not in st.session_state
        or st.session_state["ytuber_image_quality"] not in image_model_meta["quality_options"]
    ):
        st.session_state["ytuber_image_quality"] = image_model_meta["quality_options"][0]

    image_tuning_col1, image_tuning_col2 = st.columns(2)
    with image_tuning_col1:
        image_size = st.selectbox(
            "Thumbnail size",
            image_model_meta["size_options"],
            key="ytuber_image_size",
            disabled=len(image_model_meta["size_options"]) == 1,
        )
    with image_tuning_col2:
        image_quality = st.selectbox(
            "Thumbnail quality",
            image_model_meta["quality_options"],
            key="ytuber_image_quality",
            disabled=len(image_model_meta["quality_options"]) == 1,
        )

    image_background = "opaque"
    image_output_format = "png"
    if image_provider == "openai":
        if (
            "ytuber_image_background" not in st.session_state
            or st.session_state["ytuber_image_background"] not in image_model_meta.get("background_options", ["opaque"])
        ):
            st.session_state["ytuber_image_background"] = image_model_meta.get("background_options", ["opaque"])[0]
        if (
            "ytuber_image_output_format" not in st.session_state
            or st.session_state["ytuber_image_output_format"] not in image_model_meta.get("format_options", ["png"])
        ):
            st.session_state["ytuber_image_output_format"] = image_model_meta.get("format_options", ["png"])[0]

        openai_image_col1, openai_image_col2 = st.columns(2)
        with openai_image_col1:
            image_background = st.selectbox(
                "Background",
                image_model_meta.get("background_options", ["opaque"]),
                key="ytuber_image_background",
            )
        with openai_image_col2:
            image_output_format = st.selectbox(
                "Output format",
                image_model_meta.get("format_options", ["png"]),
                key="ytuber_image_output_format",
            )

    quantity_col1, quantity_col2, quantity_col3 = st.columns(3)
    with quantity_col1:
        idea_count = st.slider("Video ideas", min_value=1, max_value=12, value=5)
    with quantity_col2:
        script_count = st.slider("Scripts / outlines", min_value=1, max_value=6, value=2)
    with quantity_col3:
        thumbnail_count = st.slider("Thumbnail options", min_value=1, max_value=6, value=3)

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        audience_profile = st.selectbox("Audience level", AUDIENCE_OPTIONS, index=0)
        format_focus = st.selectbox("Format focus", FORMAT_OPTIONS, index=0)
    with filter_col2:
        strategy_filters = st.multiselect(
            "Strategy filters",
            STRATEGY_FILTERS,
            default=["Evergreen", "High CTR"],
        )
        exclude_topics = st.text_input(
            "Exclude / avoid",
            value="",
            placeholder="Overused themes, competitors, claims to avoid",
        )

    prompt_goal = _goal_from_prompt(st.session_state.get("ytuber_growth_prompt", ""))
    default_brief = (
        f"Goal: {prompt_goal}. Channel: {channel_title}. Build a sharp plan based on the current stats."
        if prompt_goal
        else f"Channel: {channel_title}. Create a high-performing next-month content plan grounded in the current stats."
    )
    creative_brief = st.text_area(
        "Creative brief",
        value=default_brief,
        key="ytuber_creative_brief",
        height=110,
    )

    output_type = st.selectbox(
        "Creative task",
        AI_STUDIO_TASKS,
        key="ytuber_ai_task",
    )

    input_tokens, output_tokens, estimated_text_cost = _estimate_text_cost(
        text_provider,
        text_model,
        output_type,
        idea_count,
        script_count,
        thumbnail_count,
    )
    image_unit_cost, estimated_image_cost = _estimate_image_cost(
        image_provider,
        image_model,
        thumbnail_count,
        image_size,
        image_quality,
    )

    estimate_col1, estimate_col2 = st.columns(2)
    with estimate_col1:
        st.markdown(
            f"""
            <div class="yt-card" style="padding:0.85rem 1rem;margin-bottom:0.75rem;">
                <div style="font-size:11px;color:#7D8AB1;letter-spacing:0.08em;text-transform:uppercase;">Estimated text spend</div>
                <div style="font-size:26px;font-weight:700;color:#FFFFFF;">${estimated_text_cost:.4f}</div>
                <div style="font-size:12px;color:#A8B0CC;margin-top:0.2rem;">~{input_tokens:,} input tokens and ~{output_tokens:,} output tokens for this task mix.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with estimate_col2:
        st.markdown(
            f"""
            <div class="yt-card" style="padding:0.85rem 1rem;margin-bottom:0.75rem;">
                <div style="font-size:11px;color:#7D8AB1;letter-spacing:0.08em;text-transform:uppercase;">Estimated thumbnail spend</div>
                <div style="font-size:26px;font-weight:700;color:#FFFFFF;">${estimated_image_cost:.4f}</div>
                <div style="font-size:12px;color:#A8B0CC;margin-top:0.2rem;">{thumbnail_count} image(s) at about ${image_unit_cost:.4f} each using {image_model_meta['label']}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)
    gen_text = col_a.button("Generate AI Content", use_container_width=True)
    gen_thumb = col_b.button("Generate Thumbnail Images", use_container_width=True)

    total_videos = len(channel_df)
    total_views = int(channel_df["views"].fillna(0).sum())
    avg_views = int(channel_df["views"].fillna(0).mean())
    med_eng = float(channel_df["engagement_rate"].median() * 100)

    if gen_text:
        count_instruction = ""
        if output_type == "Full Pack (titles + descriptions + scripts + thumbnail concepts)":
            count_instruction = (
                f"Generate exactly {idea_count} video ideas, {script_count} scripts/outlines, "
                f"and {thumbnail_count} thumbnail concepts."
            )
        elif output_type == "Video Ideas":
            count_instruction = f"Generate exactly {idea_count} video ideas."
        elif output_type == "Niche Expansion":
            count_instruction = f"Generate exactly {idea_count} niche opportunities with positioning notes."
        elif output_type == "Titles Only":
            count_instruction = f"Generate exactly {idea_count} title options."
        elif output_type == "Descriptions Only":
            count_instruction = f"Generate exactly {idea_count} video descriptions."
        elif output_type == "Scripts Only":
            count_instruction = f"Generate exactly {script_count} script or outline drafts."
        elif output_type == "Hooks + CTAs":
            count_instruction = f"Generate exactly {idea_count} hook and CTA sets."
        elif output_type == "Shorts Ideas":
            count_instruction = f"Generate exactly {idea_count} short-form ideas."
        elif output_type == "Thumbnail Concepts":
            count_instruction = f"Generate exactly {thumbnail_count} thumbnail concepts."

        prompt = (
                "You are an advanced YouTube strategist. "
                "Produce concise, high-performing outputs grounded in these channel stats.\n\n"
                f"Channel: {channel_title} ({channel_id})\n"
                f"Videos(1y): {total_videos}, Total views: {total_views}, Avg views/video: {avg_views}, Median engagement: {med_eng:.2f}%\n"
                f"Priority keywords: {', '.join(keyword_hints[:15])}\n"
                f"Task: {output_type}\n"
                f"Requested output counts: ideas={idea_count}, scripts={script_count}, thumbnails={thumbnail_count}\n"
                f"Audience level: {audience_profile}\n"
                f"Format focus: {format_focus}\n"
                f"Strategy filters: {', '.join(strategy_filters) if strategy_filters else 'None'}\n"
                f"Avoid / exclude: {exclude_topics or 'None'}\n"
                f"Brief: {creative_brief}\n\n"
                f"{count_instruction}\n"
                "When relevant include:\n"
                "- strong hooks\n"
                "- clear structure\n"
                "- search intent alignment\n"
                "- simple CTA\n"
            )
        with st.spinner("Generating AI content..."):
            try:
                model_name = text_model.strip()
                output = _generate_text_with_provider_pool(text_provider, model_name, prompt)

                st.markdown(
                    f"""
                    <div class="yt-card" style="margin-top:0.8rem;">
                        <div style="font-size:13px;color:#B0B0B0;margin-bottom:0.35rem;">AI Output</div>
                        <pre style="white-space:pre-wrap;font-size:13px;color:#FFFFFF;background:rgba(20,27,41,0.95);padding:0.75rem;border-radius:10px;border:1px solid rgba(255,255,255,0.12);max-height:520px;overflow:auto;">{output}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except RuntimeError:
                # Error already surfaced to user.
                pass
            except Exception as exc:
                st.error(f"AI generation failed: {exc}")

    if gen_thumb:
        base_title = (
            channel_df.sort_values("views", ascending=False)
            .head(1)["video_title"]
            .iloc[0]
        )

        with st.spinner("Generating thumbnails..."):
            try:
                images = _generate_images_with_provider_pool(
                    image_provider,
                    image_model.strip(),
                    title=f"Inspired by: {base_title}",
                    context=(
                        f"{creative_brief}\n"
                        f"Audience level: {audience_profile}. "
                        f"Format focus: {format_focus}. "
                        f"Filters: {', '.join(strategy_filters) if strategy_filters else 'None'}."
                    ),
                    style=(
                        "High contrast, one clear subject, bold science aesthetic, 16:9 composition"
                    ),
                    negative_prompt=(
                        f"{exclude_topics}, clutter, tiny text, low contrast"
                        if exclude_topics
                        else "clutter, tiny text, low contrast"
                    ),
                    count=thumbnail_count,
                    size=image_size,
                    quality=image_quality,
                    output_format=image_output_format,
                    background=image_background,
                )
                out_dir = os.path.join("outputs", "thumbnails")
                os.makedirs(out_dir, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                thumb_cols = st.columns(min(len(images), 3) or 1)
                for idx, generated in enumerate(images, start=1):
                    ext = "png" if "png" in generated.mime_type else "jpg"
                    filename = f"ytuber_{channel_id}_{ts}_{idx}.{ext}"
                    with open(os.path.join(out_dir, filename), "wb") as fp:
                        fp.write(generated.image_bytes)
                    with thumb_cols[(idx - 1) % len(thumb_cols)]:
                        st.image(generated.image_bytes, use_container_width=True)
                        st.download_button(
                            label="Download",
                            data=generated.image_bytes,
                            file_name=filename,
                            mime=generated.mime_type,
                            use_container_width=True,
                            key=f"ytuber_thumb_{idx}_{ts}",
                        )
            except Exception as exc:
                st.error(f"Thumbnail generation failed: {exc}")


def render() -> None:
    if build is None:
        st.error("Missing dependency: google-api-python-client. Install with: python3 -m pip install google-api-python-client")
        return

    _inject_ytuber_css()

    if "ytuber_growth_prompt" not in st.session_state:
        st.session_state["ytuber_growth_prompt"] = "@veritasium"
    provider_counts = {
        "youtube": get_provider_key_count("youtube"),
        "gemini": get_provider_key_count("gemini"),
        "openai": get_provider_key_count("openai"),
    }

    st.markdown(
        """
        <div class="ytuber-hero">
            <div class="ytuber-brand-row">
                <span class="ytuber-brand-dot"></span>
                YouTube IP V4
            </div>
            <div class="ytuber-kicker">YouTube Search And Creator Intelligence</div>
            <div class="ytuber-title">Search Any YouTube Channel and Open a Live Growth Workspace.</div>
            <div class="ytuber-subtitle">
                Review live channel performance, benchmark against competitors, and move directly into AI-assisted planning,
                titles, scripts, thumbnails, and publishing decisions.
            </div>
            <div class="ytuber-search-meta">Live Channel Sync. Competitive Context. AI Studio First.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, search_col, _ = st.columns([1, 3.8, 1])
    with search_col:
        st.markdown(
            """
            <div class="ytuber-command-title">Start With a Channel</div>
            <div class="ytuber-command-subtitle">
                Search by handle, channel name, or channel ID to launch the workspace.
            </div>
            """,
            unsafe_allow_html=True,
        )
        input_cols = st.columns([4.2, 1.45], gap="small")
        with input_cols[0]:
            growth_prompt = st.text_input(
                "Search Channel",
                key="ytuber_growth_prompt",
                placeholder="Search YouTube by @handle, channel name, or UC channel ID",
                label_visibility="collapsed",
            )
        with input_cols[1]:
            analyze = st.button(
                "Open Workspace",
                type="primary",
                use_container_width=True,
                disabled=provider_counts["youtube"] <= 0,
            )
        detected_channel = _extract_channel_query(growth_prompt)
        meta_cols = st.columns([1.2, 2.1], gap="medium")
        with meta_cols[0]:
            force_refresh = st.toggle(
                "Force live refresh",
                key="ytuber_force_refresh",
                help="Bypass cached rows and pull the last year from the YouTube API again.",
            )
        with meta_cols[1]:
            if detected_channel:
                st.markdown(
                    f"<div class='ytuber-toolbar-note'>Ready To Search: {escape(detected_channel)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='ytuber-toolbar-note'>Use a Handle Like @veritasium or Paste a Channel ID.</div>",
                    unsafe_allow_html=True,
                )

    if provider_counts["youtube"] <= 0:
        st.warning(
            "No YouTube API keys are configured. Add `YOUTUBE_API_KEYS` or `YOUTUBE_API_KEY` in Streamlit secrets to enable live channel loading."
        )

    if analyze:
        channel_query = _extract_channel_query(st.session_state.get("ytuber_growth_prompt", ""))
        if not channel_query:
            st.error("Enter a channel handle, channel name, or channel ID in the prompt box.")
            return

        with st.spinner("Loading channel data from cache/API..."):
            try:
                channel_df, source, channel_id, channel_title = _fetch_or_get_cached_channel(
                    channel_query=channel_query.strip(),
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                st.error(f"Failed to load channel data: {exc}")
                return

        st.session_state["ytuber_channel_df"] = channel_df
        st.session_state["ytuber_channel_title"] = channel_title
        st.session_state["ytuber_channel_id"] = channel_id
        st.session_state["ytuber_source"] = source
        st.session_state.pop("ytuber_keyword_hints", None)
        st.session_state.pop("ytuber_creative_brief", None)
        st.session_state.pop("ytuber_ai_task_pending", None)
        st.session_state.pop("ytuber_ai_brief_pending", None)
        st.session_state.pop("ytuber_outliers_result", None)
        st.session_state.pop("ytuber_outliers_error", None)
        st.session_state.pop("ytuber_outliers_summary_output", None)
        st.session_state.pop("ytuber_outliers_angles_output", None)
        st.session_state["ytuber_active_module"] = "AI Studio"
        st.session_state.pop("ytuber_active_module_pending", None)
        st.session_state.pop("ytuber_ai_notice", None)

    if "ytuber_channel_df" not in st.session_state:
        st.markdown(
            "<div class='ytuber-empty-copy'>Search a channel above to unlock AI Studio, audit, benchmarking, and planning.</div>",
            unsafe_allow_html=True,
        )
        _render_pool_footer(provider_counts)
        return

    channel_df = st.session_state["ytuber_channel_df"]
    channel_title = st.session_state.get("ytuber_channel_title", "")
    channel_id = st.session_state.get("ytuber_channel_id", "")
    source = st.session_state.get("ytuber_source", "")

    if channel_df.empty:
        st.warning("No videos available for this channel in the last year.")
        _render_pool_footer(provider_counts)
        return

    channel_df = _ensure_numeric_and_dates(channel_df)
    st.markdown(
        f"""
        <div class="ytuber-banner">
            <div class="ytuber-banner-title">{escape(channel_title)}</div>
            <div class="ytuber-banner-meta">Live Workspace Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pending_module = st.session_state.pop("ytuber_active_module_pending", None)
    if pending_module in WORKSPACE_MODULES:
        st.session_state["ytuber_active_module"] = pending_module
    if (
        "ytuber_active_module" not in st.session_state
        or st.session_state["ytuber_active_module"] not in WORKSPACE_MODULES
    ):
        st.session_state["ytuber_active_module"] = "AI Studio"

    active_module = st.segmented_control(
        "Workspace",
        WORKSPACE_MODULES,
        key="ytuber_active_module",
        selection_mode="single",
        label_visibility="collapsed",
    )

    if active_module == "AI Studio":
        hints = st.session_state.get("ytuber_keyword_hints") or _top_keywords(channel_df, 20)
        _render_ai_studio(channel_df, channel_title, channel_id, hints)
    elif active_module == "Overview":
        _render_overview(channel_df)
    elif active_module == "Channel Audit":
        _render_channel_audit(channel_df)
    elif active_module == "Keyword Intel":
        keyword_hints = _render_keyword_intel(channel_df)
        st.session_state["ytuber_keyword_hints"] = keyword_hints
    elif active_module == "Outliers Finder":
        _render_outliers_shortcut(channel_df, channel_title)
    elif active_module == "Title & SEO Lab":
        hints = st.session_state.get("ytuber_keyword_hints") or _top_keywords(channel_df, 20)
        _render_title_seo_lab(hints)
    elif active_module == "Competitor Benchmark":
        _render_competitor_benchmark(channel_df, channel_title, channel_id)
    elif active_module == "Content Planner":
        _render_content_planner(channel_df)

    _render_pool_footer(
        provider_counts,
        workspace_meta={
            "source_label": source.replace("_", " ").title() if source else "Unavailable",
            "channel_id": channel_id,
            "video_count": len(channel_df),
        },
    )
