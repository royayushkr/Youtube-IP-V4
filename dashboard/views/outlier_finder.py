from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.visualizations import styled_dataframe, styled_keyword_chips
from src.services.outlier_ai import InsightCard, OutlierAIReport, generate_outlier_ai_report
from src.services.outliers_finder import (
    DURATION_BUCKET_ORDER,
    OutlierSearchRequest,
    build_age_bucket_summary,
    build_duration_summary,
    build_scan_quality_summary,
    build_title_keyword_summary,
    build_title_pattern_summary,
    score_band_for_value,
    search_outlier_videos,
)
from src.utils.api_keys import get_provider_key_count


TIMEFRAME_OPTIONS = ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"]
FRESHNESS_OPTIONS = {
    "Any": None,
    "Last 7 Days": 7,
    "Last 14 Days": 14,
    "Last 30 Days": 30,
}
REGION_OPTIONS = ["Any", "US", "IN", "GB", "CA", "AU", "DE", "FR", "BR", "JP"]
LANGUAGE_OPTIONS = ["Any", "en", "es", "hi", "pt", "de", "fr", "ja"]
STRICTNESS_OPTIONS = ["Strict", "Balanced", "Loose"]
DURATION_OPTIONS = ["Any"] + [bucket for bucket in DURATION_BUCKET_ORDER if bucket != "Unknown"]
SORT_OPTIONS = {
    "Outlier Score": ("outlier_score", False),
    "Views Per Day": ("views_per_day", False),
    "Views": ("views", False),
    "Newest": ("age_days", True),
    "Language Confidence": ("language_confidence", False),
}
AI_PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openai": "OpenAI / ChatGPT",
}
AI_MODELS = {
    "gemini": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
}
SEARCH_STATE_KEYS = (
    "outlier_page_query",
    "outlier_page_timeframe",
    "outlier_page_match_mode",
    "outlier_page_region",
    "outlier_page_language",
    "outlier_page_custom_dates",
    "outlier_page_freshness",
    "outlier_page_duration",
    "outlier_page_language_strictness",
    "outlier_page_min_views",
    "outlier_page_min_subscribers",
    "outlier_page_max_subscribers",
    "outlier_page_include_hidden",
    "outlier_page_exclude_keywords",
    "outlier_page_search_pages",
    "outlier_page_baseline_channels",
    "outlier_page_baseline_videos",
    "outlier_page_result",
    "outlier_page_error",
    "outlier_page_sort",
    "outlier_page_ai_report",
    "outlier_page_ai_fingerprint",
    "outlier_page_ai_provider",
    "outlier_page_ai_model",
    "outlier_page_prefill_note",
)
PURPLE_SCALE = ["#6D28D9", "#8B5CF6", "#A855F7", "#C084FC", "#DDD6FE"]


def _inject_outlier_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stForm"] {
            background:
                radial-gradient(circle at top left, rgba(139, 92, 246, 0.14) 0%, transparent 34%),
                linear-gradient(180deg, rgba(26, 33, 64, 0.94) 0%, rgba(15, 19, 36, 0.98) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 30px;
            padding: 1.5rem 1.5rem 1.2rem;
            box-shadow: 0 20px 48px rgba(3, 6, 20, 0.42);
            margin: 0 auto 1.6rem;
            max-width: var(--app-command-width);
        }
        [data-testid="stForm"] [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
            row-gap: 0.95rem;
        }
        [data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"] {
            min-height: calc(var(--app-control-height) + 40px);
            padding: 0.05rem 0.15rem;
        }
        [data-testid="stExpander"] {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            background: rgba(255,255,255,0.02);
            margin-top: 0.15rem;
        }
        .outlier-page {
            max-width: var(--app-page-width);
            margin: 0 auto;
        }
        .outlier-hero {
            max-width: 920px;
            margin: 0 auto 1.8rem;
            text-align: center;
        }
        .outlier-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.09);
            color: #F7F8FC;
            font-size: 12px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.95rem;
        }
        .outlier-kicker-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, #A855F7, #8B5CF6);
            box-shadow: 0 0 16px rgba(139, 92, 246, 0.55);
        }
        .outlier-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: clamp(38px, 4vw, 58px);
            line-height: 0.98;
            font-weight: 700;
            color: #F7F8FC;
            letter-spacing: -0.04em;
            margin-bottom: 0.85rem;
        }
        .outlier-subtitle {
            max-width: 760px;
            margin: 0 auto;
            font-size: 16px;
            line-height: 1.65;
            color: #B8C1DA;
        }
        .outlier-trust-row {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.05rem;
        }
        .outlier-pill {
            padding: 0.4rem 0.78rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            color: #D7DDF0;
            font-size: 12px;
        }
        .outlier-shell-head {
            margin-bottom: 1.1rem;
        }
        .outlier-shell-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: #F7F8FC;
            margin-bottom: 0.25rem;
        }
        .outlier-shell-copy {
            color: #B8C1DA;
            font-size: 13px;
            line-height: 1.55;
            max-width: 620px;
        }
        .outlier-helper {
            color: #97A2C3;
            font-size: 12px;
            line-height: 1.55;
            margin-top: 0.75rem;
        }
        .outlier-inline-field-label {
            color: #B8C1DA;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }
        .outlier-row-label {
            color: #8E9AC0;
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0 0 0.5rem;
        }
        .outlier-visibility-copy {
            color: #8E9AC0;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 0.35rem;
        }
        .outlier-form-gap {
            height: 0.25rem;
        }
        .outlier-prefill-note {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.55rem 0.8rem;
            margin: 0 auto 1.05rem;
            border-radius: 999px;
            background: rgba(139, 92, 246, 0.13);
            border: 1px solid rgba(196, 181, 253, 0.18);
            color: #E6DFFC;
            font-size: 12px;
        }
        .outlier-panel-note {
            margin-top: 1rem;
            padding: 0.8rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            color: #B8C1DA;
            font-size: 12px;
            line-height: 1.6;
        }
        .outlier-section {
            margin-top: 2rem;
        }
        .outlier-results-summary {
            color: #B8C1DA;
            font-size: 13px;
            line-height: 1.55;
            margin-top: 0.18rem;
        }
        .outlier-summary-card {
            border-radius: 22px;
            min-height: 156px;
            padding: 1rem 1rem 0.95rem;
            background: linear-gradient(180deg, rgba(34, 42, 76, 0.96) 0%, rgba(20, 26, 49, 0.98) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 16px 36px rgba(3, 6, 20, 0.40);
            margin-bottom: 0.9rem;
        }
        .outlier-summary-label {
            color: #8D98BA;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 0.3rem;
        }
        .outlier-summary-value {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 30px;
            font-weight: 700;
            line-height: 1.05;
            color: #F7F8FC;
        }
        .outlier-summary-detail {
            color: #B8C1DA;
            font-size: 12px;
            margin-top: 0.25rem;
            line-height: 1.5;
        }
        .outlier-result-card {
            border-radius: 24px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(33, 40, 73, 0.98) 0%, rgba(20, 26, 49, 0.99) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 40px rgba(3, 6, 20, 0.44);
            margin-bottom: 1rem;
            min-height: 455px;
            display: flex;
            flex-direction: column;
        }
        .outlier-result-card img {
            width: 100%;
            height: 185px;
            object-fit: cover;
            display: block;
            background: rgba(255,255,255,0.03);
        }
        .outlier-result-body {
            padding: 1rem 1rem 1rem;
            display: flex;
            flex-direction: column;
            flex: 1;
        }
        .outlier-result-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
        }
        .outlier-result-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 17px;
            font-weight: 700;
            color: #F7F8FC;
            line-height: 1.32;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
            overflow: hidden;
            min-height: 46px;
        }
        .outlier-result-channel {
            color: #A7B3D3;
            font-size: 12px;
            margin-top: 0.2rem;
        }
        .outlier-score-pill {
            flex-shrink: 0;
            min-width: 94px;
            padding: 0.5rem 0.62rem;
            border-radius: 18px;
            text-align: center;
            background: rgba(139, 92, 246, 0.16);
            border: 1px solid rgba(196, 181, 253, 0.2);
        }
        .outlier-score-value {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 22px;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1;
        }
        .outlier-score-label {
            font-size: 10px;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #DCCFFF;
            margin-top: 0.12rem;
        }
        .outlier-score-band {
            margin-top: 0.26rem;
            font-size: 10px;
            color: #C4B5FD;
        }
        .outlier-metric-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.75rem 0 0.85rem;
        }
        .outlier-metric-chip {
            padding: 0.32rem 0.6rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: #DDE3F6;
            font-size: 11px;
        }
        .outlier-bullets {
            margin: 0;
            padding-left: 1rem;
            color: #EAEFFF;
            font-size: 12px;
            line-height: 1.55;
        }
        .outlier-bullets li {
            margin-bottom: 0.32rem;
        }
        .outlier-card-link {
            margin-top: auto;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: #D8C7FF !important;
            font-weight: 700;
            font-size: 12px;
            padding-top: 0.7rem;
            text-decoration: none;
        }
        .outlier-chart-shell {
            margin-bottom: 0.6rem;
            padding: 0.1rem 0 0.2rem;
        }
        .outlier-chart-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            color: #F7F8FC;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .outlier-chart-copy {
            color: #AAB5D6;
            font-size: 12px;
            line-height: 1.5;
            margin-bottom: 0.45rem;
        }
        .outlier-ai-hero {
            border-radius: 24px;
            padding: 1rem 1rem 0.95rem;
            background: linear-gradient(180deg, rgba(34, 42, 76, 0.96) 0%, rgba(20, 26, 49, 0.98) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 40px rgba(3, 6, 20, 0.40);
            margin-bottom: 1rem;
        }
        .outlier-ai-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 22px;
            font-weight: 700;
            color: #F7F8FC;
            margin-bottom: 0.25rem;
        }
        .outlier-ai-copy {
            color: #C4CDEA;
            font-size: 14px;
            line-height: 1.6;
        }
        .outlier-ai-meta {
            margin-top: 0.4rem;
            color: #9EABCF;
            font-size: 12px;
        }
        .outlier-ai-card {
            border-radius: 22px;
            min-height: 188px;
            padding: 1rem 1rem 0.95rem;
            background: linear-gradient(180deg, rgba(33, 40, 73, 0.98) 0%, rgba(20, 26, 49, 0.99) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 40px rgba(3, 6, 20, 0.38);
            margin-bottom: 0.9rem;
            display: flex;
            flex-direction: column;
        }
        .outlier-ai-card-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 16px;
            font-weight: 700;
            color: #F7F8FC;
            margin-bottom: 0.35rem;
        }
        .outlier-ai-card-body {
            color: #D7DDF0;
            font-size: 13px;
            line-height: 1.58;
            flex: 1;
        }
        .outlier-ai-card-support {
            color: #95A3C7;
            font-size: 12px;
            line-height: 1.45;
            margin-top: 0.5rem;
        }
        .outlier-quality-card {
            min-height: 320px;
            display: flex;
            flex-direction: column;
        }
        .outlier-quality-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.85rem;
        }
        .outlier-quality-metric {
            border-radius: 18px;
            padding: 0.82rem 0.85rem;
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .outlier-quality-label {
            color: #95A3C7;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }
        .outlier-quality-value {
            color: #F7F8FC;
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 24px;
            font-weight: 700;
            line-height: 1.05;
        }
        .outlier-quality-note {
            color: #B8C1DA;
            font-size: 12px;
            line-height: 1.6;
            margin-top: auto;
            padding-top: 0.85rem;
        }
        .outlier-empty-card {
            border-radius: 24px;
            padding: 1.1rem 1.15rem;
            background: linear-gradient(180deg, rgba(32, 39, 70, 0.96) 0%, rgba(16, 20, 37, 0.98) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 16px 38px rgba(3, 6, 20, 0.34);
            margin-bottom: 1rem;
        }
        .outlier-empty-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: #F7F8FC;
            margin-bottom: 0.25rem;
        }
        .outlier-empty-copy {
            color: #B8C1DA;
            font-size: 13px;
            line-height: 1.58;
        }
        .outlier-method-card {
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 0.8rem;
        }
        .outlier-method-card h4 {
            margin: 0 0 0.4rem;
            color: #F7F8FC;
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
        }
        .outlier-method-card p,
        .outlier-method-card li {
            color: #CBD4ED;
            font-size: 13px;
            line-height: 1.58;
        }
        .outlier-footnote {
            margin-top: 0.4rem;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(196, 181, 253, 0.12);
            background: rgba(139, 92, 246, 0.08);
            color: #D9DEF1;
            font-size: 12px;
            line-height: 1.62;
        }
        @media (max-width: 980px) {
            .outlier-title {
                font-size: 42px;
            }
            [data-testid="stForm"] {
                padding: 1.2rem 1rem 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _parse_exclude_keywords(text: str) -> Tuple[str, ...]:
    parts = [part.strip() for part in str(text or "").split(",")]
    values = [part for part in parts if part]
    return tuple(dict.fromkeys(values))


def _format_int(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}"


def _format_pct(value: float) -> str:
    return f"{float(value or 0):.1f}%"


def _format_subscribers(value: Optional[float], hidden: bool) -> str:
    if hidden or value is None or pd.isna(value):
        return "Hidden"
    number = int(float(value))
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return str(number)


def _result_fingerprint(result_frame: pd.DataFrame, query: str) -> str:
    top_ids = ",".join(result_frame["video_id"].head(10).astype(str).tolist()) if not result_frame.empty else ""
    return f"{query}|{top_ids}|{len(result_frame)}"


def _style_chart(fig, *, legend_title: Optional[str] = None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D7DDF0", family="Plus Jakarta Sans, Inter, system-ui"),
        title_font=dict(size=18, family="Space Grotesk, Plus Jakarta Sans, system-ui", color="#F7F8FC"),
        legend=dict(
            bgcolor="rgba(20, 26, 49, 0.72)",
            bordercolor="rgba(255,255,255,0.06)",
            borderwidth=1,
            font=dict(color="#D7DDF0"),
        ),
        legend_title_text=legend_title,
        margin=dict(l=8, r=8, t=58, b=8),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
        title_font=dict(color="#C9D2EC"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
        title_font=dict(color="#C9D2EC"),
    )
    return fig


def _build_summary_stats(result_frame: pd.DataFrame) -> Dict[str, Any]:
    if result_frame.empty:
        return {}
    return {
        "median_outlier_score": round(float(result_frame["outlier_score"].median()), 1),
        "median_views_per_day": int(result_frame["views_per_day"].median()),
        "median_views": int(result_frame["views"].median()),
        "high_language_share": round(float((result_frame["language_confidence_label"] == "High").mean()) * 100, 1),
        "top_duration_bucket": str(
            result_frame["duration_bucket"].value_counts().index[0]
            if result_frame["duration_bucket"].notna().any()
            else "Unknown"
        ),
        "top_title_pattern": str(
            result_frame["title_pattern"].value_counts().index[0]
            if result_frame["title_pattern"].notna().any()
            else "General"
        ),
    }


def _render_summary_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        (
            '<div class="outlier-summary-card">'
            f'<div class="outlier-summary-label">{escape(label)}</div>'
            f'<div class="outlier-summary-value">{escape(value)}</div>'
            f'<div class="outlier-summary-detail">{escape(detail)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_search_header() -> None:
    st.markdown(
        (
            '<div class="outlier-shell-head">'
            '<div class="outlier-shell-title">Search A Niche</div>'
            '<div class="outlier-shell-copy">Scan any topic, tighten the filters, and surface the videos that are outperforming their likely baseline.</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_search_footer_note() -> None:
    st.markdown(
        (
            '<div class="outlier-helper">'
            'Use tighter filters when you want a cleaner research set, and broader settings when you want a wider scouting pass.'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_prefill_note(note: str) -> None:
    st.markdown(
        f'<div class="outlier-prefill-note">Suggested Query Loaded • {escape(note)}</div>',
        unsafe_allow_html=True,
    )


def _truncate_copy(value: str, limit: int = 34) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit - 1].rstrip()}…"


def _render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        (
            '<div class="app-section-intro">'
            f'<div class="app-section-title">{escape(title)}</div>'
            f'<div class="app-section-copy">{escape(copy)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_subsection_label(label: str) -> None:
    st.markdown(f'<div class="app-subsection-label">{escape(label)}</div>', unsafe_allow_html=True)


def _meta_pill_html(label: str, value: str, *, state: bool = False) -> str:
    classes = "app-meta-pill app-meta-pill--state" if state else "app-meta-pill"
    if label:
        body = f"<strong>{escape(label)}:</strong> {escape(value)}"
    else:
        body = escape(value)
    return f'<span class="{classes}">{body}</span>'


def _build_filter_tags(result) -> List[Tuple[str, str]]:
    tags: List[Tuple[str, str]] = [("Niche", _truncate_copy(result.request.niche_query, 38))]
    if result.request.relevance_language:
        tags.append(
            ("Language", f"{result.request.relevance_language.upper()} ({result.request.language_strictness.title()})")
        )
    if result.request.region_code:
        tags.append(("Region", result.request.region_code))
    if result.request.duration_preference != "Any":
        tags.append(("Duration", result.request.duration_preference))
    if result.request.min_views > 0:
        tags.append(("Minimum Views", f"{result.request.min_views:,}+"))
    if result.request.match_mode == "exact":
        tags.append(("Match", "Exact Phrase"))
    if len(tags) > 5:
        visible = tags[:5]
        visible.append(("", f"+{len(tags) - 5} More"))
        return visible
    return tags


def _build_state_tags(result, result_frame: pd.DataFrame) -> List[Tuple[str, str]]:
    return [
        ("Results", f"{len(result_frame):,}"),
        ("Cache", result.cache_policy),
        ("Quota", result.quota_profile),
    ]


def _render_metadata_cluster(result, result_frame: pd.DataFrame) -> None:
    filter_html = "".join(_meta_pill_html(label, value) for label, value in _build_filter_tags(result))
    state_html = "".join(_meta_pill_html(label, value, state=True) for label, value in _build_state_tags(result, result_frame))
    st.markdown(
        (
            '<div class="app-meta-stack">'
            f'<div class="app-meta-group">{filter_html}</div>'
            f'<div class="app-meta-group">{state_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_result_card(row: pd.Series) -> None:
    thumb_html = (
        f'<img src="{escape(str(row.get("thumbnail_url", "")))}" alt="{escape(str(row.get("video_title", "")))}" />'
        if str(row.get("thumbnail_url", "")).strip()
        else ""
    )
    score = float(row.get("outlier_score", 0) or 0)
    why = f"<strong>Why It Stands Out:</strong> {escape(str(row.get('why_outlier', '')))}"
    cue = f"<strong>Research Cue:</strong> {escape(str(row.get('research_cue', '')))}"
    st.markdown(
        (
            '<div class="outlier-result-card">'
            f"{thumb_html}"
            '<div class="outlier-result-body">'
            '<div class="outlier-result-top">'
            '<div>'
            f'<div class="outlier-result-title">{escape(str(row.get("video_title", "")))}</div>'
            f'<div class="outlier-result-channel">{escape(str(row.get("channel_title", "")))}</div>'
            '</div>'
            '<div class="outlier-score-pill">'
            f'<div class="outlier-score-value">{score:.1f}</div>'
            '<div class="outlier-score-label">Outlier Score</div>'
            f'<div class="outlier-score-band">{escape(score_band_for_value(score))}</div>'
            '</div>'
            '</div>'
            '<div class="outlier-metric-row">'
            f'<span class="outlier-metric-chip">{_format_int(row.get("views"))} Views</span>'
            f'<span class="outlier-metric-chip">{_format_int(row.get("views_per_day"))} / Day</span>'
            f'<span class="outlier-metric-chip">{_format_subscribers(row.get("channel_subscriber_count"), bool(row.get("hidden_subscriber_count")))} Subs</span>'
            '</div>'
            '<ul class="outlier-bullets">'
            f"<li>{why}</li>"
            f"<li>{cue}</li>"
            '</ul>'
            f'<a class="outlier-card-link" href="{escape(str(row.get("video_url", "")))}" target="_blank">Open On YouTube ↗</a>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_result_cards(result_frame: pd.DataFrame) -> None:
    cols = st.columns(3, gap="medium")
    for idx, (_, row) in enumerate(result_frame.head(9).iterrows()):
        with cols[idx % 3]:
            _render_result_card(row)


def _render_chart_shell(title: str, copy: str) -> None:
    st.markdown(
        (
            '<div class="outlier-chart-shell">'
            f'<div class="outlier-chart-title">{escape(title)}</div>'
            f'<div class="outlier-chart-copy">{escape(copy)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _breakout_scatter(result_frame: pd.DataFrame):
    chart_df = result_frame.copy()
    chart_df["subscribers_log10"] = chart_df["channel_subscriber_count"].fillna(0).apply(
        lambda value: math.log10(float(value) + 1)
    )
    fig = px.scatter(
        chart_df,
        x="subscribers_log10",
        y="views_per_day",
        size="outlier_score",
        color="age_bucket",
        hover_name="video_title",
        color_discrete_sequence=PURPLE_SCALE,
        hover_data={
            "channel_title": True,
            "views": ":,",
            "outlier_score": ":.1f",
            "age_days": ":.1f",
            "language_confidence_label": True,
            "subscribers_log10": False,
        },
        title="Breakout Map",
        labels={
            "subscribers_log10": "Channel Subscribers (Log10 + 1)",
            "views_per_day": "Views Per Day",
            "age_bucket": "Publish Age Bucket",
        },
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.18)"), sizemin=11),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Channel: %{customdata[0]}<br>"
            "Views: %{customdata[1]:,}<br>"
            "Outlier Score: %{customdata[2]:.1f}<br>"
            "Age (Days): %{customdata[3]:.1f}<br>"
            "Language Confidence: %{customdata[4]}<extra></extra>"
        ),
    )
    return _style_chart(fig, legend_title="Publish Age Bucket")


def _age_bucket_chart(result_frame: pd.DataFrame):
    summary = build_age_bucket_summary(result_frame)
    fig = px.bar(
        summary,
        x="age_bucket",
        y="median_outlier_score",
        color="median_views_per_day",
        text="outlier_count",
        title="Outlier Score By Publish Age",
        labels={
            "age_bucket": "Publish Age Bucket",
            "median_outlier_score": "Median Outlier Score",
            "median_views_per_day": "Median Views Per Day",
        },
        color_continuous_scale=PURPLE_SCALE,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Median Outlier Score: %{y:.1f}<br>"
            "Median Views Per Day: %{marker.color:.0f}<br>"
            "Video Count: %{text}<extra></extra>"
        )
    )
    return _style_chart(fig)


def _duration_chart(result_frame: pd.DataFrame):
    summary = build_duration_summary(result_frame)
    fig = px.bar(
        summary,
        x="duration_bucket",
        y="outlier_count",
        color="median_outlier_score",
        title="Winning Video Lengths",
        labels={
            "duration_bucket": "Duration Bucket",
            "outlier_count": "Outlier Count",
            "median_outlier_score": "Median Outlier Score",
        },
        color_continuous_scale=PURPLE_SCALE,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Outlier Count: %{y}<br>"
            "Median Outlier Score: %{marker.color:.1f}<extra></extra>"
        )
    )
    return _style_chart(fig)


def _title_pattern_chart(result_frame: pd.DataFrame):
    summary = build_title_pattern_summary(result_frame)
    fig = px.bar(
        summary,
        x="title_pattern",
        y="outlier_count",
        color="median_outlier_score",
        title="Repeated Title Structures",
        labels={
            "title_pattern": "Title Pattern",
            "outlier_count": "Outlier Count",
            "median_outlier_score": "Median Outlier Score",
        },
        color_continuous_scale=PURPLE_SCALE,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Outlier Count: %{y}<br>"
            "Median Outlier Score: %{marker.color:.1f}<extra></extra>"
        )
    )
    return _style_chart(fig)


def _render_scan_quality_card(result_frame: pd.DataFrame) -> None:
    quality = build_scan_quality_summary(result_frame)
    metrics = [
        ("Language Match", _format_pct(quality["high_language_match_share"])),
        ("Recent Uploads", _format_pct(quality["recent_upload_share"])),
        ("Strong Signals", _format_pct(quality["strong_signal_share"])),
        ("Hidden Subs", _format_pct(quality["hidden_subscriber_share"])),
    ]
    metrics_html = "".join(
        (
            '<div class="outlier-quality-metric">'
            f'<div class="outlier-quality-label">{escape(label)}</div>'
            f'<div class="outlier-quality-value">{escape(value)}</div>'
            "</div>"
        )
        for label, value in metrics
    )
    st.markdown(
        (
            '<div class="outlier-quality-card">'
            '<div class="outlier-quality-grid">'
            f"{metrics_html}"
            '</div>'
            '<div class="outlier-quality-note">High language match and strong-signal share usually indicate a cleaner research set. A higher hidden-subscriber share weakens channel-size confidence.</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_ai_card(card: InsightCard) -> None:
    st.markdown(
        (
            '<div class="outlier-ai-card">'
            f'<div class="outlier-ai-card-title">{escape(card.title)}</div>'
            f'<div class="outlier-ai-card-body">{escape(card.body)}</div>'
            f'<div class="outlier-ai-card-support">{escape(card.support)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_ai_card_grid(title: str, cards: Sequence[InsightCard], columns: int = 2) -> None:
    if not cards:
        return
    _render_subsection_label(title)
    pair_size = max(1, columns)
    for start in range(0, len(cards), pair_size):
        batch = cards[start:start + pair_size]
        if len(batch) == 1:
            _render_ai_card(batch[0])
            continue
        cols = st.columns(len(batch), gap="medium")
        for col, card in zip(cols, batch):
            with col:
                _render_ai_card(card)


def _render_ai_report(report: OutlierAIReport) -> None:
    top_cols = st.columns([1.7, 1], gap="medium")
    with top_cols[0]:
        st.markdown(
            (
                '<div class="outlier-ai-hero">'
                f'<div class="outlier-ai-title">{escape(report.executive_headline)}</div>'
                f'<div class="outlier-ai-copy">{escape(report.key_takeaway)}</div>'
                f'<div class="outlier-ai-meta">Provider: {escape(report.provider.title())} • Model: {escape(report.model)} • Confidence: {escape(report.confidence_label)}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
    with top_cols[1]:
        notes_html = "".join(f"<li>{escape(note)}</li>" for note in (report.confidence_notes or ("Confidence is based on the quality of the surfaced public signals.",)))
        st.markdown(
            (
                '<div class="outlier-ai-card">'
                '<div class="outlier-ai-card-title">Confidence And Caveats</div>'
                f'<div class="outlier-ai-card-body"><ul class="outlier-bullets">{notes_html}</ul></div>'
                '<div class="outlier-ai-card-support">Treat this as research guidance, not a prediction guarantee.</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    _render_ai_card_grid("Breakout Themes", report.breakout_themes, columns=2)
    _render_ai_card_grid("Title Pattern Observations", report.title_patterns, columns=2)
    _render_ai_card_grid("Repeatable Content Angles", report.repeatable_angles, columns=2)
    _render_ai_card_grid("Notable Anomalies", report.notable_anomalies, columns=2)

    if report.next_steps:
        _render_subsection_label("What To Test Next")
        for start in range(0, len(report.next_steps), 2):
            batch = report.next_steps[start:start + 2]
            if len(batch) == 1:
                step = batch[0]
                st.markdown(
                    (
                        '<div class="outlier-ai-card">'
                        f'<div class="outlier-ai-card-title">Action {start + 1}</div>'
                        f'<div class="outlier-ai-card-body">{escape(step)}</div>'
                        '<div class="outlier-ai-card-support">Use this as the next concrete experiment to test against your current packaging and publishing rhythm.</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
                continue
            cols = st.columns(2, gap="medium")
            for idx, (col, step) in enumerate(zip(cols, batch), start=start + 1):
                with col:
                    st.markdown(
                        (
                            '<div class="outlier-ai-card">'
                            f'<div class="outlier-ai-card-title">Action {idx}</div>'
                            f'<div class="outlier-ai-card-body">{escape(step)}</div>'
                            '<div class="outlier-ai-card-support">Use this as the next concrete experiment to test against your current packaging and publishing rhythm.</div>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )

    if report.warnings:
        st.markdown(
            f'<div class="outlier-panel-note">{" | ".join(escape(warning) for warning in report.warnings)}</div>',
            unsafe_allow_html=True,
        )
    if report.raw_fallback:
        st.markdown('<div class="outlier-panel-note"><strong>Fallback Summary.</strong> The AI provider returned unstructured output for this run, so the text below is shown as plain summary copy.</div>', unsafe_allow_html=True)
        st.markdown(report.raw_fallback)


def _render_methodology_section() -> None:
    _render_section_intro(
        "How This Works",
        "Use these notes to understand what the score means, how the filters behave, and what the public-data caveats imply for your research decisions.",
    )
    with st.expander("Open Metric Definitions, Filter Rules, And Caveats", expanded=False):
        left_col, right_col = st.columns(2, gap="medium")
        with left_col:
            st.markdown(
                (
                    '<div class="outlier-method-card">'
                    '<h4>What The Outlier Score Means</h4>'
                    '<p>The score is a public-performance heuristic. It blends channel-baseline lift, peer percentile, engagement percentile, and recency boost into one 0-100 score.</p>'
                    '</div>'
                    '<div class="outlier-method-card">'
                    '<h4>Metric Definitions</h4>'
                    '<ul>'
                    '<li><strong>Views Per Day</strong>: total views divided by video age in days.</li>'
                    '<li><strong>Engagement Rate</strong>: likes plus comments divided by views.</li>'
                    '<li><strong>Views Per Subscriber</strong>: views divided by public subscriber count when available.</li>'
                    '<li><strong>Language Confidence</strong>: metadata plus title-script heuristic, not a guaranteed classifier.</li>'
                    '</ul>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
        with right_col:
            st.markdown(
                (
                    '<div class="outlier-method-card">'
                    '<h4>How The Filters Work</h4>'
                    '<ul>'
                    '<li><strong>Exact Phrase</strong> quotes the query and also checks the returned title and description text.</li>'
                    '<li><strong>Exclude Keywords</strong> are applied in the search query and then enforced again as a post-filter.</li>'
                    '<li><strong>Region</strong> filters videos viewable in that geography, not necessarily channels from that geography.</li>'
                    '<li><strong>Freshness Focus</strong> narrows the surfaced cohort to recent uploads inside the broader timeframe.</li>'
                    '</ul>'
                    '</div>'
                    '<div class="outlier-method-card">'
                    '<h4>Important Caveats</h4>'
                    '<ul>'
                    '<li>No impressions, CTR, watch time, or retention data is available here.</li>'
                    '<li>YouTube search is sampled and ranked, so this is not an exhaustive view of every matching video.</li>'
                    '<li>Subscriber counts can be hidden or rounded, which weakens some channel-size comparisons.</li>'
                    '<li>Language strictness is heuristic-based and should be treated as noise reduction, not perfect classification.</li>'
                    '</ul>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="outlier-footnote">Results come from the scanned cohort returned by the official YouTube API. Language filtering is heuristic-based and becomes stricter as confidence requirements increase.</div>',
        unsafe_allow_html=True,
    )


def _render_pre_search_methodology_teaser() -> None:
    st.markdown(
        (
            '<div class="outlier-empty-card">'
            '<div class="outlier-empty-title">Built For Fast Topic Research</div>'
            '<div class="outlier-empty-copy">The score uses public performance signals, query filters, and channel baseline lift. Search results are cached for one hour, baseline lookups for six hours, and AI research stays optional.</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    st.markdown(
        (
            '<div class="outlier-empty-card">'
            '<div class="outlier-empty-title">Start With A Niche Query</div>'
            '<div class="outlier-empty-copy">Enter a topic, tighten the filters if needed, and the page will surface public videos that are outperforming their likely baseline.</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _clear_result_state() -> None:
    for key in ("outlier_page_result", "outlier_page_error", "outlier_page_ai_report", "outlier_page_ai_fingerprint"):
        st.session_state.pop(key, None)


def _reset_search_state() -> None:
    for key in SEARCH_STATE_KEYS:
        st.session_state.pop(key, None)


def render() -> None:
    _inject_outlier_css()

    provider_counts = {
        "youtube": get_provider_key_count("youtube"),
        "gemini": get_provider_key_count("gemini"),
        "openai": get_provider_key_count("openai"),
    }

    st.markdown(
        (
            '<div class="outlier-hero">'
            '<div class="outlier-kicker"><span class="outlier-kicker-dot"></span>Outlier Finder</div>'
            '<div class="outlier-title">Discover Breakout Videos Before The Niche Catches Up</div>'
            '<div class="outlier-subtitle">Scan any topic, filter the noise, and surface overperforming videos with clear research signals. Review the winners first, understand the pattern next, and layer AI interpretation only after the evidence is visible.</div>'
            '<div class="outlier-trust-row">'
            '<span class="outlier-pill">Public YouTube API Data</span>'
            '<span class="outlier-pill">Explainable Outlier Scoring</span>'
            '<span class="outlier-pill">Quota-Aware Query Caching</span>'
            '<span class="outlier-pill">Structured AI Research</span>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    prefill_note = st.session_state.pop("outlier_page_prefill_note", None)
    if prefill_note:
        _render_prefill_note(prefill_note)

    with st.form("outlier_finder_search_form"):
        _render_search_header()

        query_cols = st.columns([1.8, 1], gap="medium")
        with query_cols[0]:
            niche_query = st.text_input(
                "Niche Or Keyword",
                key="outlier_page_query",
                placeholder="AI automation, documentary storytelling, science shorts, luxury fitness...",
            )
        with query_cols[1]:
            st.markdown('<div class="outlier-inline-field-label">Actions</div>', unsafe_allow_html=True)
            action_cols = st.columns(2, gap="small")
            with action_cols[0]:
                submitted = st.form_submit_button(
                    "Find Outliers",
                    type="primary",
                    use_container_width=True,
                    disabled=provider_counts["youtube"] <= 0,
                )
            with action_cols[1]:
                reset_clicked = st.form_submit_button("Reset Filters", use_container_width=True)

        filter_row_one = st.columns(4, gap="medium")
        with filter_row_one[0]:
            timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS, index=1, key="outlier_page_timeframe")
        with filter_row_one[1]:
            match_mode = st.segmented_control(
                "Match Mode",
                ["Broad", "Exact Phrase"],
                key="outlier_page_match_mode",
                selection_mode="single",
                default="Broad",
            )
        with filter_row_one[2]:
            region_code = st.selectbox("Region", REGION_OPTIONS, index=0, key="outlier_page_region")
        with filter_row_one[3]:
            language_code = st.selectbox("Language", LANGUAGE_OPTIONS, index=0, key="outlier_page_language")

        if timeframe == "Custom":
            default_end = datetime.now(timezone.utc).date()
            default_start = default_end - timedelta(days=30)
            custom_dates = st.date_input(
                "Custom Date Range",
                value=(default_start, default_end),
                max_value=default_end,
                key="outlier_page_custom_dates",
            )
        else:
            custom_dates = None

        filter_row_two = st.columns(4, gap="medium")
        with filter_row_two[0]:
            freshness_focus = st.selectbox(
                "Freshness Focus",
                list(FRESHNESS_OPTIONS.keys()),
                index=0,
                key="outlier_page_freshness",
            )
        with filter_row_two[1]:
            duration_preference = st.selectbox(
                "Duration Preference",
                DURATION_OPTIONS,
                index=0,
                key="outlier_page_duration",
            )
        with filter_row_two[2]:
            strictness = st.segmented_control(
                "Language Strictness",
                STRICTNESS_OPTIONS,
                key="outlier_page_language_strictness",
                selection_mode="single",
                default="Strict",
            )
        with filter_row_two[3]:
            min_views = st.selectbox(
                "Minimum Views",
                [0, 1_000, 5_000, 10_000, 50_000, 100_000],
                index=0,
                key="outlier_page_min_views",
                format_func=lambda value: "No Minimum" if value == 0 else f"{value:,}+",
            )

        channel_cols = st.columns([1.85, 1], gap="medium")
        with channel_cols[0]:
            st.markdown('<div class="outlier-row-label">Channel Size</div>', unsafe_allow_html=True)
            subscriber_cols = st.columns(2, gap="small")
            with subscriber_cols[0]:
                min_subscribers = st.number_input(
                    "Minimum Subscribers",
                    min_value=0,
                    value=0,
                    step=1_000,
                    key="outlier_page_min_subscribers",
                )
            with subscriber_cols[1]:
                max_subscribers = st.number_input(
                    "Maximum Subscribers",
                    min_value=0,
                    value=0,
                    step=1_000,
                    key="outlier_page_max_subscribers",
                    help="Leave at 0 to keep the upper bound open.",
                )
        with channel_cols[1]:
            st.markdown('<div class="outlier-row-label">Subscriber Visibility</div>', unsafe_allow_html=True)
            with st.container(border=True):
                include_hidden = st.toggle(
                    "Include Hidden Subscriber Counts",
                    value=True,
                    key="outlier_page_include_hidden",
                )
                st.markdown(
                    '<div class="outlier-visibility-copy">Keep channels with hidden subscriber counts in the scan when subscriber size is not publicly visible.</div>',
                    unsafe_allow_html=True,
                )

        exclude_keywords_text = st.text_input(
            "Exclude Keywords",
            key="outlier_page_exclude_keywords",
            placeholder="news, reaction, podcast clips",
        )

        st.markdown('<div class="outlier-form-gap"></div>', unsafe_allow_html=True)
        with st.expander("More Filters", expanded=False):
            advanced_cols = st.columns(3, gap="medium")
            with advanced_cols[0]:
                search_pages = st.slider(
                    "Search Depth",
                    min_value=2,
                    max_value=4,
                    value=2,
                    step=1,
                    key="outlier_page_search_pages",
                    help="Each extra page adds about 100 search quota units.",
                )
            with advanced_cols[1]:
                baseline_channel_limit = st.slider(
                    "Baseline Channels",
                    min_value=10,
                    max_value=20,
                    value=15,
                    step=5,
                    key="outlier_page_baseline_channels",
                )
            with advanced_cols[2]:
                baseline_video_cap = st.slider(
                    "Baseline Uploads Per Channel",
                    min_value=10,
                    max_value=30,
                    value=20,
                    step=5,
                    key="outlier_page_baseline_videos",
                )

        _render_search_footer_note()

    if reset_clicked:
        _reset_search_state()
        st.rerun()

    if provider_counts["youtube"] <= 0:
        st.warning(
            "No YouTube API keys are configured. Add `YOUTUBE_API_KEYS` or `YOUTUBE_API_KEY` in Streamlit secrets to enable live outlier scans."
        )

    if submitted:
        if not niche_query.strip():
            st.session_state["outlier_page_error"] = "Enter a niche, topic, or keyword before running the scan."
            st.session_state.pop("outlier_page_result", None)
        else:
            try:
                published_after, published_before = _timeframe_to_window(
                    timeframe,
                    custom_dates=tuple(custom_dates) if timeframe == "Custom" and custom_dates else None,
                )
                if (published_before - published_after).days > 180:
                    raise ValueError("Custom timeframe cannot exceed 180 days in this version.")

                request = OutlierSearchRequest(
                    niche_query=niche_query.strip(),
                    published_after_iso=published_after.isoformat(),
                    published_before_iso=published_before.isoformat(),
                    region_code="" if region_code == "Any" else region_code,
                    relevance_language="" if language_code == "Any" else language_code,
                    language_strictness=str(strictness or "Strict").lower(),
                    include_hidden_subscribers=include_hidden,
                    min_subscribers=int(min_subscribers) if int(min_subscribers) > 0 else None,
                    max_subscribers=int(max_subscribers) if int(max_subscribers) > 0 else None,
                    min_views=int(min_views),
                    duration_preference=duration_preference,
                    freshness_days=FRESHNESS_OPTIONS.get(freshness_focus),
                    exclude_keywords=_parse_exclude_keywords(exclude_keywords_text),
                    match_mode="exact" if match_mode == "Exact Phrase" else "broad",
                    max_results=int(search_pages * 50),
                    baseline_channel_limit=baseline_channel_limit,
                    baseline_video_cap=baseline_video_cap,
                )
                with st.spinner("Scanning the niche, filtering the cohort, and scoring potential outliers..."):
                    result = search_outlier_videos(request)
                st.session_state["outlier_page_result"] = result
                st.session_state.pop("outlier_page_error", None)
                st.session_state.pop("outlier_page_ai_report", None)
                st.session_state.pop("outlier_page_ai_fingerprint", None)
            except Exception as exc:
                st.session_state["outlier_page_error"] = str(exc)
                st.session_state.pop("outlier_page_result", None)

    error_message = st.session_state.get("outlier_page_error")
    if error_message:
        st.error(error_message)

    result = st.session_state.get("outlier_page_result")
    if not result:
        _render_empty_state()
        _render_pre_search_methodology_teaser()
        _render_methodology_section()
        return

    for warning in result.warnings:
        st.markdown(f'<div class="outlier-panel-note">{escape(warning)}</div>', unsafe_allow_html=True)

    result_frame = result.to_frame()
    if result_frame.empty:
        st.info(
            "No strong matches survived the current filters. Broaden the timeframe, loosen the language strictness, or reduce the minimum views threshold."
        )
        _render_pre_search_methodology_teaser()
        _render_methodology_section()
        return

    sort_cols = st.columns([2.5, 1], gap="medium")
    with sort_cols[0]:
        _render_section_intro(
            "Top Outliers In This Scan",
            f"{len(result_frame):,} results surfaced from {result.scanned_videos:,} scanned uploads across {result.scanned_channels:,} channels.",
        )
    with sort_cols[1]:
        sort_label = st.selectbox(
            "Sort Results By",
            list(SORT_OPTIONS.keys()),
            index=0,
            key="outlier_page_sort",
        )

    sort_column, ascending = SORT_OPTIONS[sort_label]
    sorted_frame = result_frame.sort_values(sort_column, ascending=ascending).reset_index(drop=True)
    _render_metadata_cluster(result, sorted_frame)

    _render_result_cards(sorted_frame)

    table_df = sorted_frame[
        [
            "thumbnail_url",
            "video_title",
            "channel_title",
            "outlier_score",
            "views",
            "views_per_day",
            "duration_bucket",
            "language_confidence_label",
            "why_outlier",
            "research_cue",
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
            "duration_bucket": "Duration",
            "language_confidence_label": "Language Confidence",
            "why_outlier": "Why It Stands Out",
            "research_cue": "Research Cue",
        },
        inplace=True,
    )
    styled_dataframe(
        table_df,
        title="All Results In This Scan",
        precision=2,
        image_columns=["Thumbnail"],
    )

    _render_section_intro(
        "Breakout Snapshot",
        "Read the surfaced cohort through score, velocity, packaging, runtime, and signal quality before you hand the research to AI.",
    )
    summary_stats = _build_summary_stats(sorted_frame)
    summary_cols = st.columns(4, gap="medium")
    summary_cards = [
        ("Median Outlier Score", f"{summary_stats.get('median_outlier_score', 0):.1f}", "The middle-performing winner in this scan."),
        ("Median Views / Day", _format_int(summary_stats.get("median_views_per_day")), "Typical breakout velocity across surfaced videos."),
        ("Scanned Videos / Channels", f"{result.scanned_videos:,} / {result.scanned_channels:,}", "Coverage of the scanned cohort after filters."),
        ("High-Confidence Language Match", f"{summary_stats.get('high_language_share', 0):.1f}%", "Share of results with a high-confidence language match."),
    ]
    for idx, (label, value, detail) in enumerate(summary_cards):
        with summary_cols[idx]:
            _render_summary_card(label, value, detail)

    chart_top = st.columns([7, 5], gap="medium")
    with chart_top[0]:
        with st.container(border=True):
            _render_chart_shell(
                "Breakout Map",
                "Spot which videos are gaining velocity faster than channel size alone would suggest.",
            )
            breakout_fig = _breakout_scatter(sorted_frame)
            breakout_fig.update_layout(height=400)
            st.plotly_chart(breakout_fig, use_container_width=True)
    with chart_top[1]:
        with st.container(border=True):
            _render_chart_shell(
                "Outlier Score By Publish Age",
                "See whether the niche momentum is being driven by very recent uploads or older releases.",
            )
            age_fig = _age_bucket_chart(sorted_frame)
            age_fig.update_layout(height=400)
            st.plotly_chart(age_fig, use_container_width=True)

    chart_bottom = st.columns(3, gap="medium")
    with chart_bottom[0]:
        with st.container(border=True):
            _render_chart_shell(
                "Winning Video Lengths",
                "Identify which runtime buckets are overperforming in the current result set.",
            )
            duration_fig = _duration_chart(sorted_frame)
            duration_fig.update_layout(height=320)
            st.plotly_chart(duration_fig, use_container_width=True)
    with chart_bottom[1]:
        with st.container(border=True):
            _render_chart_shell(
                "Repeated Title Structures",
                "Read the packaging patterns that appear repeatedly across the strongest videos.",
            )
            title_pattern_fig = _title_pattern_chart(sorted_frame)
            title_pattern_fig.update_layout(height=320)
            st.plotly_chart(title_pattern_fig, use_container_width=True)
    with chart_bottom[2]:
        with st.container(border=True):
            _render_chart_shell(
                "Scan Quality",
                "Use this quick read to judge how clean and actionable the surfaced cohort is before you hand it to AI.",
            )
            _render_scan_quality_card(sorted_frame)

    _render_section_intro(
        "AI Research",
        "Turn the surfaced evidence into structured theme, title, angle, anomaly, and next-step cards after you review the results above.",
    )

    available_providers = [provider for provider in ("gemini", "openai") if get_provider_key_count(provider) > 0]
    if not available_providers:
        st.markdown(
            '<div class="outlier-panel-note">Add <code>GEMINI_API_KEYS</code> and/or <code>OPENAI_API_KEYS</code> to unlock the structured AI research layer.</div>',
            unsafe_allow_html=True,
        )
    else:
        default_provider = "gemini" if "gemini" in available_providers else available_providers[0]
        if (
            "outlier_page_ai_provider" not in st.session_state
            or st.session_state["outlier_page_ai_provider"] not in available_providers
        ):
            st.session_state["outlier_page_ai_provider"] = default_provider

        provider_cols = st.columns([1.05, 1.2, 0.95], gap="small")
        with provider_cols[0]:
            provider = st.selectbox(
                "AI Provider",
                available_providers,
                key="outlier_page_ai_provider",
                format_func=lambda value: AI_PROVIDER_LABELS.get(value, value.title()),
            )
        models = AI_MODELS[provider]
        if (
            "outlier_page_ai_model" not in st.session_state
            or st.session_state["outlier_page_ai_model"] not in models
        ):
            st.session_state["outlier_page_ai_model"] = models[0]
        with provider_cols[1]:
            model = st.selectbox("AI Model", models, key="outlier_page_ai_model")
        with provider_cols[2]:
            st.markdown('<div class="outlier-inline-field-label">Research Action</div>', unsafe_allow_html=True)
            trigger_ai = st.button("Generate AI Research", type="primary", use_container_width=True)

        fingerprint = _result_fingerprint(sorted_frame, result.request.niche_query)
        if st.session_state.get("outlier_page_ai_fingerprint") != fingerprint:
            st.session_state.pop("outlier_page_ai_report", None)

        if trigger_ai:
            query_context = {
                "niche_query": result.request.niche_query,
                "language": result.request.relevance_language or "Any",
                "region": result.request.region_code or "Any",
                "timeframe_start": result.request.published_after_iso,
                "timeframe_end": result.request.published_before_iso,
                "match_mode": result.request.match_mode,
            }
            summary_payload = {
                **summary_stats,
                **build_scan_quality_summary(sorted_frame),
                "scanned_videos": result.scanned_videos,
                "scanned_channels": result.scanned_channels,
                "baseline_channels": result.baseline_channels,
            }
            with st.spinner("Generating structured AI research..."):
                report = generate_outlier_ai_report(
                    provider=provider,
                    model=model,
                    query_context=query_context,
                    summary_stats=summary_payload,
                    result_frame=sorted_frame,
                )
            st.session_state["outlier_page_ai_report"] = report
            st.session_state["outlier_page_ai_fingerprint"] = fingerprint

        report = st.session_state.get("outlier_page_ai_report")
        if report:
            _render_ai_report(report)
        else:
            keyword_summary = build_title_keyword_summary(sorted_frame)
            preview_tokens = keyword_summary["keyword"].tolist()[:8] if not keyword_summary.empty else []
            if preview_tokens:
                _render_subsection_label("Repeated Title Keywords In The Scan")
                styled_keyword_chips(preview_tokens)
            st.markdown(
                '<div class="outlier-panel-note">Generate AI research to transform the surfaced videos into theme cards, title observations, repeatable angles, and next-step recommendations.</div>',
                unsafe_allow_html=True,
            )

    _render_methodology_section()
