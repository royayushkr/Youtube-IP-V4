from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.visualizations import kpi_row, section_header, styled_dataframe, styled_keyword_chips
from src.services.outlier_ai import OutlierAIReport, generate_outlier_ai_report
from src.services.outliers_finder import (
    DURATION_BUCKET_ORDER,
    SUBSCRIBER_BUCKETS,
    OutlierSearchRequest,
    build_age_bucket_summary,
    build_duration_summary,
    build_title_keyword_summary,
    build_title_pattern_summary,
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


def _inject_outlier_css() -> None:
    st.markdown(
        """
        <style>
        .outlier-page-hero {
            max-width: 980px;
            margin: 0 auto 1.25rem;
            text-align: center;
        }
        .outlier-page-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #F5F7FF;
            font-size: 12px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }
        .outlier-page-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, #FF5A5F, #FF2D2D);
            box-shadow: 0 0 16px rgba(255, 90, 95, 0.55);
        }
        .outlier-page-title {
            font-size: clamp(34px, 4vw, 56px);
            line-height: 1.02;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 0.75rem;
        }
        .outlier-page-subtitle {
            max-width: 720px;
            margin: 0 auto;
            font-size: 16px;
            color: #CED8EF;
        }
        .outlier-pill-row {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }
        .outlier-pill {
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            color: #D7DDF0;
            font-size: 12px;
        }
        .outlier-surface {
            border-radius: 24px;
            padding: 1.1rem 1.15rem 0.45rem;
            background: linear-gradient(180deg, rgba(37, 47, 69, 0.98) 0%, rgba(20, 28, 41, 0.98) 100%);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 18px 42px rgba(6, 11, 20, 0.34);
            margin-bottom: 1rem;
        }
        .outlier-surface-title {
            font-size: 17px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.15rem;
        }
        .outlier-surface-copy {
            font-size: 13px;
            color: #B7C3E1;
            margin-bottom: 0.9rem;
        }
        .outlier-filter-note {
            font-size: 12px;
            color: #A9B6D8;
            margin-top: 0.35rem;
        }
        .outlier-stat-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 1rem;
        }
        .outlier-stat-card {
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            background: linear-gradient(180deg, rgba(41, 52, 73, 0.96) 0%, rgba(28, 37, 55, 0.99) 100%);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .outlier-stat-label {
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8FA0C9;
            margin-bottom: 0.15rem;
        }
        .outlier-stat-value {
            font-size: 27px;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.05;
        }
        .outlier-stat-detail {
            margin-top: 0.2rem;
            font-size: 12px;
            color: #B9C4DE;
        }
        .outlier-result-card {
            border-radius: 20px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(40, 50, 70, 0.98) 0%, rgba(25, 34, 49, 0.99) 100%);
            border: 1px solid rgba(255,255,255,0.09);
            box-shadow: 0 14px 32px rgba(8, 12, 20, 0.26);
            margin-bottom: 1rem;
            min-height: 470px;
            display: flex;
            flex-direction: column;
        }
        .outlier-result-card img {
            width: 100%;
            height: 184px;
            object-fit: cover;
            display: block;
            background: rgba(255,255,255,0.04);
        }
        .outlier-result-body {
            padding: 0.95rem 1rem 1rem;
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
            font-size: 16px;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.35;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
            overflow: hidden;
            min-height: 44px;
        }
        .outlier-result-channel {
            font-size: 12px;
            color: #AFBAD8;
            margin-top: 0.2rem;
        }
        .outlier-result-score {
            flex-shrink: 0;
            min-width: 74px;
            text-align: center;
            padding: 0.42rem 0.6rem;
            border-radius: 16px;
            background: rgba(255, 59, 48, 0.15);
            border: 1px solid rgba(255, 59, 48, 0.30);
            color: #FFFFFF;
        }
        .outlier-result-score strong {
            display: block;
            font-size: 21px;
            line-height: 1;
        }
        .outlier-result-score span {
            display: block;
            font-size: 10px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #FFD7D4;
            margin-top: 0.1rem;
        }
        .outlier-result-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.7rem 0 0.8rem;
        }
        .outlier-result-metric {
            padding: 0.28rem 0.56rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 11px;
            color: #D3DBEF;
        }
        .outlier-result-bullets {
            margin: 0;
            padding-left: 1rem;
            color: #E6ECFA;
            font-size: 12px;
            line-height: 1.5;
        }
        .outlier-result-bullets li {
            margin-bottom: 0.22rem;
        }
        .outlier-result-link {
            margin-top: auto;
            display: inline-block;
            color: #FF9C94 !important;
            font-size: 12px;
            font-weight: 700;
            text-decoration: none;
            padding-top: 0.65rem;
        }
        .outlier-ai-card {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            background: linear-gradient(180deg, rgba(41, 52, 73, 0.96) 0%, rgba(28, 37, 55, 0.99) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 0.8rem;
        }
        .outlier-ai-card-title {
            font-size: 15px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.25rem;
        }
        .outlier-ai-card-body {
            font-size: 13px;
            color: #D7DDF0;
            line-height: 1.55;
        }
        .outlier-ai-card-support {
            margin-top: 0.4rem;
            font-size: 12px;
            color: #AEB8D6;
        }
        .outlier-method-card {
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 0.8rem;
        }
        .outlier-method-card h4 {
            color: #FFFFFF;
            margin-bottom: 0.45rem;
        }
        .outlier-method-card p,
        .outlier-method-card li {
            color: #CBD5EE;
            font-size: 13px;
            line-height: 1.6;
        }
        @media (max-width: 900px) {
            .outlier-stat-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
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


def _render_summary_cards(result_frame: pd.DataFrame, result_meta: Dict[str, Any]) -> None:
    summary = _build_summary_stats(result_frame)
    cards = [
        ("Median Score", f"{summary.get('median_outlier_score', 0):.1f}", "Middle-performing outlier in this scan"),
        ("Median Views / Day", _format_int(summary.get("median_views_per_day")), "Typical breakout velocity"),
        ("Median Views", _format_int(summary.get("median_views")), "Raw demand across surfaced winners"),
        ("High-Confidence Language Match", f"{summary.get('high_language_share', 0):.1f}%", "Share of results with strong language confidence"),
    ]
    html = "".join(
        f"""
        <div class="outlier-stat-card">
            <div class="outlier-stat-label">{escape(label)}</div>
            <div class="outlier-stat-value">{escape(value)}</div>
            <div class="outlier-stat-detail">{escape(detail)}</div>
        </div>
        """
        for label, value, detail in cards
    )
    st.markdown(f'<div class="outlier-stat-strip">{html}</div>', unsafe_allow_html=True)

    kpi_row(
        [
            {"label": "Videos Scanned", "value": f"{result_meta['scanned_videos']:,}", "icon": "🎬"},
            {"label": "Channels Scanned", "value": f"{result_meta['scanned_channels']:,}", "icon": "📺"},
            {"label": "Channel Baselines", "value": f"{result_meta['baseline_channels']:,}", "icon": "📉"},
            {"label": "Cache Policy", "value": result_meta["cache_policy"], "icon": "🧠"},
        ]
    )


def _render_result_cards(result_frame: pd.DataFrame) -> None:
    cols = st.columns(3)
    for idx, row in result_frame.head(9).iterrows():
        thumb_html = (
            f'<img src="{escape(str(row.get("thumbnail_url", "")))}" alt="{escape(str(row.get("video_title", "")))}" />'
            if str(row.get("thumbnail_url", "")).strip()
            else ""
        )
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="outlier-result-card">
                    {thumb_html}
                    <div class="outlier-result-body">
                        <div class="outlier-result-top">
                            <div>
                                <div class="outlier-result-title">{escape(str(row.get("video_title", "")))}</div>
                                <div class="outlier-result-channel">{escape(str(row.get("channel_title", "")))}</div>
                            </div>
                            <div class="outlier-result-score">
                                <strong>{float(row.get("outlier_score", 0)):.1f}</strong>
                                <span>Score</span>
                            </div>
                        </div>
                        <div class="outlier-result-metrics">
                            <span class="outlier-result-metric">{_format_int(row.get("views"))} views</span>
                            <span class="outlier-result-metric">{_format_int(row.get("views_per_day"))} / day</span>
                            <span class="outlier-result-metric">{_format_subscribers(row.get("channel_subscriber_count"), bool(row.get("hidden_subscriber_count")))} subs</span>
                        </div>
                        <ul class="outlier-result-bullets">
                            <li>{escape(str(row.get("why_outlier", "")))}</li>
                            <li>{escape(str(row.get("research_cue", "")))}</li>
                        </ul>
                        <a class="outlier-result-link" href="{escape(str(row.get("video_url", "")))}" target="_blank">Open on YouTube</a>
                    </div>
                </div>
                """,
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
        hover_data={
            "channel_title": True,
            "views": ":,",
            "outlier_score": ":.1f",
            "age_days": ":.1f",
            "language_confidence_label": True,
            "subscribers_log10": False,
        },
        title="Breakout Map: Which Videos Are Scaling Faster Than Channel Size Suggests?",
        labels={
            "subscribers_log10": "Channel Subscribers (log10 + 1)",
            "views_per_day": "Views Per Day",
            "age_bucket": "Publish Age Bucket",
        },
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.18)"), sizemin=12))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D5DCF1"),
        legend_title_text="Publish Age Bucket",
        margin=dict(l=8, r=8, t=56, b=8),
    )
    return fig


def _age_bucket_chart(result_frame: pd.DataFrame):
    summary = build_age_bucket_summary(result_frame)
    fig = px.bar(
        summary,
        x="age_bucket",
        y="median_outlier_score",
        color="median_views_per_day",
        text="outlier_count",
        title="Are Recent Uploads Or Older Uploads Driving The Outliers?",
        labels={
            "age_bucket": "Publish Age Bucket",
            "median_outlier_score": "Median Outlier Score",
            "median_views_per_day": "Median Views Per Day",
        },
        color_continuous_scale="Reds",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Median Outlier Score: %{y:.1f}<br>"
            "Median Views Per Day: %{marker.color:.0f}<br>"
            "Outlier Count: %{text}<extra></extra>"
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D5DCF1"),
        margin=dict(l=8, r=8, t=56, b=8),
    )
    return fig


def _duration_chart(result_frame: pd.DataFrame):
    summary = build_duration_summary(result_frame)
    fig = px.bar(
        summary,
        x="duration_bucket",
        y="outlier_count",
        color="median_outlier_score",
        title="Which Video Lengths Are Overperforming In This Niche?",
        labels={
            "duration_bucket": "Duration Bucket",
            "outlier_count": "Outlier Count",
            "median_outlier_score": "Median Outlier Score",
        },
        color_continuous_scale="Reds",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Outlier Count: %{y}<br>"
            "Median Outlier Score: %{marker.color:.1f}<extra></extra>"
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D5DCF1"),
        margin=dict(l=8, r=8, t=56, b=8),
    )
    return fig


def _title_pattern_chart(result_frame: pd.DataFrame):
    summary = build_title_pattern_summary(result_frame)
    fig = px.bar(
        summary,
        x="title_pattern",
        y="outlier_count",
        color="median_outlier_score",
        title="What Title Structures Repeat Across The Outliers?",
        labels={
            "title_pattern": "Title Pattern",
            "outlier_count": "Outlier Count",
            "median_outlier_score": "Median Outlier Score",
        },
        color_continuous_scale="Reds",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Outlier Count: %{y}<br>"
            "Median Outlier Score: %{marker.color:.1f}<extra></extra>"
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D5DCF1"),
        margin=dict(l=8, r=8, t=56, b=8),
    )
    return fig


def _render_insight_cards(title: str, cards: Tuple[Any, ...], columns: int = 3) -> None:
    if not cards:
        return
    st.markdown(f"**{title}**")
    cols = st.columns(columns)
    for idx, card in enumerate(cards):
        with cols[idx % columns]:
            st.markdown(
                f"""
                <div class="outlier-ai-card">
                    <div class="outlier-ai-card-title">{escape(card.title)}</div>
                    <div class="outlier-ai-card-body">{escape(card.body)}</div>
                    <div class="outlier-ai-card-support">{escape(card.support)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_ai_report(report: OutlierAIReport) -> None:
    st.markdown(
        f"""
        <div class="outlier-surface">
            <div class="outlier-surface-title">{escape(report.executive_headline)}</div>
            <div class="outlier-surface-copy">{escape(report.key_takeaway)}</div>
            <div class="outlier-filter-note">Confidence: {escape(report.confidence_label)}  •  Provider: {escape(report.provider.title())}  •  Model: {escape(report.model)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_insight_cards("Breakout Themes", report.breakout_themes, columns=2)
    _render_insight_cards("Title Pattern Observations", report.title_patterns, columns=2)
    _render_insight_cards("Repeatable Content Angles", report.repeatable_angles, columns=3)
    if report.next_steps:
        st.markdown("**What To Do Next**")
        for step in report.next_steps:
            st.markdown(
                f"""
                <div class="outlier-ai-card">
                    <div class="outlier-ai-card-body">{escape(step)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if report.warnings:
        st.warning(" | ".join(report.warnings))
    if report.raw_fallback:
        st.caption("Fallback AI summary:")
        st.markdown(report.raw_fallback)


def _render_methodology_tab() -> None:
    st.markdown(
        """
        <div class="outlier-method-card">
            <h4>What The Outlier Score Means</h4>
            <p>The score is a public-performance heuristic, not a private YouTube growth signal. It blends channel-baseline lift, peer percentile, engagement percentile, and recency boost into one 0-100 score.</p>
        </div>
        <div class="outlier-method-card">
            <h4>Metric Definitions</h4>
            <ul>
                <li><strong>Views Per Day</strong>: total views divided by video age in days.</li>
                <li><strong>Engagement Rate</strong>: (likes + comments) divided by views.</li>
                <li><strong>Views Per Subscriber</strong>: views divided by public subscriber count when available.</li>
                <li><strong>Baseline Lift</strong>: how far this video is above the channel's recent median views/day and engagement.</li>
                <li><strong>Language Confidence</strong>: metadata plus title-script heuristic, not a guaranteed language classifier.</li>
            </ul>
        </div>
        <div class="outlier-method-card">
            <h4>How Filters Work</h4>
            <ul>
                <li><strong>Exact Phrase</strong> quotes the niche query and also checks title/description text after the API returns results.</li>
                <li><strong>Exclude Keywords</strong> are applied in the search query and again as a post-filter.</li>
                <li><strong>Region</strong> filters videos viewable in that geography, not necessarily creators from that geography.</li>
                <li><strong>Freshness</strong> narrows results to more recent uploads inside the chosen timeframe.</li>
            </ul>
        </div>
        <div class="outlier-method-card">
            <h4>API Limitations And Caveats</h4>
            <ul>
                <li>No impressions, CTR, retention, watch time, or traffic-source data is available through this workflow.</li>
                <li>YouTube search is sampled and ranked, so this is not an exhaustive view of all videos in the niche.</li>
                <li>Subscriber counts can be hidden or rounded, which weakens channel-size filtering for some results.</li>
                <li>Language filtering is heuristic-based; strict mode reduces noise but cannot guarantee perfect filtering.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_discover_empty_state() -> None:
    st.markdown(
        """
        <div class="outlier-surface">
            <div class="outlier-surface-title">Start With A Niche Query</div>
            <div class="outlier-surface-copy">Search for a topic, tighten the filters, and the page will surface public videos that are outperforming their likely baseline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    _inject_outlier_css()

    provider_counts = {
        "youtube": get_provider_key_count("youtube"),
        "gemini": get_provider_key_count("gemini"),
        "openai": get_provider_key_count("openai"),
    }

    st.markdown(
        """
        <div class="outlier-page-hero">
            <div class="outlier-page-kicker"><span class="outlier-page-dot"></span>Outlier Finder</div>
            <div class="outlier-page-title">Find Your Next Viral Video Before The Niche Gets Crowded.</div>
            <div class="outlier-page-subtitle">
                Discover breakout content in any niche, filter the noise, and turn public outlier signals into clearer content strategy.
            </div>
            <div class="outlier-pill-row">
                <span class="outlier-pill">Public YouTube API data only</span>
                <span class="outlier-pill">Quota-aware caching</span>
                <span class="outlier-pill">Structured AI research</span>
                <span class="outlier-pill">Explainable outlier scoring</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prefill_note = st.session_state.pop("outlier_page_prefill_note", None)
    if prefill_note:
        st.info(prefill_note)

    with st.form("outlier_finder_search_form"):
        st.markdown(
            """
            <div class="outlier-surface">
                <div class="outlier-surface-title">Search A Niche And Tighten The Signal</div>
                <div class="outlier-surface-copy">Use precise filters when you need cleaner research, or keep them broad when you want a wider scouting pass.</div>
            """,
            unsafe_allow_html=True,
        )

        input_cols = st.columns([3.2, 1.2], gap="small")
        with input_cols[0]:
            niche_query = st.text_input(
                "Niche or keyword",
                key="outlier_page_query",
                placeholder="AI automation, documentary storytelling, science shorts, luxury fitness...",
            )
        with input_cols[1]:
            submitted = st.form_submit_button(
                "Run Outlier Scan",
                type="primary",
                use_container_width=True,
                disabled=provider_counts["youtube"] <= 0,
            )

        top_filter_cols = st.columns(4)
        with top_filter_cols[0]:
            timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS, index=1, key="outlier_page_timeframe")
        with top_filter_cols[1]:
            match_mode = st.segmented_control(
                "Match mode",
                ["Broad", "Exact Phrase"],
                key="outlier_page_match_mode",
                selection_mode="single",
                default="Broad",
            )
        with top_filter_cols[2]:
            region_code = st.selectbox("Region", REGION_OPTIONS, index=0, key="outlier_page_region")
        with top_filter_cols[3]:
            language_code = st.selectbox("Language", LANGUAGE_OPTIONS, index=0, key="outlier_page_language")

        if timeframe == "Custom":
            default_end = datetime.now(timezone.utc).date()
            default_start = default_end - timedelta(days=30)
            custom_dates = st.date_input(
                "Custom date range",
                value=(default_start, default_end),
                max_value=default_end,
                key="outlier_page_custom_dates",
            )
        else:
            custom_dates = None

        filter_cols = st.columns(4)
        with filter_cols[0]:
            freshness_focus = st.selectbox(
                "Upload recency focus",
                list(FRESHNESS_OPTIONS.keys()),
                index=0,
                key="outlier_page_freshness",
            )
        with filter_cols[1]:
            duration_preference = st.selectbox(
                "Duration preference",
                DURATION_OPTIONS,
                index=0,
                key="outlier_page_duration",
            )
        with filter_cols[2]:
            strictness = st.segmented_control(
                "Language strictness",
                STRICTNESS_OPTIONS,
                key="outlier_page_language_strictness",
                selection_mode="single",
                default="Strict",
            )
        with filter_cols[3]:
            min_views = st.selectbox(
                "Minimum views",
                [0, 1_000, 5_000, 10_000, 50_000, 100_000],
                index=0,
                key="outlier_page_min_views",
                format_func=lambda value: "No minimum" if value == 0 else f"{value:,}+",
            )

        numeric_cols = st.columns(3)
        with numeric_cols[0]:
            min_subscribers = st.number_input(
                "Minimum subscribers",
                min_value=0,
                value=0,
                step=1_000,
                key="outlier_page_min_subscribers",
            )
        with numeric_cols[1]:
            max_subscribers = st.number_input(
                "Maximum subscribers",
                min_value=0,
                value=0,
                step=1_000,
                key="outlier_page_max_subscribers",
                help="Leave at 0 for no upper limit.",
            )
        with numeric_cols[2]:
            include_hidden = st.toggle(
                "Include hidden subscriber counts",
                value=True,
                key="outlier_page_include_hidden",
            )

        exclude_keywords_text = st.text_input(
            "Exclude keywords",
            key="outlier_page_exclude_keywords",
            placeholder="news, reaction, podcast clips",
        )
        st.markdown(
            "<div class='outlier-filter-note'>Search results are still limited to the cohort returned by the official YouTube API. Strict language filtering reduces noise but is heuristic-based.</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Advanced search controls", expanded=False):
            advanced_cols = st.columns(3)
            with advanced_cols[0]:
                search_pages = st.slider(
                    "Search pages",
                    min_value=2,
                    max_value=4,
                    value=2,
                    step=1,
                    key="outlier_page_search_pages",
                    help="Each extra page adds roughly 100 search quota units.",
                )
            with advanced_cols[1]:
                baseline_channel_limit = st.slider(
                    "Baseline channels",
                    min_value=10,
                    max_value=20,
                    value=15,
                    step=5,
                    key="outlier_page_baseline_channels",
                )
            with advanced_cols[2]:
                baseline_video_cap = st.slider(
                    "Baseline uploads per channel",
                    min_value=10,
                    max_value=30,
                    value=20,
                    step=5,
                    key="outlier_page_baseline_videos",
                )

        st.markdown("</div>", unsafe_allow_html=True)

    if provider_counts["youtube"] <= 0:
        st.warning("No YouTube API keys are configured. Add `YOUTUBE_API_KEYS` or `YOUTUBE_API_KEY` in Streamlit secrets to enable live outlier scans.")

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

    discover_tab, ai_tab, results_tab, methodology_tab = st.tabs(
        ["Discover", "AI Research", "Results", "Methodology"]
    )

    result = st.session_state.get("outlier_page_result")
    if not result:
        with discover_tab:
            _render_discover_empty_state()
        with ai_tab:
            st.info("Run a scan first to unlock the structured AI research panel.")
        with results_tab:
            st.info("Run a scan first to view standardized result cards and the sortable table.")
        with methodology_tab:
            _render_methodology_tab()
        return

    for warning in result.warnings:
        st.warning(warning)

    result_frame = result.to_frame()
    if result_frame.empty:
        with discover_tab:
            st.info("No strong matches survived the current filters. Broaden the timeframe, loosen the language strictness, or reduce the minimum views threshold.")
        with ai_tab:
            st.info("No AI research is available because the scan returned no qualifying videos.")
        with results_tab:
            st.info("No result cards are available for the current filter set.")
        with methodology_tab:
            _render_methodology_tab()
        return

    sort_label = st.selectbox(
        "Sort results by",
        list(SORT_OPTIONS.keys()),
        index=0,
        key="outlier_page_sort",
    )
    sort_column, ascending = SORT_OPTIONS[sort_label]
    sorted_frame = result_frame.sort_values(sort_column, ascending=ascending).reset_index(drop=True)

    active_filters = [result.request.niche_query]
    if result.request.relevance_language:
        active_filters.append(f"Language: {result.request.relevance_language.upper()} ({result.request.language_strictness.title()})")
    if result.request.region_code:
        active_filters.append(f"Region: {result.request.region_code}")
    if result.request.duration_preference != "Any":
        active_filters.append(result.request.duration_preference)
    if result.request.min_views > 0:
        active_filters.append(f"{result.request.min_views:,}+ views")
    styled_keyword_chips(active_filters[:8])

    result_meta = {
        "scanned_videos": result.scanned_videos,
        "scanned_channels": result.scanned_channels,
        "baseline_channels": result.baseline_channels,
        "cache_policy": result.cache_policy,
    }

    with discover_tab:
        section_header("Breakout Snapshot", icon="📡")
        _render_summary_cards(sorted_frame, result_meta)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(_breakout_scatter(sorted_frame), use_container_width=True)
        with chart_cols[1]:
            st.plotly_chart(_age_bucket_chart(sorted_frame), use_container_width=True)
        secondary_cols = st.columns(2)
        with secondary_cols[0]:
            st.plotly_chart(_duration_chart(sorted_frame), use_container_width=True)
        with secondary_cols[1]:
            st.plotly_chart(_title_pattern_chart(sorted_frame), use_container_width=True)

    with ai_tab:
        section_header("AI Research Panel", icon="🧠")
        available_providers = [
            provider for provider in ("gemini", "openai") if get_provider_key_count(provider) > 0
        ]
        if not available_providers:
            st.info("Add `GEMINI_API_KEYS` and/or `OPENAI_API_KEYS` to unlock the structured AI research layer.")
        else:
            default_provider = "gemini" if "gemini" in available_providers else available_providers[0]
            if (
                "outlier_page_ai_provider" not in st.session_state
                or st.session_state["outlier_page_ai_provider"] not in available_providers
            ):
                st.session_state["outlier_page_ai_provider"] = default_provider
            provider = st.selectbox(
                "AI provider",
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
            model = st.selectbox("AI model", models, key="outlier_page_ai_model")

            fingerprint = _result_fingerprint(sorted_frame, result.request.niche_query)
            if st.session_state.get("outlier_page_ai_fingerprint") != fingerprint:
                st.session_state.pop("outlier_page_ai_report", None)

            if st.button("Generate Structured AI Research", type="primary", use_container_width=True):
                query_context = {
                    "niche_query": result.request.niche_query,
                    "language": result.request.relevance_language or "Any",
                    "region": result.request.region_code or "Any",
                    "timeframe_start": result.request.published_after_iso,
                    "timeframe_end": result.request.published_before_iso,
                    "match_mode": result.request.match_mode,
                }
                summary_stats = {
                    **_build_summary_stats(sorted_frame),
                    "scanned_videos": result.scanned_videos,
                    "scanned_channels": result.scanned_channels,
                    "baseline_channels": result.baseline_channels,
                }
                with st.spinner("Generating structured AI research..."):
                    report = generate_outlier_ai_report(
                        provider=provider,
                        model=model,
                        query_context=query_context,
                        summary_stats=summary_stats,
                        result_frame=sorted_frame,
                    )
                st.session_state["outlier_page_ai_report"] = report
                st.session_state["outlier_page_ai_fingerprint"] = fingerprint

            report = st.session_state.get("outlier_page_ai_report")
            if report:
                _render_ai_report(report)
            else:
                st.info("Generate AI research to turn the outlier scan into theme cards, title observations, repeatable angles, and next-step recommendations.")

    with results_tab:
        section_header("Scanned Outlier Videos", icon="🎬")
        keyword_summary = build_title_keyword_summary(sorted_frame)
        if not keyword_summary.empty:
            st.markdown("**Repeated title keywords across the outliers**")
            styled_keyword_chips(keyword_summary["keyword"].tolist())
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
            title="All surfaced results",
            precision=2,
            image_columns=["Thumbnail"],
        )

    with methodology_tab:
        section_header("Methodology And Caveats", icon="📘")
        _render_methodology_tab()
