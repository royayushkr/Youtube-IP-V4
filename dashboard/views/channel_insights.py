from __future__ import annotations

from html import escape
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st

from dashboard.components.visualizations import (
    kpi_row,
    plotly_bar_chart,
    plotly_line_chart,
    section_header,
    styled_dataframe,
    styled_keyword_chips,
)
from src.services.channel_insights_service import (
    TOPIC_MODE_BERTOPIC_OPTIONAL,
    TOPIC_MODE_HEURISTIC,
    list_connected_channels,
    load_channel_insights,
    refresh_channel_insights,
)
from src.services.model_artifact_service import get_bertopic_artifact_status
from src.services.google_oauth_service import (
    build_google_authorization_url,
    clear_google_oauth_session,
    complete_google_oauth_callback,
    get_google_credentials,
    get_google_profile,
    oauth_configured,
    oauth_ready_error,
)
from src.services.youtube_owner_analytics_service import list_owned_channels


STATE_KEYS = (
    "channel_insights_selected_channel",
    "channel_insights_input",
    "channel_insights_force_refresh",
    "channel_insights_topic_mode",
    "channel_insights_error",
)


def _inject_channel_insights_css() -> None:
    st.markdown(
        """
        <style>
        .channel-insights-page {
            max-width: var(--app-page-width);
            margin: 0 auto;
        }
        .ci-hero {
            max-width: 930px;
            margin: 0 auto 1.4rem;
            text-align: center;
        }
        .ci-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.78rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: #F7F8FC;
            font-size: 12px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.95rem;
        }
        .ci-kicker-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, #A855F7, #8B5CF6);
            box-shadow: 0 0 16px rgba(139, 92, 246, 0.45);
        }
        .ci-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            font-size: clamp(34px, 3.8vw, 50px);
            line-height: 1.02;
            font-weight: 700;
            color: #F7F8FC;
            letter-spacing: -0.04em;
            margin-bottom: 0.8rem;
        }
        .ci-subtitle {
            color: #B8C1DA;
            font-size: 16px;
            line-height: 1.62;
            max-width: 760px;
            margin: 0 auto;
        }
        .ci-card {
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.08);
            background:
                radial-gradient(circle at top left, rgba(139, 92, 246, 0.10) 0%, transparent 30%),
                linear-gradient(180deg, rgba(26, 33, 64, 0.95) 0%, rgba(15, 19, 36, 0.98) 100%);
            box-shadow: 0 20px 46px rgba(3, 6, 20, 0.40);
            padding: 1.2rem 1.25rem;
            margin-bottom: 1rem;
        }
        .ci-card-title {
            font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
            color: #F7F8FC;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .ci-card-copy {
            color: #B8C1DA;
            font-size: 13px;
            line-height: 1.55;
        }
        .ci-summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.95rem;
        }
        .ci-summary-item {
            padding: 0.75rem 0.85rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
        }
        .ci-summary-label {
            color: #8993B2;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.22rem;
        }
        .ci-summary-value {
            color: #F7F8FC;
            font-size: 14px;
            font-weight: 700;
            line-height: 1.5;
        }
        .ci-list {
            margin: 0;
            padding-left: 1rem;
            color: #D7DDF0;
            font-size: 13px;
            line-height: 1.6;
        }
        .ci-list li {
            margin-bottom: 0.35rem;
        }
        .ci-theme-card {
            padding: 0.85rem 0.95rem;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            margin-bottom: 0.8rem;
        }
        .ci-theme-title {
            color: #F7F8FC;
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .ci-theme-copy {
            color: #B8C1DA;
            font-size: 13px;
            line-height: 1.6;
        }
        .ci-source-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(139, 92, 246, 0.14);
            border: 1px solid rgba(196, 181, 253, 0.16);
            color: #F7F8FC;
            font-size: 11px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .ci-note {
            color: #97A2C3;
            font-size: 12px;
            line-height: 1.55;
        }
        .ci-empty {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            border: 1px dashed rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.02);
            color: #B8C1DA;
            font-size: 13px;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_int(value: Any) -> str:
    if value is None or value == "":
        return "0"
    return f"{int(round(float(value))):,}"


def _format_pct(value: Any) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def _format_ratio_pct(value: Any) -> str:
    number = float(value or 0)
    if number <= 1.0:
        number *= 100
    return f"{number:.1f}%"


def _format_hours(value: Any) -> str:
    return f"{float(value or 0):,.1f}h"


def _format_duration_seconds(value: Any) -> str:
    total_seconds = int(float(value or 0))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _history_delta_text(value: float, suffix: str = "") -> str:
    if value > 0:
        return f"+{value:.1f}{suffix}"
    if value < 0:
        return f"{value:.1f}{suffix}"
    return f"0.0{suffix}"


def _queue_outlier_finder_theme(theme: str, channel_title: str) -> None:
    st.session_state["app_page"] = "Outlier Finder"
    st.session_state["outlier_page_query"] = theme
    st.session_state["outlier_page_prefill_note"] = (
        f"Suggested from {channel_title}'s latest channel insights. Use this theme as a niche seed and refine it before scanning."
    )


def _topic_mode_label(topic_mode: str) -> str:
    if topic_mode == TOPIC_MODE_BERTOPIC_OPTIONAL:
        return "Model-Backed Topics (Beta)"
    return "Heuristic Topics"


def _artifact_status_label(state: str) -> str:
    mapping = {
        "disabled": "Unavailable",
        "unconfigured": "Unavailable",
        "download_required": "Download Required",
        "ready": "Ready",
        "invalid": "Failed / Fallback Active",
        "load_failed": "Failed / Fallback Active",
        "transform_failed": "Failed / Fallback Active",
    }
    return mapping.get(str(state or "").strip().lower(), "Unavailable")


def _render_hero() -> None:
    st.markdown(
        """
        <div class="channel-insights-page">
            <div class="ci-hero">
                <div class="ci-kicker"><span class="ci-kicker-dot"></span>Channel Insights</div>
                <div class="ci-title">Connect A Channel And Turn Real Performance Signals Into Better Topic Decisions.</div>
                <div class="ci-subtitle">
                    Track a creator channel over time, persist refresh snapshots, find the themes and formats that are actually working,
                    and turn those public signals into grounded next-topic recommendations. Connect Google OAuth if you want owner-only watch-time, retention, and thumbnail signal overlays.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _handle_oauth_callback() -> None:
    try:
        profile = complete_google_oauth_callback()
    except Exception as exc:
        st.session_state["channel_insights_error"] = str(exc)
    else:
        if profile is not None:
            st.session_state.pop("channel_insights_error", None)


def _owned_channels() -> List[Dict[str, Any]]:
    credentials = get_google_credentials()
    if credentials is None:
        return []
    try:
        return list_owned_channels(credentials)
    except Exception as exc:
        st.session_state["channel_insights_error"] = f"Google account is connected, but owned channels could not be loaded: {exc}"
        return []


def _render_connect_card(connected_channels: List[Dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="ci-card">
            <div class="ci-card-title">Connect A Channel</div>
            <div class="ci-card-copy">
                Add any public channel by URL, handle, or channel ID. If you also connect Google OAuth, Channel Insights can layer in owner-only watch time, retention, thumbnail impressions, and thumbnail click-through signal during the same refresh.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    credentials = get_google_credentials()
    profile = get_google_profile()
    owned_channels = _owned_channels() if credentials is not None else []
    artifact_status = get_bertopic_artifact_status()

    config_cols = st.columns([1.45, 1], gap="large")
    with config_cols[0]:
        with st.form("channel_insights_connect_form"):
            channel_input = st.text_input(
                "Channel URL, Handle, Or Channel ID",
                key="channel_insights_input",
                placeholder="https://www.youtube.com/@veritasium or @veritasium",
            )
            action_cols = st.columns([1.35, 1], gap="small")
            with action_cols[0]:
                connect_clicked = st.form_submit_button("Add Or Analyze Channel", type="primary", use_container_width=True)
            with action_cols[1]:
                force_refresh = st.toggle(
                    "Force Live Refresh",
                    key="channel_insights_force_refresh",
                    help="Bypass dataset-backed rows and pull public channel data from the YouTube API again.",
                )

        if connect_clicked:
            if not channel_input.strip():
                st.session_state["channel_insights_error"] = "Enter a public channel URL, handle, or channel ID first."
            else:
                with st.spinner("Analyzing the channel and storing a fresh snapshot..."):
                    try:
                        payload = refresh_channel_insights(
                            channel_input.strip(),
                            force_refresh=force_refresh,
                            topic_mode=st.session_state.get("channel_insights_topic_mode", TOPIC_MODE_HEURISTIC),
                            owner_credentials=credentials,
                        )
                    except Exception as exc:
                        st.session_state["channel_insights_error"] = str(exc)
                    else:
                        st.session_state["channel_insights_selected_channel"] = payload["channel"]["channel_id"]
                        st.session_state.pop("channel_insights_error", None)
                        st.rerun()

    with config_cols[1]:
        st.markdown(
            """
            <div class="ci-card">
                <div class="ci-card-title">Connect Google Account</div>
                <div class="ci-card-copy">
                    Owner metrics are optional. Connect Google OAuth if you want private channel analytics like watch time, average percentage viewed, and thumbnail impression signals for channels you own.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not oauth_configured():
            st.markdown(f"<div class='ci-empty'>{escape(oauth_ready_error())}</div>", unsafe_allow_html=True)
        elif credentials is None:
            try:
                auth_url = build_google_authorization_url()
            except Exception as exc:
                st.markdown(f"<div class='ci-empty'>{escape(str(exc))}</div>", unsafe_allow_html=True)
            else:
                st.link_button("Connect Google Account", auth_url, use_container_width=True)
                st.caption("Scopes: Google Sign-In, YouTube Read-Only, and YouTube Analytics Read-Only.")
        else:
            connected_label = profile.get("email") or profile.get("name") or "Connected Google Account"
            st.success(f"Connected: {connected_label}")
            metric_cols = st.columns(2, gap="small")
            with metric_cols[0]:
                st.metric("Owned Channels", len(owned_channels))
            with metric_cols[1]:
                if st.button("Disconnect", use_container_width=True, key="channel_insights_disconnect_google"):
                    clear_google_oauth_session()
                    st.rerun()

            if owned_channels:
                owned_choice = st.selectbox(
                    "Your YouTube Channels",
                    owned_channels,
                    key="channel_insights_owned_channel",
                    format_func=lambda row: row["channel_title"],
                )
                if st.button("Track My Selected Channel", type="primary", use_container_width=True, key="channel_insights_track_owned"):
                    with st.spinner("Analyzing your owner channel and storing a fresh snapshot..."):
                        try:
                            payload = refresh_channel_insights(
                                owned_choice["channel_id"],
                                topic_mode=st.session_state.get("channel_insights_topic_mode", TOPIC_MODE_HEURISTIC),
                                owner_credentials=credentials,
                            )
                        except Exception as exc:
                            st.session_state["channel_insights_error"] = str(exc)
                        else:
                            st.session_state["channel_insights_selected_channel"] = payload["channel"]["channel_id"]
                            st.session_state.pop("channel_insights_error", None)
                            st.rerun()
            else:
                st.info("This Google account did not return any owned YouTube channels.")

        tracked_options = connected_channels
        selected_channel_id = st.session_state.get("channel_insights_selected_channel", "")
        if tracked_options and selected_channel_id not in {row["channel_id"] for row in tracked_options}:
            selected_channel_id = tracked_options[0]["channel_id"]
            st.session_state["channel_insights_selected_channel"] = selected_channel_id

        if tracked_options:
            choice = st.selectbox(
                "Tracked Channels",
                tracked_options,
                index=next((i for i, row in enumerate(tracked_options) if row["channel_id"] == selected_channel_id), 0),
                format_func=lambda row: row["channel_title"],
            )
            st.session_state["channel_insights_selected_channel"] = choice["channel_id"]
        else:
            st.markdown("<div class='ci-empty'>No tracked channels yet. Add one on the left to unlock persisted snapshots and theme history.</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="ci-card" style="margin-top:0.85rem;">
                <div class="ci-card-title">Current Workspace</div>
                <div class="ci-summary-grid">
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Tracked Channels</div>
                        <div class="ci-summary-value">{len(connected_channels)}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Refresh Mode</div>
                        <div class="ci-summary-value">Manual Snapshots</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Owner Access</div>
                        <div class="ci-summary-value">{'Connected' if credentials is not None else 'Public Only'}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">OAuth Setup</div>
                        <div class="ci-summary-value">{'Ready' if oauth_configured() else 'Missing Config'}</div>
                    </div>
                </div>
                <div class="ci-note" style="margin-top:0.85rem;">V1 stores manual snapshots over time. Google OAuth unlocks owner-only metrics during the current session, but background daily jobs still need an external scheduler later.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="ci-card" style="margin-top:0.85rem;">
                <div class="ci-card-title">Experimental Topic Model</div>
                <div class="ci-card-copy">
                    Keep the default heuristic topic flow for the safest path, or switch to the optional BERTopic beta when artifact settings are configured.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.selectbox(
            "Topic Analysis Mode",
            options=[TOPIC_MODE_HEURISTIC, TOPIC_MODE_BERTOPIC_OPTIONAL],
            key="channel_insights_topic_mode",
            format_func=_topic_mode_label,
        )
        st.caption(f"Artifact Status: {_artifact_status_label(artifact_status.state)}")
        if artifact_status.state == "ready":
            st.success(artifact_status.message or "BERTopic bundle is ready.")
        elif artifact_status.state == "download_required":
            st.info(artifact_status.message or "The BERTopic bundle will download when you run the beta mode.")
        elif artifact_status.state in {"invalid"}:
            st.warning(artifact_status.failure_reason or artifact_status.message or "BERTopic beta mode is currently unavailable.")
        else:
            st.caption(artifact_status.message or "The heuristic mode remains the default until artifact configuration is complete.")


def _render_summary_action_row(payload: Dict[str, Any]) -> None:
    channel = payload["channel"]
    summary = payload["summary"]
    credentials = get_google_credentials()
    owner_metrics_active = bool(summary.get("owner_metrics_available"))
    topic_mode_used = summary.get("topic_mode_used", TOPIC_MODE_HEURISTIC)
    topic_mode_requested = summary.get("topic_mode_requested", TOPIC_MODE_HEURISTIC)
    topic_model_message = summary.get("topic_model_message", "")
    topic_model_failure_reason = summary.get("topic_model_failure_reason", "")
    artifact_status = payload.get("topic_artifact_status") or get_bertopic_artifact_status()
    action_cols = st.columns([2.2, 1.1, 1], gap="large")
    with action_cols[0]:
        st.markdown(
            f"""
            <div class="ci-card">
                <div class="ci-source-pill">{escape('Public + Owner Metrics' if owner_metrics_active else payload['source'].replace('_', ' ').title())}</div>
                <div class="ci-card-title" style="margin-top:0.65rem;">{escape(channel['channel_title'])}</div>
                <div class="ci-card-copy">
                    Latest Snapshot: {escape(payload['snapshot_at'])}<br/>
                    {'Owner-only metrics were blended into this snapshot using Google OAuth.' if owner_metrics_active else 'Public channel analytics only. Connect Google OAuth if you want owner-only watch-time, retention, and thumbnail signals.'}<br/>
                    Topic Mode Used: {escape(_topic_mode_label(topic_mode_used))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_cols[1]:
        if st.button("Refresh Analysis", type="primary", use_container_width=True):
            with st.spinner("Refreshing channel insights and writing a new snapshot..."):
                try:
                    fresh_payload = refresh_channel_insights(
                        channel["channel_id"],
                        force_refresh=False,
                        topic_mode=st.session_state.get("channel_insights_topic_mode", TOPIC_MODE_HEURISTIC),
                        owner_credentials=credentials,
                    )
                except Exception as exc:
                    st.session_state["channel_insights_error"] = str(exc)
                else:
                    st.session_state["channel_insights_selected_channel"] = fresh_payload["channel"]["channel_id"]
                    st.session_state.pop("channel_insights_error", None)
                    st.rerun()
    with action_cols[2]:
        st.link_button("Open Channel", channel["canonical_url"], use_container_width=True)

    st.markdown(
        f"""
        <div class="ci-card" style="margin-top:0.75rem;">
            <div class="ci-card-title">Experimental Topic Model</div>
            <div class="ci-summary-grid">
                <div class="ci-summary-item">
                    <div class="ci-summary-label">Requested Mode</div>
                    <div class="ci-summary-value">{escape(_topic_mode_label(topic_mode_requested))}</div>
                </div>
                <div class="ci-summary-item">
                    <div class="ci-summary-label">Applied Mode</div>
                    <div class="ci-summary-value">{escape(_topic_mode_label(topic_mode_used))}</div>
                </div>
                <div class="ci-summary-item">
                    <div class="ci-summary-label">Artifact Status</div>
                    <div class="ci-summary-value">{escape(_artifact_status_label(getattr(artifact_status, 'state', 'disabled')))}</div>
                </div>
                <div class="ci-summary-item">
                    <div class="ci-summary-label">Bundle Version</div>
                    <div class="ci-summary-value">{escape(str(summary.get('topic_model_bundle_version', '') or 'Not Loaded'))}</div>
                </div>
            </div>
            <div class="ci-note" style="margin-top:0.85rem;">{escape(topic_model_message or 'The heuristic topic flow remains the fallback when BERTopic is unavailable.')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if topic_mode_requested == TOPIC_MODE_BERTOPIC_OPTIONAL and topic_mode_used != TOPIC_MODE_BERTOPIC_OPTIONAL:
        st.warning(topic_model_failure_reason or "BERTopic beta mode could not run for this snapshot, so the page fell back to heuristic topics.")

    deltas = payload.get("history_delta", {})
    kpi_row(
        [
            {
                "label": "Upload Cadence",
                "value": f"{summary.get('avg_upload_gap_days', 0):.1f} Days",
                "delta": _history_delta_text(deltas.get("upload_gap_delta", 0), "d") if deltas else None,
            },
            {
                "label": "Recent Outliers",
                "value": _format_int(summary.get("recent_outlier_count", 0)),
                "delta": _history_delta_text(deltas.get("outlier_count_delta", 0)) if deltas else None,
            },
            {
                "label": "Strongest Theme",
                "value": summary.get("strongest_theme", "N/A"),
            },
            {
                "label": "Weakest Theme",
                "value": summary.get("weakest_theme", "N/A"),
            },
            {
                "label": "Median Views / Day",
                "value": _format_int(summary.get("median_views_per_day", 0)),
                "delta": _history_delta_text(deltas.get("median_views_per_day_delta", 0)) if deltas else None,
            },
        ]
    )

    if owner_metrics_active:
        owner_kpis = [
            {
                "label": "Thumbnail Impressions",
                "value": _format_int(summary.get("owner_thumbnail_impressions", 0)),
            },
            {
                "label": "Thumbnail CTR",
                "value": _format_ratio_pct(summary.get("owner_thumbnail_ctr", 0)),
            },
            {
                "label": "Watch Time",
                "value": _format_hours(summary.get("owner_watch_hours", 0)),
            },
            {
                "label": "Average Viewed",
                "value": _format_ratio_pct(summary.get("owner_average_view_percentage", 0)),
            },
            {
                "label": "Avg View Duration",
                "value": _format_duration_seconds(summary.get("owner_average_view_duration_seconds", 0)),
            },
        ]
        kpi_row(owner_kpis)
    if summary.get("owner_note"):
        st.caption(summary["owner_note"])


def _render_overview_tab(payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    recommendations = payload.get("recommendations", {})
    topic_metrics_df = payload.get("topic_metrics_df", pd.DataFrame())
    duration_metrics_df = payload.get("duration_metrics_df", pd.DataFrame())
    owner_daily_metrics_df = payload.get("owner_daily_metrics_df", pd.DataFrame())

    overview_cols = st.columns([1.15, 1], gap="large")
    with overview_cols[0]:
        st.markdown(
            f"""
            <div class="ci-card">
                <div class="ci-card-title">What The Channel Is Signaling Right Now</div>
                <div class="ci-card-copy">{escape(recommendations.get('summary', 'Refresh the channel to generate grounded summary signals.'))}</div>
                <div class="ci-summary-grid">
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Best Duration Bucket</div>
                        <div class="ci-summary-value">{escape(summary.get('best_duration_bucket', 'N/A'))}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Best Title Pattern</div>
                        <div class="ci-summary-value">{escape(summary.get('best_title_pattern', 'N/A'))}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Shorts Share</div>
                        <div class="ci-summary-value">{_format_pct(summary.get('shorts_ratio', 0))}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Median Engagement</div>
                        <div class="ci-summary-value">{_format_pct(summary.get('median_engagement', 0))}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Owner Metrics</div>
                        <div class="ci-summary-value">{'Connected' if summary.get('owner_metrics_available') else 'Public Only'}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Topic Source</div>
                        <div class="ci-summary-value">{escape(_topic_mode_label(summary.get('topic_mode_used', TOPIC_MODE_HEURISTIC)))}</div>
                    </div>
                    <div class="ci-summary-item">
                        <div class="ci-summary-label">Thumbnail CTR</div>
                        <div class="ci-summary-value">{_format_ratio_pct(summary.get('owner_thumbnail_ctr', 0)) if summary.get('owner_metrics_available') else 'Connect OAuth'}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        actions = recommendations.get("actions", [])
        st.markdown("**Recommended Next Actions**")
        if actions:
            st.markdown(
                "<ul class='ci-list'>" + "".join(f"<li>{escape(action)}</li>" for action in actions) + "</ul>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='ci-empty'>Refresh the channel to generate action recommendations.</div>", unsafe_allow_html=True)

    with overview_cols[1]:
        if not topic_metrics_df.empty:
            topic_fig = plotly_bar_chart(
                topic_metrics_df.head(8).sort_values("trend_score", ascending=True),
                x="topic_label",
                y="trend_score",
                title="Rising Themes In The Current Window",
                horizontal=True,
            )
            st.plotly_chart(topic_fig, use_container_width=True)

        if not duration_metrics_df.empty:
            duration_fig = plotly_bar_chart(
                duration_metrics_df.sort_values("median_views_per_day", ascending=True),
                x="duration_bucket",
                y="median_views_per_day",
                title="Winning Duration Buckets",
                horizontal=True,
            )
            st.plotly_chart(duration_fig, use_container_width=True)

        if not owner_daily_metrics_df.empty:
            owner_chart = owner_daily_metrics_df.copy()
            owner_chart["day"] = pd.to_datetime(owner_chart["day"], errors="coerce")
            if "estimatedMinutesWatched" in owner_chart.columns:
                owner_chart["estimated_watch_hours"] = pd.to_numeric(owner_chart["estimatedMinutesWatched"], errors="coerce").fillna(0) / 60.0
            owner_y_cols = [
                column
                for column in ["views", "videoThumbnailImpressions", "estimated_watch_hours"]
                if column in owner_chart.columns
            ]
            if owner_y_cols:
                owner_fig = plotly_line_chart(
                    owner_chart,
                    x="day",
                    y_cols=owner_y_cols,
                    title="Owner Metrics Trendline",
                )
                st.plotly_chart(owner_fig, use_container_width=True)

        if summary.get("owner_note"):
            st.caption(summary["owner_note"])
        if summary.get("topic_mode_requested") == TOPIC_MODE_BERTOPIC_OPTIONAL and summary.get("topic_mode_used") != TOPIC_MODE_BERTOPIC_OPTIONAL:
            st.caption(summary.get("topic_model_failure_reason") or "BERTopic beta mode fell back to the heuristic topic flow.")


def _render_topic_trends_tab(payload: Dict[str, Any]) -> None:
    topic_metrics_df = payload.get("topic_metrics_df", pd.DataFrame())
    if topic_metrics_df.empty:
        st.markdown("<div class='ci-empty'>This channel needs more public uploads before theme clustering becomes stable.</div>", unsafe_allow_html=True)
        return

    styled_dataframe(
        topic_metrics_df[
            ["topic_label", "video_count", "median_views_per_day", "outlier_count", "trend_score", "avg_engagement"]
        ],
        title="Topic Cluster Performance",
        precision=2,
    )

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        trend_fig = plotly_bar_chart(
            topic_metrics_df.head(10).sort_values("trend_score", ascending=True),
            x="topic_label",
            y="trend_score",
            title="Theme Momentum",
            horizontal=True,
        )
        st.plotly_chart(trend_fig, use_container_width=True)
    with chart_cols[1]:
        views_fig = plotly_bar_chart(
            topic_metrics_df.head(10).sort_values("median_views_per_day", ascending=True),
            x="topic_label",
            y="median_views_per_day",
            title="Median Views / Day By Theme",
            horizontal=True,
        )
        st.plotly_chart(views_fig, use_container_width=True)

    top_topics = topic_metrics_df["topic_label"].head(10).tolist()
    if top_topics:
        st.markdown("**Theme Vocabulary**")
        styled_keyword_chips(top_topics)


def _render_formats_tab(payload: Dict[str, Any]) -> None:
    duration_metrics_df = payload.get("duration_metrics_df", pd.DataFrame())
    title_pattern_metrics_df = payload.get("title_pattern_metrics_df", pd.DataFrame())
    publish_day_metrics_df = payload.get("publish_day_metrics_df", pd.DataFrame())
    publish_hour_metrics_df = payload.get("publish_hour_metrics_df", pd.DataFrame())

    top_cols = st.columns(2, gap="large")
    with top_cols[0]:
        if not duration_metrics_df.empty:
            styled_dataframe(duration_metrics_df, title="Duration Performance", precision=2)
    with top_cols[1]:
        if not title_pattern_metrics_df.empty:
            styled_dataframe(title_pattern_metrics_df, title="Title Pattern Performance", precision=2)

    bottom_cols = st.columns(2, gap="large")
    with bottom_cols[0]:
        if not publish_day_metrics_df.empty:
            day_fig = plotly_bar_chart(
                publish_day_metrics_df.sort_values("median_views_per_day", ascending=True),
                x="publish_day",
                y="median_views_per_day",
                title="Best Publish Days",
                horizontal=True,
            )
            st.plotly_chart(day_fig, use_container_width=True)
    with bottom_cols[1]:
        if not publish_hour_metrics_df.empty:
            hour_fig = plotly_line_chart(
                publish_hour_metrics_df,
                x="publish_hour",
                y_cols=["median_views_per_day", "videos"],
                title="Publish Hour Signal",
                secondary_y=["videos"],
            )
            st.plotly_chart(hour_fig, use_container_width=True)


def _render_outliers_tab(payload: Dict[str, Any]) -> None:
    outliers_df = payload.get("outliers_df", pd.DataFrame())
    underperformers_df = payload.get("underperformers_df", pd.DataFrame())
    owner_metrics_active = bool(payload.get("summary", {}).get("owner_metrics_available"))
    outlier_cols = st.columns(2, gap="large")
    with outlier_cols[0]:
        st.markdown("**Recent Outliers**")
        if outliers_df.empty:
            st.markdown("<div class='ci-empty'>No strong outliers have been detected in the current window yet.</div>", unsafe_allow_html=True)
        else:
            outlier_columns = ["video_title", "primary_topic", "views", "views_per_day", "performance_score", "why_it_worked"]
            if owner_metrics_active:
                for extra in ["owner_video_thumbnail_impressions_click_rate", "owner_average_view_percentage"]:
                    if extra in outliers_df.columns:
                        outlier_columns.insert(-1, extra)
            styled_dataframe(
                outliers_df[outlier_columns],
                title=None,
                precision=2,
            )
    with outlier_cols[1]:
        st.markdown("**Underperformers**")
        if underperformers_df.empty:
            st.markdown("<div class='ci-empty'>No low-signal videos are available for comparison yet.</div>", unsafe_allow_html=True)
        else:
            underperformer_columns = ["video_title", "primary_topic", "views", "views_per_day", "performance_score", "why_it_lagged"]
            if owner_metrics_active:
                for extra in ["owner_video_thumbnail_impressions_click_rate", "owner_average_view_percentage"]:
                    if extra in underperformers_df.columns:
                        underperformer_columns.insert(-1, extra)
            styled_dataframe(
                underperformers_df[underperformer_columns],
                title=None,
                precision=2,
            )


def _render_theme_cards(title: str, rows: Iterable[Dict[str, Any]], channel_title: str) -> None:
    st.markdown(f"**{title}**")
    items = list(rows)
    if not items:
        st.markdown("<div class='ci-empty'>No theme recommendations are available yet.</div>", unsafe_allow_html=True)
        return

    for item in items:
        st.markdown(
            f"""
            <div class="ci-theme-card">
                <div class="ci-theme-title">{escape(str(item.get('theme', 'Theme')))}</div>
                <div class="ci-theme-copy">{escape(str(item.get('rationale', '')))}</div>
                <div class="ci-theme-copy" style="margin-top:0.35rem;color:#F7F8FC;">{escape(str(item.get('action', '')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        theme = str(item.get("theme", "")).strip()
        if theme:
            if st.button(f"Send '{theme}' To Outlier Finder", key=f"channel_insights_theme_{title}_{theme}", use_container_width=True):
                _queue_outlier_finder_theme(theme, channel_title)
                st.rerun()


def _render_next_topics_tab(payload: Dict[str, Any]) -> None:
    recommendations = payload.get("recommendations", {})
    channel_title = payload["channel"]["channel_title"]
    columns = st.columns(3, gap="large")
    with columns[0]:
        _render_theme_cards("Double Down", recommendations.get("double_down", []), channel_title)
    with columns[1]:
        _render_theme_cards("Avoid Or Repackage", recommendations.get("avoid", []), channel_title)
    with columns[2]:
        _render_theme_cards("Test Next", recommendations.get("test_next", []), channel_title)

    st.markdown("**Suggested Video Directions**")
    ideas = recommendations.get("video_ideas", [])
    if ideas:
        for item in ideas:
            st.markdown(
                f"""
                <div class="ci-theme-card">
                    <div class="ci-theme-title">{escape(str(item.get('title', 'Idea')))}</div>
                    <div class="ci-theme-copy">{escape(str(item.get('why_now', '')))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='ci-empty'>No grounded ideas are available yet.</div>", unsafe_allow_html=True)

    ai_overlay = recommendations.get("ai_overlay", "")
    if ai_overlay:
        st.markdown("**AI Explanation Layer**")
        st.markdown(ai_overlay)


def _render_history_tab(payload: Dict[str, Any]) -> None:
    history_df = payload.get("history_df", pd.DataFrame())
    if history_df.empty:
        st.markdown("<div class='ci-empty'>Refresh the connected channel again later to unlock historical comparison.</div>", unsafe_allow_html=True)
        return

    styled_dataframe(history_df, title="Snapshot History", precision=2)
    if len(history_df) > 1:
        history_line = history_df.sort_values("snapshot_at").copy()
        history_line["snapshot_at"] = pd.to_datetime(history_line["snapshot_at"], errors="coerce")
        fig = plotly_line_chart(
            history_line,
            x="snapshot_at",
            y_cols=["median_views_per_day", "recent_outlier_count"],
            title="Snapshot Trendline",
            secondary_y=["recent_outlier_count"],
        )
        st.plotly_chart(fig, use_container_width=True)
        if history_line.get("owner_metrics_available") is not None and history_line["owner_metrics_available"].fillna(False).any():
            owner_columns = [
                column
                for column in ["owner_watch_hours", "owner_average_view_percentage", "owner_thumbnail_ctr"]
                if column in history_line.columns
            ]
            if owner_columns:
                owner_fig = plotly_line_chart(
                    history_line,
                    x="snapshot_at",
                    y_cols=owner_columns,
                    title="Owner Metrics Across Snapshots",
                )
                st.plotly_chart(owner_fig, use_container_width=True)


def render() -> None:
    _handle_oauth_callback()
    _inject_channel_insights_css()
    _render_hero()

    connected_channels = list_connected_channels()
    _render_connect_card(connected_channels)

    error_message = st.session_state.get("channel_insights_error")
    if error_message:
        st.error(error_message)

    selected_channel_id = st.session_state.get("channel_insights_selected_channel")
    if not selected_channel_id:
        st.markdown(
            "<div class='ci-empty'>Add a public channel above to build your first persisted insight snapshot. This workflow is designed for recurring manual refreshes now, with true scheduled daily monitoring reserved for a later phase.</div>",
            unsafe_allow_html=True,
        )
        return

    payload = load_channel_insights(selected_channel_id)
    if not payload:
        st.markdown(
            "<div class='ci-empty'>This tracked channel does not have a usable stored snapshot yet. Refresh the analysis to generate one.</div>",
            unsafe_allow_html=True,
        )
        return

    _render_summary_action_row(payload)

    tabs = st.tabs(["Overview", "Topic Trends", "Formats & Patterns", "Outliers", "Next Topics", "History"])
    with tabs[0]:
        _render_overview_tab(payload)
    with tabs[1]:
        _render_topic_trends_tab(payload)
    with tabs[2]:
        _render_formats_tab(payload)
    with tabs[3]:
        _render_outliers_tab(payload)
    with tabs[4]:
        _render_next_topics_tab(payload)
    with tabs[5]:
        _render_history_tab(payload)
