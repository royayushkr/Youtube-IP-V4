from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.services.channel_idea_service import build_grounded_idea_bundle, maybe_generate_ai_overlay
from src.services.model_artifact_service import get_bertopic_artifact_status
from src.services.channel_snapshot_store import (
    DEFAULT_CHANNEL_INSIGHTS_DB,
    get_tracked_channel,
    list_channel_snapshot_history,
    list_tracked_channels,
    load_latest_channel_snapshot,
    store_channel_snapshot,
    upsert_tracked_channel,
)
from src.services.public_channel_service import PublicChannelWorkspace, ensure_public_channel_frame, load_public_channel_workspace
from src.services.topic_model_runtime import apply_optional_topic_model
from src.services.topic_analysis_service import (
    add_channel_video_features,
    assign_topic_labels,
    build_duration_metrics,
    build_publish_day_metrics,
    build_publish_hour_metrics,
    build_title_pattern_metrics,
    build_topic_metrics,
)
from src.services.youtube_owner_analytics_service import OwnerAnalyticsBundle, fetch_owner_channel_analytics
from src.utils.channel_parser import normalize_channel_input


TOPIC_MODE_HEURISTIC = "heuristic"
TOPIC_MODE_BERTOPIC_OPTIONAL = "bertopic_optional"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_upload_gap_days(df: pd.DataFrame) -> float:
    ordered = df.sort_values("video_publishedAt").copy()
    if len(ordered) < 2:
        return 0.0
    gaps = ordered["video_publishedAt"].diff().dt.total_seconds().dropna() / 86400
    return float(gaps.mean()) if not gaps.empty else 0.0


def _score_videos(channel_df: pd.DataFrame) -> pd.DataFrame:
    if channel_df.empty:
        return channel_df.copy()
    df = channel_df.copy()
    views_rank = df["views_per_day"].rank(method="average", pct=True).fillna(0)
    engagement_rank = df["engagement_rate"].rank(method="average", pct=True).fillna(0)
    recency_raw = 1 / df["age_days"].clip(lower=0.5)
    recency_rank = recency_raw.rank(method="average", pct=True).fillna(0)
    weighted_parts = [
        (views_rank, 55.0),
        (engagement_rank, 25.0),
        (recency_rank, 20.0),
    ]

    if "owner_video_thumbnail_impressions_click_rate" in df.columns:
        ctr_rank = pd.to_numeric(
            df["owner_video_thumbnail_impressions_click_rate"], errors="coerce"
        ).fillna(0).rank(method="average", pct=True)
        weighted_parts.append((ctr_rank, 12.0))
    if "owner_average_view_percentage" in df.columns:
        retention_rank = pd.to_numeric(
            df["owner_average_view_percentage"], errors="coerce"
        ).fillna(0).rank(method="average", pct=True)
        weighted_parts.append((retention_rank, 8.0))

    total_weight = sum(weight for _, weight in weighted_parts) or 1.0
    score_series = sum(series * weight for series, weight in weighted_parts) / total_weight * 100
    df["performance_score"] = score_series.clip(0, 100)
    return df


def _normalize_topic_mode(topic_mode: str) -> str:
    if str(topic_mode or "").strip().lower() == TOPIC_MODE_BERTOPIC_OPTIONAL:
        return TOPIC_MODE_BERTOPIC_OPTIONAL
    return TOPIC_MODE_HEURISTIC


def _apply_heuristic_topics(channel_df: pd.DataFrame) -> pd.DataFrame:
    df = assign_topic_labels(channel_df)
    df["topic_source"] = "heuristic"
    df["model_topic_id"] = pd.NA
    df["model_topic_label_raw"] = ""
    df["model_topic_label"] = ""
    return df


def _apply_requested_topic_mode(channel_df: pd.DataFrame, topic_mode: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    requested_mode = _normalize_topic_mode(topic_mode)
    if requested_mode == TOPIC_MODE_HEURISTIC:
        return _apply_heuristic_topics(channel_df), {
            "topic_mode_requested": TOPIC_MODE_HEURISTIC,
            "topic_mode_used": TOPIC_MODE_HEURISTIC,
            "model_bundle_version": "",
            "model_available": False,
            "model_failure_reason": "",
            "model_status": "disabled",
            "topic_model_message": "Using the default heuristic topic clustering flow.",
        }

    inference = apply_optional_topic_model(channel_df)
    if inference.success:
        topic_df = pd.DataFrame(inference.topic_rows)
        merged = channel_df.merge(topic_df, on="video_id", how="left")
        merged["primary_topic"] = merged["model_topic_label"].fillna("").astype(str).replace("", "Unassigned")
        merged["topic_labels"] = merged["primary_topic"].apply(lambda value: [str(value)])
        merged["topic_source"] = merged["topic_source"].fillna("bertopic_global")
        return merged, {
            "topic_mode_requested": TOPIC_MODE_BERTOPIC_OPTIONAL,
            "topic_mode_used": TOPIC_MODE_BERTOPIC_OPTIONAL,
            "model_bundle_version": inference.bundle_version,
            "model_available": True,
            "model_failure_reason": "",
            "model_status": inference.status,
            "topic_model_message": inference.message or "Using the optional BERTopic model.",
        }

    fallback_df = _apply_heuristic_topics(channel_df)
    return fallback_df, {
        "topic_mode_requested": TOPIC_MODE_BERTOPIC_OPTIONAL,
        "topic_mode_used": TOPIC_MODE_HEURISTIC,
        "model_bundle_version": inference.bundle_version,
        "model_available": False,
        "model_failure_reason": inference.failure_reason,
        "model_status": inference.status,
        "topic_model_message": "BERTopic beta mode was requested, but the page fell back to heuristic topics.",
    }


def _merge_owner_video_metrics(channel_df: pd.DataFrame, owner_bundle: OwnerAnalyticsBundle) -> pd.DataFrame:
    if channel_df.empty or not owner_bundle.available or owner_bundle.video_metrics_df.empty:
        return channel_df

    owner_df = owner_bundle.video_metrics_df.copy()
    rename_map = {
        "views": "owner_window_views",
        "likes": "owner_window_likes",
        "comments": "owner_window_comments",
        "estimatedMinutesWatched": "owner_estimated_minutes_watched",
        "averageViewDuration": "owner_average_view_duration_seconds",
        "averageViewPercentage": "owner_average_view_percentage",
        "videoThumbnailImpressions": "owner_video_thumbnail_impressions",
        "videoThumbnailImpressionsClickRate": "owner_video_thumbnail_impressions_click_rate",
    }
    owner_df = owner_df.rename(columns=rename_map)
    merge_columns = [column for column in ["video_id", *rename_map.values()] if column in owner_df.columns]
    if "video_id" not in merge_columns:
        return channel_df
    deduped = owner_df[merge_columns].drop_duplicates(subset=["video_id"])
    return channel_df.merge(deduped, on="video_id", how="left")


def _outlier_and_underperformer_tables(channel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if channel_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ranked = channel_df.sort_values(["performance_score", "views_per_day"], ascending=[False, False]).copy()
    outliers = ranked[ranked["performance_score"] >= 75].head(12).copy()
    underperformers = ranked.sort_values(["performance_score", "views_per_day"], ascending=[True, True]).head(12).copy()

    outliers["why_it_worked"] = outliers.apply(
        lambda row: f"{row['primary_topic'].replace('_', ' ').title()} is outperforming the channel median with {row['views_per_day']:.0f} views/day.",
        axis=1,
    )
    underperformers["why_it_lagged"] = underperformers.apply(
        lambda row: f"{row['primary_topic'].replace('_', ' ').title()} is lagging the channel baseline and the packaging may need a new angle.",
        axis=1,
    )
    return outliers, underperformers


def _format_metrics(topic_metrics: pd.DataFrame, duration_metrics: pd.DataFrame, title_pattern_metrics: pd.DataFrame) -> Dict[str, Any]:
    strongest_theme = topic_metrics.iloc[0]["topic_label"] if not topic_metrics.empty else "No Theme Yet"
    weakest_theme = topic_metrics.sort_values("median_views_per_day", ascending=True).iloc[0]["topic_label"] if not topic_metrics.empty else "No Theme Yet"
    best_duration = duration_metrics.iloc[0]["duration_bucket"] if not duration_metrics.empty else "No Pattern Yet"
    best_title_pattern = title_pattern_metrics.iloc[0]["title_pattern"] if not title_pattern_metrics.empty else "No Pattern Yet"
    return {
        "strongest_theme": strongest_theme,
        "weakest_theme": weakest_theme,
        "best_duration_bucket": best_duration,
        "best_title_pattern": best_title_pattern,
    }


def _recommended_actions(
    channel_df: pd.DataFrame,
    topic_metrics: pd.DataFrame,
    duration_metrics: pd.DataFrame,
    outliers: pd.DataFrame,
) -> List[str]:
    actions: List[str] = []
    upload_gap = _compute_upload_gap_days(channel_df)

    if not topic_metrics.empty:
        strongest = topic_metrics.iloc[0]["topic_label"]
        actions.append(f"Double Down On {strongest} because it currently leads your channel on median views per day.")
        weakest = topic_metrics.sort_values("median_views_per_day", ascending=True).iloc[0]["topic_label"]
        if weakest != strongest:
            actions.append(f"Reduce volume on {weakest} unless you can repackage it with a stronger promise or format.")

    if not duration_metrics.empty:
        best_duration = duration_metrics.iloc[0]["duration_bucket"]
        actions.append(f"Test more uploads in the {best_duration} bucket because it is your current strongest duration pattern.")

    if upload_gap > 0:
        actions.append(f"Your average upload gap is {upload_gap:.1f} days. Tighten the cadence if you want the trend signals to compound faster.")

    if len(outliers) > 0:
        actions.append("Study the top outlier titles and reuse the same promise structure before trying a completely new topic.")

    unique_actions = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)
    return unique_actions[:4]


def _history_delta(history_df: pd.DataFrame) -> Dict[str, float]:
    if history_df.empty or len(history_df) < 2:
        return {}
    latest = history_df.iloc[0]
    previous = history_df.iloc[1]
    return {
        "median_views_per_day_delta": float(latest.get("median_views_per_day", 0) - previous.get("median_views_per_day", 0)),
        "outlier_count_delta": float(latest.get("recent_outlier_count", 0) - previous.get("recent_outlier_count", 0)),
        "upload_gap_delta": float(latest.get("upload_gap_days", 0) - previous.get("upload_gap_days", 0)),
    }


def _build_summary(
    workspace: PublicChannelWorkspace,
    channel_df: pd.DataFrame,
    topic_metrics: pd.DataFrame,
    duration_metrics: pd.DataFrame,
    title_pattern_metrics: pd.DataFrame,
    outliers: pd.DataFrame,
    topic_mode_metadata: Dict[str, Any],
    owner_bundle: Optional[OwnerAnalyticsBundle] = None,
) -> Dict[str, Any]:
    metrics = _format_metrics(topic_metrics, duration_metrics, title_pattern_metrics)
    summary = {
        "channel_id": workspace.channel_id,
        "channel_title": workspace.channel_title,
        "canonical_url": workspace.canonical_url,
        "snapshot_at": _iso_now(),
        "video_count": int(len(channel_df)),
        "median_views_per_day": float(channel_df["views_per_day"].median()) if not channel_df.empty else 0.0,
        "median_engagement": float(channel_df["engagement_rate"].median()) if not channel_df.empty else 0.0,
        "avg_upload_gap_days": _compute_upload_gap_days(channel_df),
        "shorts_ratio": float(channel_df["is_short"].mean()) if not channel_df.empty else 0.0,
        "recent_outlier_count": int(len(outliers)),
        **metrics,
        "topic_mode_requested": topic_mode_metadata.get("topic_mode_requested", TOPIC_MODE_HEURISTIC),
        "topic_mode_used": topic_mode_metadata.get("topic_mode_used", TOPIC_MODE_HEURISTIC),
        "topic_model_status": topic_mode_metadata.get("model_status", "disabled"),
        "topic_model_available": bool(topic_mode_metadata.get("model_available", False)),
        "topic_model_bundle_version": topic_mode_metadata.get("model_bundle_version", ""),
        "topic_model_failure_reason": topic_mode_metadata.get("model_failure_reason", ""),
        "topic_model_message": topic_mode_metadata.get("topic_model_message", ""),
    }
    if owner_bundle and owner_bundle.available:
        owner_summary = owner_bundle.summary
        summary.update(
            {
                "owner_metrics_available": True,
                "owner_window_days": owner_summary.get("window_days", 0),
                "owner_start_date": owner_summary.get("start_date", ""),
                "owner_end_date": owner_summary.get("end_date", ""),
                "owner_window_views": float(owner_summary.get("views", 0) or 0),
                "owner_watch_hours": float(owner_summary.get("estimated_watch_hours", 0) or 0),
                "owner_average_view_duration_seconds": float(owner_summary.get("average_view_duration_seconds", 0) or 0),
                "owner_average_view_percentage": float(owner_summary.get("average_view_percentage", 0) or 0),
                "owner_thumbnail_impressions": float(owner_summary.get("video_thumbnail_impressions", 0) or 0),
                "owner_thumbnail_ctr": float(owner_summary.get("video_thumbnail_impressions_click_rate", 0) or 0),
                "owner_subscribers_gained": float(owner_summary.get("subscribers_gained", 0) or 0),
                "owner_subscribers_lost": float(owner_summary.get("subscribers_lost", 0) or 0),
                "owner_missing_metrics": owner_bundle.missing_metrics,
                "owner_note": owner_bundle.note,
            }
        )
    else:
        summary["owner_metrics_available"] = False
        summary["owner_note"] = owner_bundle.note if owner_bundle else ""
    return summary


def _insight_payload(
    *,
    channel_df: pd.DataFrame,
    topic_metrics: pd.DataFrame,
    duration_metrics: pd.DataFrame,
    title_pattern_metrics: pd.DataFrame,
    publish_day_metrics: pd.DataFrame,
    publish_hour_metrics: pd.DataFrame,
    outliers: pd.DataFrame,
    underperformers: pd.DataFrame,
    summary: Dict[str, Any],
    recommendations: Dict[str, Any],
    topic_mode_metadata: Dict[str, Any],
    owner_bundle: Optional[OwnerAnalyticsBundle] = None,
) -> Dict[str, Any]:
    return {
        "summary": summary,
        "topic_mode_metadata": topic_mode_metadata,
        "topic_metrics": topic_metrics.to_dict(orient="records"),
        "duration_metrics": duration_metrics.to_dict(orient="records"),
        "title_pattern_metrics": title_pattern_metrics.to_dict(orient="records"),
        "publish_day_metrics": publish_day_metrics.to_dict(orient="records"),
        "publish_hour_metrics": publish_hour_metrics.to_dict(orient="records"),
        "outliers": outliers.to_dict(orient="records"),
        "underperformers": underperformers.to_dict(orient="records"),
        "recommendations": recommendations,
        "owner_daily_metrics": owner_bundle.daily_metrics_df.to_dict(orient="records") if owner_bundle and owner_bundle.available else [],
        "owner_video_metrics": owner_bundle.video_metrics_df.to_dict(orient="records") if owner_bundle and owner_bundle.available else [],
        "owner_available_metrics": owner_bundle.available_metrics if owner_bundle and owner_bundle.available else [],
        "owner_missing_metrics": owner_bundle.missing_metrics if owner_bundle and owner_bundle.available else [],
        "owner_note": owner_bundle.note if owner_bundle else "",
    }


def refresh_channel_insights(
    channel_input: str,
    *,
    force_refresh: bool = False,
    topic_mode: str = TOPIC_MODE_HEURISTIC,
    db_path: Path = DEFAULT_CHANNEL_INSIGHTS_DB,
    owner_credentials: Any = None,
) -> Dict[str, Any]:
    parsed_input = normalize_channel_input(channel_input)
    workspace = load_public_channel_workspace(parsed_input.lookup_value, force_refresh=force_refresh)
    channel_df = ensure_public_channel_frame(workspace.channel_df)
    channel_df = add_channel_video_features(channel_df)
    channel_df, topic_mode_metadata = _apply_requested_topic_mode(channel_df, topic_mode)

    owner_bundle: Optional[OwnerAnalyticsBundle] = None
    if owner_credentials is not None:
        try:
            owner_bundle = fetch_owner_channel_analytics(
                owner_credentials,
                target_channel_id=workspace.channel_id,
                video_ids=channel_df["video_id"].astype(str).tolist(),
            )
        except Exception:
            owner_bundle = None
        else:
            channel_df = _merge_owner_video_metrics(channel_df, owner_bundle)

    channel_df = _score_videos(channel_df)

    topic_metrics = build_topic_metrics(channel_df)
    duration_metrics = build_duration_metrics(channel_df)
    title_pattern_metrics = build_title_pattern_metrics(channel_df)
    publish_day_metrics = build_publish_day_metrics(channel_df)
    publish_hour_metrics = build_publish_hour_metrics(channel_df)
    outliers, underperformers = _outlier_and_underperformer_tables(channel_df)

    summary = _build_summary(
        workspace=workspace,
        channel_df=channel_df,
        topic_metrics=topic_metrics,
        duration_metrics=duration_metrics,
        title_pattern_metrics=title_pattern_metrics,
        outliers=outliers,
        topic_mode_metadata=topic_mode_metadata,
        owner_bundle=owner_bundle,
    )

    idea_bundle = build_grounded_idea_bundle(
        workspace.channel_title,
        topic_metrics.to_dict(orient="records"),
        outliers.to_dict(orient="records"),
        underperformers.to_dict(orient="records"),
    )
    recommendations = {
        "summary": idea_bundle.summary,
        "double_down": [item.__dict__ for item in idea_bundle.double_down],
        "avoid": [item.__dict__ for item in idea_bundle.avoid],
        "test_next": [item.__dict__ for item in idea_bundle.test_next],
        "video_ideas": [item.__dict__ for item in idea_bundle.video_ideas],
    }
    recommendations["actions"] = _recommended_actions(channel_df, topic_metrics, duration_metrics, outliers)
    try:
        recommendations["ai_overlay"] = maybe_generate_ai_overlay(
            workspace.channel_title,
            summary,
            topic_metrics.to_dict(orient="records"),
        )
    except Exception:
        recommendations["ai_overlay"] = ""

    snapshot_at = summary["snapshot_at"]
    channel_handle = parsed_input.handle or (parsed_input.lookup_value if parsed_input.lookup_value.startswith("@") else "")
    upsert_tracked_channel(
        channel_id=workspace.channel_id,
        input_value=parsed_input.lookup_value,
        canonical_url=workspace.canonical_url,
        channel_title=workspace.channel_title,
        channel_handle=channel_handle,
        source=workspace.source,
        added_at=snapshot_at,
        last_refresh_at=snapshot_at,
        db_path=db_path,
    )
    store_channel_snapshot(
        channel_id=workspace.channel_id,
        snapshot_at=snapshot_at,
        source=workspace.source,
        summary=summary,
        videos_df=channel_df,
        topic_metrics_df=topic_metrics,
        insights_payload=_insight_payload(
            channel_df=channel_df,
            topic_metrics=topic_metrics,
            duration_metrics=duration_metrics,
            title_pattern_metrics=title_pattern_metrics,
            publish_day_metrics=publish_day_metrics,
            publish_hour_metrics=publish_hour_metrics,
            outliers=outliers,
            underperformers=underperformers,
            summary=summary,
            recommendations=recommendations,
            topic_mode_metadata=topic_mode_metadata,
            owner_bundle=owner_bundle,
        ),
        db_path=db_path,
    )
    return load_channel_insights(workspace.channel_id, db_path=db_path) or {}


def list_connected_channels(db_path: Path = DEFAULT_CHANNEL_INSIGHTS_DB) -> List[Dict[str, Any]]:
    return list_tracked_channels(db_path=db_path)


def load_channel_insights(channel_id: str, *, db_path: Path = DEFAULT_CHANNEL_INSIGHTS_DB) -> Optional[Dict[str, Any]]:
    tracked = get_tracked_channel(channel_id, db_path=db_path)
    snapshot = load_latest_channel_snapshot(channel_id, db_path=db_path)
    if not tracked or not snapshot:
        return None

    history_df = list_channel_snapshot_history(channel_id, db_path=db_path)
    insights = snapshot.get("insights", {})
    videos_df = pd.DataFrame(snapshot.get("videos", []))
    topic_metrics_df = pd.DataFrame(snapshot.get("topic_metrics", []))
    summary = snapshot.get("summary", {})
    history_delta = _history_delta(history_df)

    if videos_df.empty and "outliers" in insights:
        videos_df = pd.DataFrame(insights.get("outliers", []))

    return {
        "channel": tracked,
        "snapshot_at": snapshot["snapshot_at"],
        "source": snapshot["source"],
        "summary": summary,
        "topic_mode_metadata": insights.get("topic_mode_metadata", {}),
        "topic_artifact_status": get_bertopic_artifact_status(),
        "history_delta": history_delta,
        "videos_df": videos_df,
        "topic_metrics_df": topic_metrics_df,
        "duration_metrics_df": pd.DataFrame(insights.get("duration_metrics", [])),
        "title_pattern_metrics_df": pd.DataFrame(insights.get("title_pattern_metrics", [])),
        "publish_day_metrics_df": pd.DataFrame(insights.get("publish_day_metrics", [])),
        "publish_hour_metrics_df": pd.DataFrame(insights.get("publish_hour_metrics", [])),
        "outliers_df": pd.DataFrame(insights.get("outliers", [])),
        "underperformers_df": pd.DataFrame(insights.get("underperformers", [])),
        "owner_daily_metrics_df": pd.DataFrame(insights.get("owner_daily_metrics", [])),
        "owner_video_metrics_df": pd.DataFrame(insights.get("owner_video_metrics", [])),
        "owner_available_metrics": insights.get("owner_available_metrics", []),
        "owner_missing_metrics": insights.get("owner_missing_metrics", []),
        "owner_note": insights.get("owner_note", ""),
        "recommendations": insights.get("recommendations", {}),
        "history_df": history_df,
    }
