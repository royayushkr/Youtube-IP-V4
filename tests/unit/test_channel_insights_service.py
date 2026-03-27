from pathlib import Path

import pandas as pd

from src.services.channel_insights_service import (
    TOPIC_MODE_BERTOPIC_OPTIONAL,
    TOPIC_MODE_HEURISTIC,
    load_channel_insights,
    refresh_channel_insights,
)
from src.services.public_channel_service import PublicChannelWorkspace
from src.services.topic_model_runtime import TopicModelInferenceResult
from src.services.youtube_owner_analytics_service import OwnerAnalyticsBundle


def test_refresh_channel_insights_persists_snapshot(monkeypatch, tmp_path: Path) -> None:
    sample_df = pd.DataFrame(
        [
            {
                "channel_id": "UC123",
                "channel_title": "Demo Channel",
                "video_id": "v1",
                "video_title": "How To Build Better Hooks",
                "video_description": "hooks for creators",
                "video_tags": "hooks|creator|youtube",
                "video_publishedAt": "2026-03-10T12:00:00Z",
                "views": 10000,
                "likes": 600,
                "comments": 45,
                "duration": "PT8M0S",
                "duration_seconds": 480,
            },
            {
                "channel_id": "UC123",
                "channel_title": "Demo Channel",
                "video_id": "v2",
                "video_title": "Shorts Strategy That Actually Converts",
                "video_description": "shorts strategy and topic testing",
                "video_tags": "shorts|strategy|topic",
                "video_publishedAt": "2026-03-01T12:00:00Z",
                "views": 6000,
                "likes": 340,
                "comments": 20,
                "duration": "PT45S",
                "duration_seconds": 45,
            },
        ]
    )

    workspace = PublicChannelWorkspace(
        channel_df=sample_df,
        source="youtube_api",
        channel_id="UC123",
        channel_title="Demo Channel",
        canonical_url="https://www.youtube.com/channel/UC123",
        query_used="@demo",
    )

    monkeypatch.setattr(
        "src.services.channel_insights_service.load_public_channel_workspace",
        lambda channel_input, force_refresh=False: workspace,
    )
    monkeypatch.setattr(
        "src.services.channel_insights_service.maybe_generate_ai_overlay",
        lambda *args, **kwargs: "",
    )

    db_path = tmp_path / "channel_insights.db"
    payload = refresh_channel_insights("@demo", db_path=db_path)
    loaded = load_channel_insights("UC123", db_path=db_path)

    assert payload["channel"]["channel_id"] == "UC123"
    assert loaded is not None
    assert loaded["summary"]["strongest_theme"]
    assert not loaded["outliers_df"].empty
    assert loaded["summary"]["topic_mode_used"] == TOPIC_MODE_HEURISTIC


def test_refresh_channel_insights_includes_owner_metrics(monkeypatch, tmp_path: Path) -> None:
    sample_df = pd.DataFrame(
        [
            {
                "channel_id": "UC123",
                "channel_title": "Demo Channel",
                "video_id": "v1",
                "video_title": "How To Build Better Hooks",
                "video_description": "hooks for creators",
                "video_tags": "hooks|creator|youtube",
                "video_publishedAt": "2026-03-10T12:00:00Z",
                "views": 10000,
                "likes": 600,
                "comments": 45,
                "duration": "PT8M0S",
                "duration_seconds": 480,
            }
        ]
    )

    workspace = PublicChannelWorkspace(
        channel_df=sample_df,
        source="youtube_api",
        channel_id="UC123",
        channel_title="Demo Channel",
        canonical_url="https://www.youtube.com/channel/UC123",
        query_used="@demo",
    )

    monkeypatch.setattr(
        "src.services.channel_insights_service.load_public_channel_workspace",
        lambda channel_input, force_refresh=False: workspace,
    )
    monkeypatch.setattr(
        "src.services.channel_insights_service.maybe_generate_ai_overlay",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(
        "src.services.channel_insights_service.fetch_owner_channel_analytics",
        lambda credentials, target_channel_id, video_ids: OwnerAnalyticsBundle(
            available=True,
            owned_channels=[{"channel_id": "UC123", "channel_title": "Demo Channel"}],
            summary={
                "window_days": 28,
                "start_date": "2026-02-14",
                "end_date": "2026-03-13",
                "views": 25000,
                "estimated_watch_hours": 130.5,
                "averageViewPercentage": 52.3,
                "average_view_percentage": 52.3,
                "videoThumbnailImpressions": 44000,
                "video_thumbnail_impressions": 44000,
                "videoThumbnailImpressionsClickRate": 0.061,
                "video_thumbnail_impressions_click_rate": 0.061,
                "averageViewDuration": 210,
                "average_view_duration_seconds": 210,
                "subscribers_gained": 120,
                "subscribers_lost": 12,
            },
            daily_metrics_df=pd.DataFrame([{"day": "2026-03-13", "views": 1200}]),
            video_metrics_df=pd.DataFrame(
                [
                    {
                        "video_id": "v1",
                        "videoThumbnailImpressionsClickRate": 0.061,
                        "averageViewPercentage": 52.3,
                    }
                ]
            ),
            available_metrics=["views", "videoThumbnailImpressionsClickRate"],
            missing_metrics=[],
            note="",
        ),
    )

    db_path = tmp_path / "channel_insights.db"
    payload = refresh_channel_insights("@demo", db_path=db_path, owner_credentials={"token": "demo"})

    assert payload["summary"]["owner_metrics_available"] is True
    assert payload["summary"]["owner_watch_hours"] == 130.5
    assert not payload["owner_daily_metrics_df"].empty


def test_refresh_channel_insights_falls_back_when_optional_topic_model_fails(monkeypatch, tmp_path: Path) -> None:
    sample_df = pd.DataFrame(
        [
            {
                "channel_id": "UC123",
                "channel_title": "Demo Channel",
                "video_id": "v1",
                "video_title": "How To Build Better Hooks",
                "video_description": "hooks for creators",
                "video_tags": "hooks|creator|youtube",
                "video_publishedAt": "2026-03-10T12:00:00Z",
                "views": 10000,
                "likes": 600,
                "comments": 45,
                "duration": "PT8M0S",
                "duration_seconds": 480,
            }
        ]
    )

    workspace = PublicChannelWorkspace(
        channel_df=sample_df,
        source="youtube_api",
        channel_id="UC123",
        channel_title="Demo Channel",
        canonical_url="https://www.youtube.com/channel/UC123",
        query_used="@demo",
    )

    monkeypatch.setattr(
        "src.services.channel_insights_service.load_public_channel_workspace",
        lambda channel_input, force_refresh=False: workspace,
    )
    monkeypatch.setattr(
        "src.services.channel_insights_service.maybe_generate_ai_overlay",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(
        "src.services.channel_insights_service.apply_optional_topic_model",
        lambda df: TopicModelInferenceResult(
            success=False,
            topic_rows=(),
            bundle_version="2026.03.27",
            failure_reason="Bundle missing",
            status="download_required",
        ),
    )

    db_path = tmp_path / "channel_insights.db"
    payload = refresh_channel_insights("@demo", db_path=db_path, topic_mode=TOPIC_MODE_BERTOPIC_OPTIONAL)

    assert payload["summary"]["topic_mode_requested"] == TOPIC_MODE_BERTOPIC_OPTIONAL
    assert payload["summary"]["topic_mode_used"] == TOPIC_MODE_HEURISTIC
    assert payload["summary"]["topic_model_failure_reason"] == "Bundle missing"


def test_refresh_channel_insights_uses_optional_topic_model_when_available(monkeypatch, tmp_path: Path) -> None:
    sample_df = pd.DataFrame(
        [
            {
                "channel_id": "UC123",
                "channel_title": "Demo Channel",
                "video_id": "v1",
                "video_title": "How To Improve Packaging",
                "video_description": "packaging breakdown",
                "video_tags": "packaging|ctr",
                "video_publishedAt": "2026-03-10T12:00:00Z",
                "views": 10000,
                "likes": 600,
                "comments": 45,
                "duration": "PT8M0S",
                "duration_seconds": 480,
            }
        ]
    )

    workspace = PublicChannelWorkspace(
        channel_df=sample_df,
        source="youtube_api",
        channel_id="UC123",
        channel_title="Demo Channel",
        canonical_url="https://www.youtube.com/channel/UC123",
        query_used="@demo",
    )

    monkeypatch.setattr(
        "src.services.channel_insights_service.load_public_channel_workspace",
        lambda channel_input, force_refresh=False: workspace,
    )
    monkeypatch.setattr(
        "src.services.channel_insights_service.maybe_generate_ai_overlay",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(
        "src.services.channel_insights_service.apply_optional_topic_model",
        lambda df: TopicModelInferenceResult(
            success=True,
            topic_rows=(
                {
                    "video_id": "v1",
                    "model_topic_id": 18,
                    "model_topic_label_raw": "18_packaging_ctr_thumbnail_hooks",
                    "model_topic_label": "Packaging / Ctr / Thumbnail / Hooks",
                    "topic_source": "bertopic_global",
                    "bertopic_text": "packaging ctr thumbnail hooks",
                    "bertopic_token_count": 6,
                    "is_sparse_text": False,
                },
            ),
            bundle_version="2026.03.27",
            message="BERTopic topic assignments were applied successfully.",
            status="ready",
        ),
    )

    db_path = tmp_path / "channel_insights.db"
    payload = refresh_channel_insights("@demo", db_path=db_path, topic_mode=TOPIC_MODE_BERTOPIC_OPTIONAL)

    assert payload["summary"]["topic_mode_requested"] == TOPIC_MODE_BERTOPIC_OPTIONAL
    assert payload["summary"]["topic_mode_used"] == TOPIC_MODE_BERTOPIC_OPTIONAL
    assert payload["videos_df"]["topic_source"].iloc[0] == "bertopic_global"
    assert payload["videos_df"]["primary_topic"].iloc[0] == "Packaging / Ctr / Thumbnail / Hooks"
