from datetime import datetime, timedelta, timezone

import math
import pandas as pd

from src.services.outliers_finder import (
    ChannelBaseline,
    OutlierSearchRequest,
    build_age_bucket_summary,
    build_duration_summary,
    build_title_pattern_summary,
    filter_candidates_by_subscriber_bucket,
    score_outlier_candidates_frame,
)


def _request() -> OutlierSearchRequest:
    now = datetime.now(timezone.utc)
    return OutlierSearchRequest(
        niche_query="science",
        published_after_iso=(now - timedelta(days=30)).isoformat(),
        published_before_iso=now.isoformat(),
    )


def _candidate_frame() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    published = now - timedelta(days=5)
    return pd.DataFrame(
        [
            {
                "video_id": "video-1",
                "video_title": "Physics Hack That Exploded",
                "channel_id": "channel-1",
                "channel_title": "Channel One",
                "video_url": "https://youtube.com/watch?v=video-1",
                "thumbnail_url": "https://img.youtube.com/vi/video-1/hqdefault.jpg",
                "published_at_iso": published.isoformat(),
                "age_hours": 5 * 24,
                "age_days": 5.0,
                "views": 40_000,
                "likes": 3_000,
                "comments": 300,
                "engagement_rate": 0.0825,
                "views_per_day": 8_000.0,
                "views_per_subscriber": 4.0,
                "channel_subscriber_count": 10_000,
                "hidden_subscriber_count": False,
                "channel_country": "US",
                "channel_default_language": "en",
                "video_default_language": "en",
                "size_bucket": "10K - 100K",
                "age_bucket": "3-7d",
            },
            {
                "video_id": "video-2",
                "video_title": "Regular Physics Upload",
                "channel_id": "channel-1",
                "channel_title": "Channel One",
                "video_url": "https://youtube.com/watch?v=video-2",
                "thumbnail_url": "https://img.youtube.com/vi/video-2/hqdefault.jpg",
                "published_at_iso": published.isoformat(),
                "age_hours": 5 * 24,
                "age_days": 5.0,
                "views": 4_000,
                "likes": 120,
                "comments": 12,
                "engagement_rate": 0.033,
                "views_per_day": 800.0,
                "views_per_subscriber": 0.4,
                "channel_subscriber_count": 10_000,
                "hidden_subscriber_count": False,
                "channel_country": "US",
                "channel_default_language": "en",
                "video_default_language": "en",
                "size_bucket": "10K - 100K",
                "age_bucket": "3-7d",
            },
            {
                "video_id": "video-3",
                "video_title": "Peer Video",
                "channel_id": "channel-2",
                "channel_title": "Channel Two",
                "video_url": "https://youtube.com/watch?v=video-3",
                "thumbnail_url": "https://img.youtube.com/vi/video-3/hqdefault.jpg",
                "published_at_iso": published.isoformat(),
                "age_hours": 5 * 24,
                "age_days": 5.0,
                "views": 15_000,
                "likes": 500,
                "comments": 50,
                "engagement_rate": 0.0366,
                "views_per_day": 3_000.0,
                "views_per_subscriber": 0.3,
                "channel_subscriber_count": 50_000,
                "hidden_subscriber_count": False,
                "channel_country": "US",
                "channel_default_language": "en",
                "video_default_language": "en",
                "size_bucket": "10K - 100K",
                "age_bucket": "3-7d",
            },
            {
                "video_id": "video-4",
                "video_title": "Hidden Subscriber Breakout",
                "channel_id": "channel-3",
                "channel_title": "Channel Three",
                "video_url": "https://youtube.com/watch?v=video-4",
                "thumbnail_url": "https://img.youtube.com/vi/video-4/hqdefault.jpg",
                "published_at_iso": published.isoformat(),
                "age_hours": 5 * 24,
                "age_days": 5.0,
                "views": 11_000,
                "likes": 600,
                "comments": 80,
                "engagement_rate": 0.0618,
                "views_per_day": 2_200.0,
                "views_per_subscriber": None,
                "channel_subscriber_count": None,
                "hidden_subscriber_count": True,
                "channel_country": "US",
                "channel_default_language": "en",
                "video_default_language": "en",
                "size_bucket": "Hidden",
                "age_bucket": "3-7d",
            },
        ]
    )


def test_score_outlier_candidates_ranks_true_overperformer_first() -> None:
    frame = _candidate_frame()
    baselines = {
        "channel-1": ChannelBaseline(
            channel_id="channel-1",
            channel_title="Channel One",
            sample_size=12,
            median_views=4_500.0,
            median_views_per_day=900.0,
            median_engagement_rate=0.031,
            median_views_per_subscriber=0.45,
        ),
        "channel-2": ChannelBaseline(
            channel_id="channel-2",
            channel_title="Channel Two",
            sample_size=12,
            median_views=12_000.0,
            median_views_per_day=2_400.0,
            median_engagement_rate=0.032,
            median_views_per_subscriber=0.25,
        ),
    }

    scored = score_outlier_candidates_frame(frame, _request(), baselines)

    assert scored.iloc[0]["video_id"] == "video-1"
    assert scored.iloc[0]["outlier_score"] > scored.iloc[1]["outlier_score"]
    assert scored.iloc[0]["baseline_component"] is not None


def test_score_outlier_candidates_handles_hidden_subscribers_without_nan_scores() -> None:
    frame = _candidate_frame()
    scored = score_outlier_candidates_frame(frame, _request(), baselines={})

    hidden_row = scored.loc[scored["video_id"] == "video-4"].iloc[0]

    assert math.isfinite(float(hidden_row["outlier_score"]))
    assert float(hidden_row["peer_percentile"]) >= 0.0
    assert pd.isna(hidden_row["baseline_component"])


def test_filter_candidates_by_subscriber_bucket_respects_hidden_toggle() -> None:
    frame = _candidate_frame()

    filtered = filter_candidates_by_subscriber_bucket(
        frame,
        subscriber_bucket="10K - 100K",
        include_hidden_subscribers=False,
    )

    assert set(filtered["video_id"].tolist()) == {"video-1", "video-2", "video-3"}
    assert "video-4" not in filtered["video_id"].tolist()


def test_summary_builders_keep_human_readable_ordering() -> None:
    frame = _candidate_frame().copy()
    frame["outlier_score"] = [91.0, 62.0, 74.0, 68.0]
    frame["duration_bucket"] = ["4-12 min", "Shorts (<=60s)", "12-30 min", "1-4 min"]
    frame["title_pattern"] = ["How / Why", "Numbered", "Versus", "Explainer"]

    age_summary = build_age_bucket_summary(frame)
    duration_summary = build_duration_summary(frame)
    pattern_summary = build_title_pattern_summary(frame)

    assert age_summary.iloc[0]["age_bucket"] == "3-7d"
    assert duration_summary["duration_bucket"].tolist()[:4] == [
        "Shorts (<=60s)",
        "1-4 min",
        "4-12 min",
        "12-30 min",
    ]
    assert pattern_summary["title_pattern"].tolist()[:4] == [
        "How / Why",
        "Numbered",
        "Explainer",
        "Versus",
    ]
