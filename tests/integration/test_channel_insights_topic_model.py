from pathlib import Path

import pandas as pd

from src.services.channel_insights_service import TOPIC_MODE_HEURISTIC, refresh_channel_insights
from src.services.public_channel_service import PublicChannelWorkspace


def test_default_channel_insights_refresh_does_not_touch_optional_topic_model(monkeypatch, tmp_path: Path) -> None:
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
        lambda df: (_ for _ in ()).throw(AssertionError("Optional topic model should not run in heuristic mode")),
    )

    db_path = tmp_path / "channel_insights.db"
    payload = refresh_channel_insights("@demo", db_path=db_path, topic_mode=TOPIC_MODE_HEURISTIC)

    assert payload["summary"]["topic_mode_used"] == TOPIC_MODE_HEURISTIC
