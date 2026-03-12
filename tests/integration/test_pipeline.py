from datetime import datetime, timedelta, timezone

from src.services import outliers_finder


def test_search_outlier_videos_respects_advanced_filters(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    request = outliers_finder.OutlierSearchRequest(
        niche_query="AI automation",
        published_after_iso=(now - timedelta(days=30)).isoformat(),
        published_before_iso=now.isoformat(),
        relevance_language="en",
        language_strictness="strict",
        min_views=10_000,
        duration_preference="4-12 min",
        freshness_days=14,
        exclude_keywords=("podcast",),
        match_mode="exact",
        max_results=50,
        baseline_channel_limit=10,
        baseline_video_cap=5,
    )

    outliers_finder._search_outlier_videos_cached.clear()
    outliers_finder._fetch_channel_baseline_cached.clear()

    def fake_run_with_provider_keys(provider, operation, retryable_error=None):
        return operation("fake-key")

    def fake_youtube_get(api_key, endpoint, params):
        if endpoint == "search":
            return {
                "items": [
                    {"id": {"videoId": "video-1"}},
                    {"id": {"videoId": "video-2"}},
                    {"id": {"videoId": "video-3"}},
                    {"id": {"videoId": "video-4"}},
                ]
            }

        if endpoint == "videos":
            requested_ids = params.get("id", "").split(",")
            if requested_ids == ["video-1", "video-2", "video-3", "video-4"]:
                return {
                    "items": [
                        {
                            "id": "video-1",
                            "snippet": {
                                "title": "AI Automation Systems",
                                "description": "A breakdown of AI automation workflows.",
                                "channelId": "channel-1",
                                "channelTitle": "Channel One",
                                "publishedAt": (now - timedelta(days=5)).isoformat(),
                                "defaultLanguage": "en",
                                "thumbnails": {"high": {"url": "https://img.youtube.com/vi/video-1/hqdefault.jpg"}},
                            },
                            "statistics": {"viewCount": "42000", "likeCount": "2400", "commentCount": "220"},
                            "contentDetails": {"duration": "PT8M30S"},
                        },
                        {
                            "id": "video-2",
                            "snippet": {
                                "title": "Automatizacion IA",
                                "description": "Contenido en espanol.",
                                "channelId": "channel-2",
                                "channelTitle": "Channel Two",
                                "publishedAt": (now - timedelta(days=5)).isoformat(),
                                "defaultLanguage": "es",
                            },
                            "statistics": {"viewCount": "80000", "likeCount": "1800", "commentCount": "150"},
                            "contentDetails": {"duration": "PT7M"},
                        },
                        {
                            "id": "video-3",
                            "snippet": {
                                "title": "AI Automation Podcast Clips",
                                "description": "Podcast discussion about AI automation.",
                                "channelId": "channel-3",
                                "channelTitle": "Channel Three",
                                "publishedAt": (now - timedelta(days=5)).isoformat(),
                                "defaultLanguage": "en",
                            },
                            "statistics": {"viewCount": "95000", "likeCount": "1000", "commentCount": "110"},
                            "contentDetails": {"duration": "PT9M"},
                        },
                        {
                            "id": "video-4",
                            "snippet": {
                                "title": "AI Automation Systems",
                                "description": "Older upload that should fail freshness.",
                                "channelId": "channel-4",
                                "channelTitle": "Channel Four",
                                "publishedAt": (now - timedelta(days=25)).isoformat(),
                                "defaultLanguage": "en",
                            },
                            "statistics": {"viewCount": "60000", "likeCount": "1900", "commentCount": "130"},
                            "contentDetails": {"duration": "PT8M"},
                        },
                    ]
                }

            return {
                "items": [
                    {
                        "id": "baseline-1",
                        "snippet": {"publishedAt": (now - timedelta(days=20)).isoformat()},
                        "statistics": {"viewCount": "12000", "likeCount": "400", "commentCount": "50"},
                    },
                    {
                        "id": "baseline-2",
                        "snippet": {"publishedAt": (now - timedelta(days=18)).isoformat()},
                        "statistics": {"viewCount": "13500", "likeCount": "420", "commentCount": "44"},
                    },
                    {
                        "id": "baseline-3",
                        "snippet": {"publishedAt": (now - timedelta(days=15)).isoformat()},
                        "statistics": {"viewCount": "11000", "likeCount": "390", "commentCount": "40"},
                    },
                ]
            }

        if endpoint == "channels":
            return {
                "items": [
                    {
                        "id": "channel-1",
                        "statistics": {"subscriberCount": "18000"},
                        "contentDetails": {"relatedPlaylists": {"uploads": "uploads-1"}},
                        "brandingSettings": {"channel": {"defaultLanguage": "en", "country": "US"}},
                    },
                    {
                        "id": "channel-2",
                        "statistics": {"subscriberCount": "22000"},
                        "contentDetails": {"relatedPlaylists": {"uploads": "uploads-2"}},
                        "brandingSettings": {"channel": {"defaultLanguage": "es", "country": "ES"}},
                    },
                    {
                        "id": "channel-3",
                        "statistics": {"subscriberCount": "32000"},
                        "contentDetails": {"relatedPlaylists": {"uploads": "uploads-3"}},
                        "brandingSettings": {"channel": {"defaultLanguage": "en", "country": "US"}},
                    },
                    {
                        "id": "channel-4",
                        "statistics": {"subscriberCount": "45000"},
                        "contentDetails": {"relatedPlaylists": {"uploads": "uploads-4"}},
                        "brandingSettings": {"channel": {"defaultLanguage": "en", "country": "US"}},
                    },
                ]
            }

        if endpoint == "playlistItems":
            return {
                "items": [
                    {
                        "contentDetails": {"videoId": "baseline-1"},
                        "snippet": {"publishedAt": (now - timedelta(days=20)).isoformat()},
                    },
                    {
                        "contentDetails": {"videoId": "baseline-2"},
                        "snippet": {"publishedAt": (now - timedelta(days=18)).isoformat()},
                    },
                    {
                        "contentDetails": {"videoId": "baseline-3"},
                        "snippet": {"publishedAt": (now - timedelta(days=15)).isoformat()},
                    },
                ]
            }

        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(outliers_finder, "run_with_provider_keys", fake_run_with_provider_keys)
    monkeypatch.setattr(outliers_finder, "_youtube_get", fake_youtube_get)

    result = outliers_finder.search_outlier_videos(request)

    assert result.scanned_videos == 1
    assert result.scanned_channels == 1
    assert result.baseline_channels == 1
    assert len(result.candidates) == 1
    assert result.candidates[0].video_id == "video-1"
    assert result.candidates[0].language_confidence_label == "High"
    assert result.candidates[0].duration_bucket == "4-12 min"
