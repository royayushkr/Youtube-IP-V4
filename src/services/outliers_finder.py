from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
import streamlit as st

from src.utils.api_keys import run_with_provider_keys


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
OUTLIER_CACHE_TTL_SECONDS = 3600
BASELINE_CACHE_TTL_SECONDS = 21600
AGE_BUCKET_ORDER = ["0-2d", "3-7d", "8-30d", "31-90d", "90d+"]
DURATION_BUCKET_ORDER = ["Shorts (<=60s)", "1-4 min", "4-12 min", "12-30 min", "30+ min", "Unknown"]
TITLE_PATTERN_ORDER = [
    "How / Why",
    "Numbered",
    "Challenge / Test",
    "Explainer",
    "Versus",
    "News / Update",
    "General",
]
SUBSCRIBER_BUCKETS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "Any": (None, None),
    "0 - 10K": (0, 10_000),
    "10K - 100K": (10_000, 100_000),
    "100K - 1M": (100_000, 1_000_000),
    "1M+": (1_000_000, None),
}
DURATION_BUCKETS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "Any": (None, None),
    "Shorts (<=60s)": (0, 60),
    "1-4 min": (61, 4 * 60),
    "4-12 min": (4 * 60 + 1, 12 * 60),
    "12-30 min": (12 * 60 + 1, 30 * 60),
    "30+ min": (30 * 60 + 1, None),
}
LANGUAGE_THRESHOLD_MAP = {
    "strict": 0.72,
    "balanced": 0.45,
    "loose": 0.20,
}
LATIN_LANGUAGES = {"en", "es", "pt", "de", "fr"}
NON_LATIN_PATTERNS = {
    "hi": re.compile(r"[\u0900-\u097F]"),
    "ja": re.compile(r"[\u3040-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]"),
}
STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "for", "with", "on", "and", "or", "at", "is",
    "are", "was", "were", "this", "that", "how", "why", "what", "when", "from", "your",
    "you", "my", "we", "our", "it", "vs", "into", "their", "about", "using", "over",
}


@dataclass(frozen=True)
class OutlierSearchRequest:
    niche_query: str
    published_after_iso: str
    published_before_iso: str
    region_code: str = ""
    relevance_language: str = ""
    language_strictness: str = "strict"
    subscriber_bucket: str = "Any"
    include_hidden_subscribers: bool = True
    min_subscribers: Optional[int] = None
    max_subscribers: Optional[int] = None
    min_views: int = 0
    duration_preference: str = "Any"
    freshness_days: Optional[int] = None
    exclude_keywords: Tuple[str, ...] = tuple()
    match_mode: str = "broad"
    max_results: int = 100
    baseline_channel_limit: int = 15
    baseline_video_cap: int = 20
    baseline_lookback_days: int = 180


@dataclass(frozen=True)
class ChannelBaseline:
    channel_id: str
    channel_title: str
    sample_size: int
    median_views: float
    median_views_per_day: float
    median_engagement_rate: float
    median_views_per_subscriber: Optional[float]


@dataclass(frozen=True)
class OutlierCandidate:
    video_id: str
    video_title: str
    channel_id: str
    channel_title: str
    video_url: str
    thumbnail_url: str
    published_at_iso: str
    age_hours: float
    age_days: float
    views: int
    likes: int
    comments: int
    engagement_rate: float
    views_per_day: float
    views_per_subscriber: Optional[float]
    channel_subscriber_count: Optional[int]
    hidden_subscriber_count: bool
    channel_country: str
    channel_default_language: str
    video_default_language: str
    language_confidence: float
    language_confidence_label: str
    duration_seconds: int
    duration_bucket: str
    title_pattern: str
    outlier_score: float
    peer_percentile: float
    engagement_percentile: float
    baseline_component: Optional[float]
    recency_boost: float
    baseline_sample_size: int
    baseline_views_ratio: Optional[float]
    baseline_engagement_ratio: Optional[float]
    size_bucket: str
    age_bucket: str
    why_outlier: str
    research_cue: str


@dataclass(frozen=True)
class OutlierSearchResult:
    request: OutlierSearchRequest
    candidates: Tuple[OutlierCandidate, ...]
    warnings: Tuple[str, ...]
    scanned_videos: int
    scanned_channels: int
    baseline_channels: int
    cache_policy: str
    quota_profile: str

    def to_frame(self) -> pd.DataFrame:
        rows = [asdict(candidate) for candidate in self.candidates]
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["explanation_text"] = frame[["why_outlier", "research_cue"]].apply(
            lambda row: " | ".join(str(value) for value in row if str(value).strip()),
            axis=1,
        )
        return frame


def _safe_get(d: Mapping[str, Any], path: Sequence[str], default=None):
    current: Any = d
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", False):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_language_code(value: str) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text.split("-", 1)[0] if text else ""


def _parse_timestamp(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise RuntimeError(f"Invalid timestamp returned by YouTube API: {value}")
    return parsed


def _age_hours_from_timestamp(published_at: pd.Timestamp) -> float:
    now = pd.Timestamp.now(tz=timezone.utc)
    delta_hours = (now - published_at).total_seconds() / 3600
    return max(delta_hours, 1.0)


def _bucket_for_age(age_days: float) -> str:
    if age_days <= 2:
        return "0-2d"
    if age_days <= 7:
        return "3-7d"
    if age_days <= 30:
        return "8-30d"
    if age_days <= 90:
        return "31-90d"
    return "90d+"


def _bucket_for_subscribers(subscribers: Optional[int], hidden: bool) -> str:
    if hidden or subscribers is None:
        return "Hidden"
    if subscribers < 10_000:
        return "0 - 10K"
    if subscribers < 100_000:
        return "10K - 100K"
    if subscribers < 1_000_000:
        return "100K - 1M"
    return "1M+"


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


def _duration_bucket_for_seconds(duration_seconds: int) -> str:
    if duration_seconds <= 0:
        return "Unknown"
    for label, (minimum, maximum) in DURATION_BUCKETS.items():
        if label == "Any":
            continue
        lower_bound = minimum or 0
        if duration_seconds < lower_bound:
            continue
        if maximum is not None and duration_seconds > maximum:
            continue
        return label
    return "30+ min"


def _weighted_average(values: Iterable[Tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        if value is None or pd.isna(value):
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def _ratio_to_unit_interval(ratio: Optional[float], cap: float) -> Optional[float]:
    if ratio is None or pd.isna(ratio) or ratio <= 0:
        return None
    capped = min(float(ratio), cap)
    return max(0.0, min(math.log1p(capped) / math.log1p(cap), 1.0))


def _percentile(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    return values.rank(pct=True, method="average").clip(0.0, 1.0)


def _title_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def _title_pattern(title: str) -> str:
    lower = str(title or "").lower()
    stripped = lower.strip()
    if re.search(r"\bhow\b|\bwhy\b", stripped):
        return "How / Why"
    if re.search(r"\b\d+\b", stripped):
        return "Numbered"
    if " vs " in stripped or " versus " in stripped:
        return "Versus"
    if any(token in stripped for token in ("challenge", "test", "experiment", "trying")):
        return "Challenge / Test"
    if any(token in stripped for token in ("explained", "guide", "breakdown", "tutorial")):
        return "Explainer"
    if any(token in stripped for token in ("news", "update", "announced", "just happened")):
        return "News / Update"
    return "General"


def _title_script_confidence(title: str, language_code: str) -> float:
    target = _normalize_language_code(language_code)
    text = str(title or "")
    if not text or not target:
        return 0.0

    if target in NON_LATIN_PATTERNS:
        return 1.0 if NON_LATIN_PATTERNS[target].search(text) else 0.0

    if target in LATIN_LANGUAGES:
        latin_chars = len(re.findall(r"[A-Za-z]", text))
        total_letters = len(re.findall(r"[A-Za-z\u00C0-\u024F]", text))
        if total_letters <= 0:
            return 0.0
        return min(1.0, latin_chars / total_letters + 0.20)

    return 0.0


def _language_confidence(
    title: str,
    video_default_language: str,
    channel_default_language: str,
    target_language: str,
) -> float:
    target = _normalize_language_code(target_language)
    if not target:
        return 1.0

    video_lang = _normalize_language_code(video_default_language)
    channel_lang = _normalize_language_code(channel_default_language)
    score = 0.0

    if video_lang == target:
        score += 0.55
    elif video_lang and video_lang != target:
        score -= 0.20

    if channel_lang == target:
        score += 0.20
    elif channel_lang and channel_lang != target:
        score -= 0.08

    score += _title_script_confidence(title, target) * 0.25

    if target == "en" and not video_lang and not channel_lang:
        if re.search(r"[\u0900-\u097F\u3040-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]", title):
            score -= 0.35

    return max(0.0, min(score, 1.0))


def _confidence_label(score: float) -> str:
    if score >= 0.72:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def _language_threshold(strictness: str) -> float:
    return LANGUAGE_THRESHOLD_MAP.get(str(strictness or "").strip().lower(), 0.45)


def _query_string_for_request(request: OutlierSearchRequest) -> str:
    query = request.niche_query.strip()
    if request.match_mode.lower() == "exact":
        query = f'"{query}"'
    for keyword in request.exclude_keywords:
        text = str(keyword).strip()
        if not text:
            continue
        query += f' -"{text}"' if " " in text else f" -{text}"
    return query


def _matches_query_mode(title: str, description: str, request: OutlierSearchRequest) -> bool:
    if request.match_mode.lower() != "exact":
        return True
    haystack = f"{title} {description}".lower()
    return request.niche_query.strip().lower() in haystack


def _contains_excluded_keyword(title: str, description: str, exclude_keywords: Sequence[str]) -> bool:
    haystack = f"{title} {description}".lower()
    return any(str(keyword).strip().lower() in haystack for keyword in exclude_keywords if str(keyword).strip())


def _is_youtube_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_tokens = (
        "quota",
        "rate limit",
        "resource exhausted",
        "daily limit",
        "too many requests",
        "backenderror",
        "service unavailable",
        "403",
        "429",
        "500",
        "503",
        "api key",
    )
    return any(token in text for token in retry_tokens)


def _youtube_get(api_key: str, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    payload = {key: value for key, value in params.items() if value not in ("", None)}
    payload["key"] = api_key
    response = requests.get(f"{YOUTUBE_API_BASE}/{endpoint}", params=payload, timeout=45)
    if response.status_code >= 400:
        try:
            body = response.json()
            error_message = _safe_get(body, ["error", "message"], response.text[:300])
        except Exception:
            error_message = response.text[:300]
        raise RuntimeError(
            f"YouTube API error ({response.status_code}) on {endpoint}: {error_message}"
        )
    return response.json()


def _search_video_ids(api_key: str, request: OutlierSearchRequest) -> List[str]:
    video_ids: List[str] = []
    page_token: Optional[str] = None
    pages = min(max(1, math.ceil(request.max_results / 50)), 4)

    for _ in range(pages):
        body = _youtube_get(
            api_key,
            "search",
            {
                "part": "snippet",
                "q": _query_string_for_request(request),
                "type": "video",
                "order": "date",
                "publishedAfter": request.published_after_iso,
                "publishedBefore": request.published_before_iso,
                "regionCode": request.region_code or None,
                "relevanceLanguage": request.relevance_language or None,
                "maxResults": min(50, request.max_results - len(video_ids)),
                "pageToken": page_token or None,
            },
        )
        for item in body.get("items", []):
            video_id = _safe_get(item, ["id", "videoId"])
            if video_id:
                video_ids.append(str(video_id))
        page_token = body.get("nextPageToken")
        if not page_token or len(video_ids) >= request.max_results:
            break

    seen = set()
    deduped: List[str] = []
    for video_id in video_ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        deduped.append(video_id)
    return deduped[: request.max_results]


def _fetch_videos(api_key: str, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start : start + 50]
        if not chunk:
            continue
        body = _youtube_get(
            api_key,
            "videos",
            {
                "part": "snippet,statistics,contentDetails,status",
                "id": ",".join(chunk),
                "maxResults": len(chunk),
            },
        )
        rows.extend(body.get("items", []))
    return rows


def _fetch_channels(api_key: str, channel_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(channel_ids), 50):
        chunk = channel_ids[start : start + 50]
        if not chunk:
            continue
        body = _youtube_get(
            api_key,
            "channels",
            {
                "part": "snippet,statistics,contentDetails,brandingSettings,status",
                "id": ",".join(chunk),
                "maxResults": len(chunk),
            },
        )
        for item in body.get("items", []):
            channel_id = str(item.get("id", "")).strip()
            if channel_id:
                out[channel_id] = item
    return out


def _filter_subscriber_bucket(
    frame: pd.DataFrame,
    subscriber_bucket: str,
    include_hidden_subscribers: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    min_subs, max_subs = SUBSCRIBER_BUCKETS.get(subscriber_bucket, (None, None))

    mask = pd.Series(True, index=frame.index)
    if not include_hidden_subscribers:
        mask &= ~frame["hidden_subscriber_count"].fillna(False)

    if min_subs is not None:
        mask &= frame["channel_subscriber_count"].fillna(-1) >= min_subs
    if max_subs is not None:
        mask &= frame["channel_subscriber_count"].fillna(float("inf")) < max_subs
    return frame[mask].copy()


def _apply_request_filters(frame: pd.DataFrame, request: OutlierSearchRequest) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = _filter_subscriber_bucket(
        frame,
        request.subscriber_bucket,
        request.include_hidden_subscribers,
    )
    if out.empty:
        return out

    if request.min_subscribers is not None:
        out = out[
            out["channel_subscriber_count"].fillna(-1) >= int(request.min_subscribers)
        ]
    if request.max_subscribers is not None:
        out = out[
            out["channel_subscriber_count"].fillna(float("inf")) <= int(request.max_subscribers)
        ]
    if request.min_views > 0:
        out = out[out["views"] >= int(request.min_views)]
    if request.freshness_days is not None:
        out = out[out["age_days"] <= float(request.freshness_days)]
    if request.duration_preference != "Any":
        out = out[out["duration_bucket"] == request.duration_preference]
    if request.match_mode.lower() == "exact":
        out = out[out["exact_match"].fillna(False)]
    if request.exclude_keywords:
        out = out[~out["excluded_keyword_hit"].fillna(False)]
    if request.relevance_language:
        threshold = _language_threshold(request.language_strictness)
        out = out[out["language_confidence"] >= threshold]

    return out.copy()


def _build_candidate_frame(
    videos: Sequence[Dict[str, Any]],
    channels: Mapping[str, Dict[str, Any]],
    request: OutlierSearchRequest,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    published_after = _parse_timestamp(request.published_after_iso)
    published_before = _parse_timestamp(request.published_before_iso)

    for video in videos:
        snippet = video.get("snippet", {}) or {}
        stats = video.get("statistics", {}) or {}
        content = video.get("contentDetails", {}) or {}
        channel_id = str(snippet.get("channelId", "")).strip()
        if not channel_id or channel_id not in channels:
            continue

        published_at_raw = snippet.get("publishedAt")
        if not published_at_raw:
            continue
        published_at = _parse_timestamp(str(published_at_raw))
        if published_at < published_after or published_at > published_before:
            continue

        channel = channels[channel_id]
        channel_stats = channel.get("statistics", {}) or {}
        channel_branding = _safe_get(channel, ["brandingSettings", "channel"], {}) or {}

        subscribers = _coerce_int(channel_stats.get("subscriberCount"))
        views = _coerce_int(stats.get("viewCount")) or 0
        likes = _coerce_int(stats.get("likeCount")) or 0
        comments = _coerce_int(stats.get("commentCount")) or 0
        duration_seconds = _parse_iso_duration_seconds(str(content.get("duration", "")))

        title = str(snippet.get("title", "")).strip()
        description = str(snippet.get("description", "")).strip()
        video_lang = str(snippet.get("defaultLanguage") or snippet.get("defaultAudioLanguage") or "").strip()
        channel_lang = str(channel_branding.get("defaultLanguage", "")).strip()
        language_confidence = _language_confidence(
            title=title,
            video_default_language=video_lang,
            channel_default_language=channel_lang,
            target_language=request.relevance_language,
        )

        age_hours = _age_hours_from_timestamp(published_at)
        age_days = age_hours / 24.0
        engagement_rate = (likes + comments) / max(views, 1)
        views_per_day = views / max(age_days, 1 / 24)
        hidden_subscribers = subscribers in (None, 0)
        views_per_subscriber = None
        if subscribers and subscribers > 0:
            views_per_subscriber = views / subscribers

        thumbnails = snippet.get("thumbnails", {}) or {}
        thumbnail_url = (
            _safe_get(thumbnails, ["high", "url"])
            or _safe_get(thumbnails, ["medium", "url"])
            or _safe_get(thumbnails, ["default", "url"])
            or ""
        )

        rows.append(
            {
                "video_id": str(video.get("id", "")).strip(),
                "video_title": title,
                "video_description": description,
                "channel_id": channel_id,
                "channel_title": str(snippet.get("channelTitle", "")).strip()
                or str(_safe_get(channel, ["snippet", "title"], "")).strip(),
                "video_url": f"https://www.youtube.com/watch?v={video.get('id', '')}",
                "thumbnail_url": thumbnail_url,
                "published_at_iso": published_at.isoformat(),
                "age_hours": age_hours,
                "age_days": age_days,
                "views": views,
                "likes": likes,
                "comments": comments,
                "engagement_rate": engagement_rate,
                "views_per_day": views_per_day,
                "views_per_subscriber": views_per_subscriber,
                "channel_subscriber_count": subscribers,
                "hidden_subscriber_count": hidden_subscribers,
                "channel_country": str(channel_branding.get("country", "")).strip(),
                "channel_default_language": channel_lang,
                "video_default_language": video_lang,
                "language_confidence": language_confidence,
                "language_confidence_label": _confidence_label(language_confidence),
                "duration_seconds": duration_seconds,
                "duration_bucket": _duration_bucket_for_seconds(duration_seconds),
                "title_pattern": _title_pattern(title),
                "exact_match": _matches_query_mode(title, description, request),
                "excluded_keyword_hit": _contains_excluded_keyword(
                    title,
                    description,
                    request.exclude_keywords,
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = _apply_request_filters(frame, request)
    if frame.empty:
        return frame

    frame["size_bucket"] = frame.apply(
        lambda row: _bucket_for_subscribers(
            _coerce_int(row["channel_subscriber_count"]),
            bool(row["hidden_subscriber_count"]),
        ),
        axis=1,
    )
    frame["age_bucket"] = pd.Categorical(
        frame["age_days"].apply(_bucket_for_age),
        categories=AGE_BUCKET_ORDER,
        ordered=True,
    )
    frame["duration_bucket"] = pd.Categorical(
        frame["duration_bucket"],
        categories=DURATION_BUCKET_ORDER,
        ordered=True,
    )
    frame["title_pattern"] = pd.Categorical(
        frame["title_pattern"],
        categories=TITLE_PATTERN_ORDER,
        ordered=True,
    )
    return frame


def _prepare_peer_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    out["views_per_day_log"] = out["views_per_day"].fillna(0).apply(lambda value: math.log1p(max(value, 0)))
    out["views_per_subscriber_filled"] = out["views_per_subscriber"].fillna(0.40)
    out["views_per_subscriber_log"] = out["views_per_subscriber_filled"].apply(
        lambda value: math.log1p(max(value, 0))
    )

    detailed_group_counts = (
        out.groupby(["size_bucket", "age_bucket"], dropna=False, observed=True)["video_id"]
        .transform("count")
    )
    age_group_counts = out.groupby("age_bucket", dropna=False, observed=True)["video_id"].transform("count")

    out["peer_group"] = out["size_bucket"].astype(str) + "|" + out["age_bucket"].astype(str)
    out.loc[detailed_group_counts < 5, "peer_group"] = "age:" + out["age_bucket"].astype(str)
    out.loc[age_group_counts < 8, "peer_group"] = "all"

    out["peer_views_percentile"] = (
        out.groupby("peer_group", dropna=False)["views_per_day_log"]
        .transform(_percentile)
        .fillna(0.5)
    )
    out["peer_engagement_percentile"] = (
        out.groupby("peer_group", dropna=False)["engagement_rate"]
        .transform(_percentile)
        .fillna(0.5)
    )
    out["peer_subscriber_percentile"] = (
        out.groupby("peer_group", dropna=False)["views_per_subscriber_log"]
        .transform(_percentile)
        .fillna(0.5)
    )
    out["peer_percentile"] = (
        out["peer_views_percentile"] * 0.60
        + out["peer_engagement_percentile"] * 0.25
        + out["peer_subscriber_percentile"] * 0.15
    )
    return out


def _recency_boost(age_days: float, request: OutlierSearchRequest) -> float:
    return max(0.1, 1 - min(age_days / max(request.baseline_lookback_days, 1), 0.9))


@st.cache_data(ttl=BASELINE_CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_channel_baseline_cached(
    channel_id: str,
    uploads_playlist_id: str,
    channel_title: str,
    subscriber_count: Optional[int],
    lookback_days: int,
    max_videos: int,
) -> Optional[ChannelBaseline]:
    if not uploads_playlist_id:
        return None

    def _load_with_key(api_key: str) -> Optional[ChannelBaseline]:
        published_after = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=lookback_days)
        video_ids: List[str] = []
        page_token: Optional[str] = None
        stop = False

        while len(video_ids) < max_videos and not stop:
            body = _youtube_get(
                api_key,
                "playlistItems",
                {
                    "part": "contentDetails,snippet",
                    "playlistId": uploads_playlist_id,
                    "maxResults": min(50, max_videos - len(video_ids)),
                    "pageToken": page_token or None,
                },
            )
            for item in body.get("items", []):
                video_id = _safe_get(item, ["contentDetails", "videoId"])
                published_at_raw = _safe_get(item, ["snippet", "publishedAt"])
                if not video_id or not published_at_raw:
                    continue
                published_at = _parse_timestamp(str(published_at_raw))
                if published_at < published_after:
                    stop = True
                    break
                video_ids.append(str(video_id))
            page_token = body.get("nextPageToken")
            if not page_token:
                break

        if not video_ids:
            return None

        details = _fetch_videos(api_key, video_ids)
        baseline_rows: List[Dict[str, Any]] = []
        for video in details:
            snippet = video.get("snippet", {}) or {}
            stats = video.get("statistics", {}) or {}
            published_at_raw = snippet.get("publishedAt")
            if not published_at_raw:
                continue
            published_at = _parse_timestamp(str(published_at_raw))
            views = _coerce_int(stats.get("viewCount")) or 0
            likes = _coerce_int(stats.get("likeCount")) or 0
            comments = _coerce_int(stats.get("commentCount")) or 0
            age_hours = _age_hours_from_timestamp(published_at)
            age_days = age_hours / 24.0
            views_per_day = views / max(age_days, 1 / 24)
            engagement_rate = (likes + comments) / max(views, 1)
            views_per_subscriber = None
            if subscriber_count and subscriber_count > 0:
                views_per_subscriber = views / subscriber_count
            baseline_rows.append(
                {
                    "views": views,
                    "views_per_day": views_per_day,
                    "engagement_rate": engagement_rate,
                    "views_per_subscriber": views_per_subscriber,
                }
            )

        baseline_frame = pd.DataFrame(baseline_rows)
        if baseline_frame.empty:
            return None

        return ChannelBaseline(
            channel_id=channel_id,
            channel_title=channel_title,
            sample_size=int(len(baseline_frame)),
            median_views=float(baseline_frame["views"].median()),
            median_views_per_day=float(baseline_frame["views_per_day"].median()),
            median_engagement_rate=float(baseline_frame["engagement_rate"].median()),
            median_views_per_subscriber=(
                float(baseline_frame["views_per_subscriber"].dropna().median())
                if baseline_frame["views_per_subscriber"].notna().any()
                else None
            ),
        )

    return run_with_provider_keys(
        "youtube",
        _load_with_key,
        retryable_error=_is_youtube_retryable_error,
    )


def _score_outlier_frame(
    frame: pd.DataFrame,
    baselines: Mapping[str, ChannelBaseline],
    request: OutlierSearchRequest,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = _prepare_peer_percentiles(frame)
    out["recency_boost"] = out["age_days"].apply(lambda value: _recency_boost(float(value), request))

    baseline_components: List[Optional[float]] = []
    baseline_views_ratios: List[Optional[float]] = []
    baseline_engagement_ratios: List[Optional[float]] = []
    baseline_sample_sizes: List[int] = []

    for row in out.to_dict(orient="records"):
        baseline = baselines.get(str(row["channel_id"]))
        baseline_sample_sizes.append(int(baseline.sample_size) if baseline else 0)
        if not baseline or baseline.sample_size < 3:
            baseline_components.append(None)
            baseline_views_ratios.append(None)
            baseline_engagement_ratios.append(None)
            continue

        view_ratio = None
        if baseline.median_views_per_day > 0:
            view_ratio = float(row["views_per_day"]) / baseline.median_views_per_day
        engagement_ratio = None
        if baseline.median_engagement_rate > 0:
            engagement_ratio = float(row["engagement_rate"]) / baseline.median_engagement_rate
        subscriber_ratio = None
        if (
            baseline.median_views_per_subscriber is not None
            and baseline.median_views_per_subscriber > 0
            and row.get("views_per_subscriber") is not None
        ):
            subscriber_ratio = float(row["views_per_subscriber"]) / baseline.median_views_per_subscriber

        baseline_component = _weighted_average(
            [
                (_ratio_to_unit_interval(view_ratio, cap=8), 0.60),
                (_ratio_to_unit_interval(engagement_ratio, cap=4), 0.25),
                (_ratio_to_unit_interval(subscriber_ratio, cap=4), 0.15),
            ]
        )
        baseline_components.append(baseline_component)
        baseline_views_ratios.append(view_ratio)
        baseline_engagement_ratios.append(engagement_ratio)

    out["baseline_component"] = baseline_components
    out["baseline_views_ratio"] = baseline_views_ratios
    out["baseline_engagement_ratio"] = baseline_engagement_ratios
    out["baseline_sample_size"] = baseline_sample_sizes
    out["engagement_percentile"] = out["peer_engagement_percentile"].fillna(0.5)

    scores: List[float] = []
    for row in out.to_dict(orient="records"):
        if row.get("baseline_component") is not None and not pd.isna(row["baseline_component"]):
            score = 100 * (
                0.45 * float(row["baseline_component"])
                + 0.30 * float(row["peer_percentile"])
                + 0.15 * float(row["engagement_percentile"])
                + 0.10 * float(row["recency_boost"])
            )
        else:
            score = 100 * (
                0.55 * float(row["peer_percentile"])
                + 0.25 * float(row["engagement_percentile"])
                + 0.20 * float(row["recency_boost"])
            )
        scores.append(score)

    out["outlier_score"] = scores
    out = out.sort_values("outlier_score", ascending=False).reset_index(drop=True)
    return out


def _primary_outlier_reason(row: Mapping[str, Any]) -> str:
    reasons: List[Tuple[float, str]] = []
    baseline_view_ratio = _coerce_float(row.get("baseline_views_ratio"))
    baseline_engagement_ratio = _coerce_float(row.get("baseline_engagement_ratio"))
    peer_percentile = float(row.get("peer_percentile") or 0)
    engagement_percentile = float(row.get("engagement_percentile") or 0)
    age_days = float(row.get("age_days") or 0)

    if baseline_view_ratio and baseline_view_ratio >= 1.2:
        reasons.append((baseline_view_ratio, f"{baseline_view_ratio:.1f}x the channel's median views/day"))
    if baseline_engagement_ratio and baseline_engagement_ratio >= 1.15:
        reasons.append((baseline_engagement_ratio, f"{baseline_engagement_ratio:.1f}x the channel's median engagement rate"))
    if peer_percentile >= 0.85:
        top_share = max(1, int(round((1 - peer_percentile) * 100)))
        reasons.append((peer_percentile, f"Top {top_share}% for age-adjusted performance in this scanned cohort"))
    if engagement_percentile >= 0.80:
        top_share = max(1, int(round((1 - engagement_percentile) * 100)))
        reasons.append((engagement_percentile, f"Top {top_share}% for engagement among comparable results"))
    if age_days <= 7:
        reasons.append((1.1, f"Published {age_days:.1f} day(s) ago and already accelerating"))

    if not reasons:
        return "Strong relative performance across the scanned results"

    reasons.sort(key=lambda item: item[0], reverse=True)
    return reasons[0][1]


def _research_cue(row: Mapping[str, Any]) -> str:
    title_pattern = str(row.get("title_pattern") or "General")
    duration_bucket = str(row.get("duration_bucket") or "Unknown")
    language_label = str(row.get("language_confidence_label") or "Low")
    cues: List[str] = []

    if title_pattern != "General":
        cues.append(f"{title_pattern} packaging is repeating in this niche")
    if duration_bucket not in ("Any", "Unknown"):
        cues.append(f"{duration_bucket} videos are surfacing in the winning set")
    if language_label == "High":
        cues.append("Language match is strong enough for focused niche research")
    if not cues:
        cues.append("Review the hook, framing, and runtime for repeatable packaging ideas")
    return cues[0]


def _frame_to_candidates(frame: pd.DataFrame) -> Tuple[OutlierCandidate, ...]:
    candidates: List[OutlierCandidate] = []
    for row in frame.to_dict(orient="records"):
        why_outlier = _primary_outlier_reason(row)
        research_cue = _research_cue(row)
        candidates.append(
            OutlierCandidate(
                video_id=str(row["video_id"]),
                video_title=str(row["video_title"]),
                channel_id=str(row["channel_id"]),
                channel_title=str(row["channel_title"]),
                video_url=str(row["video_url"]),
                thumbnail_url=str(row.get("thumbnail_url", "")),
                published_at_iso=str(row["published_at_iso"]),
                age_hours=float(row["age_hours"]),
                age_days=float(row["age_days"]),
                views=int(row["views"]),
                likes=int(row["likes"]),
                comments=int(row["comments"]),
                engagement_rate=float(row["engagement_rate"]),
                views_per_day=float(row["views_per_day"]),
                views_per_subscriber=(
                    float(row["views_per_subscriber"])
                    if row.get("views_per_subscriber") is not None and not pd.isna(row["views_per_subscriber"])
                    else None
                ),
                channel_subscriber_count=(
                    int(row["channel_subscriber_count"])
                    if row.get("channel_subscriber_count") is not None and not pd.isna(row["channel_subscriber_count"])
                    else None
                ),
                hidden_subscriber_count=bool(row["hidden_subscriber_count"]),
                channel_country=str(row.get("channel_country", "")),
                channel_default_language=str(row.get("channel_default_language", "")),
                video_default_language=str(row.get("video_default_language", "")),
                language_confidence=float(row.get("language_confidence") or 0.0),
                language_confidence_label=str(row.get("language_confidence_label", "Low")),
                duration_seconds=int(row.get("duration_seconds") or 0),
                duration_bucket=str(row.get("duration_bucket", "Unknown")),
                title_pattern=str(row.get("title_pattern", "General")),
                outlier_score=float(row["outlier_score"]),
                peer_percentile=float(row["peer_percentile"]),
                engagement_percentile=float(row["engagement_percentile"]),
                baseline_component=(
                    float(row["baseline_component"])
                    if row.get("baseline_component") is not None and not pd.isna(row["baseline_component"])
                    else None
                ),
                recency_boost=float(row["recency_boost"]),
                baseline_sample_size=int(row["baseline_sample_size"]),
                baseline_views_ratio=(
                    float(row["baseline_views_ratio"])
                    if row.get("baseline_views_ratio") is not None and not pd.isna(row["baseline_views_ratio"])
                    else None
                ),
                baseline_engagement_ratio=(
                    float(row["baseline_engagement_ratio"])
                    if row.get("baseline_engagement_ratio") is not None and not pd.isna(row["baseline_engagement_ratio"])
                    else None
                ),
                size_bucket=str(row["size_bucket"]),
                age_bucket=str(row["age_bucket"]),
                why_outlier=why_outlier,
                research_cue=research_cue,
            )
        )
    return tuple(candidates)


def score_outlier_candidates_frame(
    frame: pd.DataFrame,
    request: OutlierSearchRequest,
    baselines: Mapping[str, ChannelBaseline],
) -> pd.DataFrame:
    return _score_outlier_frame(frame, baselines, request)


def filter_candidates_by_subscriber_bucket(
    frame: pd.DataFrame,
    subscriber_bucket: str,
    include_hidden_subscribers: bool,
) -> pd.DataFrame:
    return _filter_subscriber_bucket(frame, subscriber_bucket, include_hidden_subscribers)


def build_age_bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["age_bucket", "outlier_count", "median_outlier_score", "median_views_per_day"])
    working = frame.copy()
    working["age_bucket"] = pd.Categorical(
        working["age_bucket"],
        categories=AGE_BUCKET_ORDER,
        ordered=True,
    )
    out = (
        working.groupby("age_bucket", dropna=False, observed=True)
        .agg(
            outlier_count=("video_id", "count"),
            median_outlier_score=("outlier_score", "median"),
            median_views_per_day=("views_per_day", "median"),
        )
        .reset_index()
    )
    return out.sort_values("age_bucket")


def build_duration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["duration_bucket", "outlier_count", "median_outlier_score", "median_views_per_day"])
    working = frame.copy()
    working["duration_bucket"] = pd.Categorical(
        working["duration_bucket"],
        categories=[bucket for bucket in DURATION_BUCKET_ORDER if bucket != "Unknown"] + ["Unknown"],
        ordered=True,
    )
    out = (
        working.groupby("duration_bucket", dropna=False, observed=True)
        .agg(
            outlier_count=("video_id", "count"),
            median_outlier_score=("outlier_score", "median"),
            median_views_per_day=("views_per_day", "median"),
        )
        .reset_index()
    )
    return out.sort_values("duration_bucket")


def build_title_pattern_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["title_pattern", "outlier_count", "median_outlier_score"])
    working = frame.copy()
    working["title_pattern"] = pd.Categorical(
        working["title_pattern"],
        categories=TITLE_PATTERN_ORDER,
        ordered=True,
    )
    out = (
        working.groupby("title_pattern", dropna=False, observed=True)
        .agg(
            outlier_count=("video_id", "count"),
            median_outlier_score=("outlier_score", "median"),
        )
        .reset_index()
    )
    return out.sort_values("title_pattern")


def build_title_keyword_summary(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["keyword", "count"])
    tokens: List[str] = []
    for title in frame["video_title"].fillna("").astype(str).tolist():
        tokens.extend(_title_tokens(title))
    if not tokens:
        return pd.DataFrame(columns=["keyword", "count"])
    series = pd.Series(tokens)
    out = series.value_counts().head(top_n).reset_index()
    out.columns = ["keyword", "count"]
    return out


@st.cache_data(ttl=OUTLIER_CACHE_TTL_SECONDS, show_spinner=False)
def _search_outlier_videos_cached(request: OutlierSearchRequest) -> OutlierSearchResult:
    warnings: List[str] = []

    def _search_with_key(api_key: str) -> OutlierSearchResult:
        video_ids = _search_video_ids(api_key, request)
        if not video_ids:
            return OutlierSearchResult(
                request=request,
                candidates=tuple(),
                warnings=("No public videos matched this niche query and timeframe.",),
                scanned_videos=0,
                scanned_channels=0,
                baseline_channels=0,
                cache_policy="1h query cache / 6h baseline cache",
                quota_profile="Interactive scan (~250-350 units uncached)",
            )

        videos = _fetch_videos(api_key, video_ids)
        channel_ids = sorted(
            {
                str(_safe_get(video, ["snippet", "channelId"], "")).strip()
                for video in videos
                if _safe_get(video, ["snippet", "channelId"])
            }
        )
        channels = _fetch_channels(api_key, channel_ids)
        frame = _build_candidate_frame(videos, channels, request)

        if frame.empty:
            return OutlierSearchResult(
                request=request,
                candidates=tuple(),
                warnings=(
                    "The scanned cohort did not contain videos that survived your filters. Try broadening language, region, subscriber, or duration settings.",
                ),
                scanned_videos=0,
                scanned_channels=0,
                baseline_channels=0,
                cache_policy="1h query cache / 6h baseline cache",
                quota_profile="Interactive scan (~250-350 units uncached)",
            )

        preliminary = _prepare_peer_percentiles(frame)
        preliminary["preliminary_score"] = 100 * (
            preliminary["peer_percentile"] * 0.70
            + preliminary["peer_engagement_percentile"] * 0.20
            + preliminary["age_days"].apply(lambda value: _recency_boost(float(value), request)) * 0.10
        )

        shortlist = (
            preliminary.sort_values("preliminary_score", ascending=False)
            .drop_duplicates("channel_id")
            .head(request.baseline_channel_limit)
        )

        baselines: Dict[str, ChannelBaseline] = {}
        baseline_failures = 0
        for row in shortlist.to_dict(orient="records"):
            uploads_playlist_id = str(
                _safe_get(
                    channels.get(str(row["channel_id"]), {}),
                    ["contentDetails", "relatedPlaylists", "uploads"],
                    "",
                )
            ).strip()
            baseline = None
            try:
                baseline = _fetch_channel_baseline_cached(
                    channel_id=str(row["channel_id"]),
                    uploads_playlist_id=uploads_playlist_id,
                    channel_title=str(row["channel_title"]),
                    subscriber_count=_coerce_int(row.get("channel_subscriber_count")),
                    lookback_days=request.baseline_lookback_days,
                    max_videos=request.baseline_video_cap,
                )
            except Exception:
                baseline_failures += 1
            if baseline:
                baselines[str(row["channel_id"])] = baseline

        if baseline_failures:
            warnings.append(
                f"Baseline enrichment failed for {baseline_failures} channel(s); affected videos use peer-only fallback scoring."
            )

        if request.relevance_language:
            threshold = _language_threshold(request.language_strictness)
            low_confidence = int((frame["language_confidence"] < threshold).sum())
            if low_confidence:
                warnings.append(
                    "Language filtering uses metadata and title heuristics. Some results may still require manual review."
                )

        scored = _score_outlier_frame(frame, baselines, request)
        return OutlierSearchResult(
            request=request,
            candidates=_frame_to_candidates(scored),
            warnings=tuple(dict.fromkeys(warnings)),
            scanned_videos=int(len(frame)),
            scanned_channels=int(frame["channel_id"].nunique()),
            baseline_channels=int(len(baselines)),
            cache_policy="1h query cache / 6h baseline cache",
            quota_profile="Interactive scan (~250-350 units uncached)",
        )

    return run_with_provider_keys(
        "youtube",
        _search_with_key,
        retryable_error=_is_youtube_retryable_error,
    )


def search_outlier_videos(request: OutlierSearchRequest) -> OutlierSearchResult:
    return _search_outlier_videos_cached(request)
