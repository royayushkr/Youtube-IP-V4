"""Service layer helpers for interactive product modules."""

from src.services.outlier_ai import InsightCard, OutlierAIReport, generate_outlier_ai_report
from src.services.outliers_finder import (
    ChannelBaseline,
    DURATION_BUCKETS,
    OutlierCandidate,
    OutlierSearchRequest,
    OutlierSearchResult,
    build_age_bucket_summary,
    build_duration_summary,
    build_title_keyword_summary,
    build_title_pattern_summary,
    search_outlier_videos,
)

__all__ = [
    "InsightCard",
    "OutlierAIReport",
    "generate_outlier_ai_report",
    "ChannelBaseline",
    "DURATION_BUCKETS",
    "OutlierCandidate",
    "OutlierSearchRequest",
    "OutlierSearchResult",
    "build_age_bucket_summary",
    "build_duration_summary",
    "build_title_keyword_summary",
    "build_title_pattern_summary",
    "search_outlier_videos",
]
