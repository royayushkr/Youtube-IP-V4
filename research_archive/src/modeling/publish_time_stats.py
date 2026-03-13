"""
publish_time_stats.py — derives per-category optimal publish windows and
upload cadence benchmarks from the training corpus and writes a static lookup JSON.

publish time and cadence signals are computed as partial correlations controlling
for log_subscribers to remove channel-size confounding.

run once during training (and nightly in the retraining job alongside xgboost/bertopic).
output is consumed by the inference orchestrator — no model, pure corpus statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_RAW_CSV   = Path("data/processed/combined_videos_raw.csv")
_CLEAN_CSV = Path("data/processed/combined_videos_clean.csv")
_OUT_PATH  = Path("outputs/models/publish_time_stats.json")

_MIN_CELL_VIDEOS    = 20
_MIN_CHANNEL_VIDEOS = 10
_TOP_QUARTILE_WEIGHT = 2.0

_CATEGORIES = [
    "entertainment", "fitness", "food", "gaming", "research_science", "tech"
]

_DOW_LABELS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _partial_corr_residuals(df: pd.DataFrame, x_col: str, y_col: str,
                             covariate: str) -> tuple[pd.Series, pd.Series]:
    """
    returns residuals of x and y after regressing out covariate via OLS.
    used to compute partial correlation controlling for log_subscribers.
    """
    sub = df[[x_col, y_col, covariate]].dropna()
    sub = sub[np.isfinite(sub).all(axis=1)]
    if len(sub) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    c = sub[covariate].values
    c_dm = c - c.mean()
    denom = (c_dm ** 2).sum()
    if denom == 0:
        return sub[x_col], sub[y_col]

    beta_x = (c_dm * sub[x_col].values).sum() / denom
    beta_y = (c_dm * sub[y_col].values).sum() / denom

    resid_x = pd.Series(sub[x_col].values - beta_x * c_dm, index=sub.index)
    resid_y = pd.Series(sub[y_col].values - beta_y * c_dm, index=sub.index)
    return resid_x, resid_y


def _partial_correlation(resid_x: pd.Series, resid_y: pd.Series) -> float:
    if len(resid_x) < 3 or resid_x.std() == 0 or resid_y.std() == 0:
        return 0.0
    corr = float(np.corrcoef(resid_x, resid_y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _hour_stats(sub: pd.DataFrame) -> tuple[dict, int | None]:
    hour_stats: dict[int, dict] = {}
    for hour in range(24):
        cell = sub[sub["publish_hour"] == hour]
        if len(cell) < _MIN_CELL_VIDEOS:
            continue
        resid_x, resid_y = _partial_corr_residuals(
            cell.assign(hour_indicator=1.0),
            "hour_indicator", "log_vpd", "log_subscribers"
        )
        # for within-cell mean we still use weighted mean — partial corr
        # is used for cross-hour ranking
        wmean = float(np.average(cell["log_vpd"], weights=cell["weight"]))
        # partial corr of being in this hour vs log_vpd controlling for subs
        # computed across all videos: in_hour indicator vs log_vpd
        indicator = (sub["publish_hour"] == hour).astype(float)
        rx, ry = _partial_corr_residuals(
            sub.assign(indicator=indicator),
            "indicator", "log_vpd", "log_subscribers"
        )
        pcorr = _partial_correlation(rx, ry)
        hour_stats[hour] = {
            "weighted_mean_log_vpd": round(wmean, 4),
            "partial_correlation":   round(pcorr, 4),
            "n_videos":              int(len(cell)),
        }

    if hour_stats:
        scores = {h: s["partial_correlation"] for h, s in hour_stats.items()}
        lo, hi = min(scores.values()), max(scores.values())
        denom  = (hi - lo) if hi > lo else 1.0
        for h in hour_stats:
            hour_stats[h]["score"] = round((scores[h] - lo) / denom, 4)
        best_hour = max(hour_stats, key=lambda h: hour_stats[h]["score"])
    else:
        best_hour = None

    return hour_stats, best_hour


def _dow_stats(sub: pd.DataFrame) -> tuple[dict, str | None]:
    dow_stats: dict[int, dict] = {}
    for dow in range(7):
        cell = sub[sub["publish_dow"] == dow]
        if len(cell) < _MIN_CELL_VIDEOS:
            continue
        wmean = float(np.average(cell["log_vpd"], weights=cell["weight"]))
        indicator = (sub["publish_dow"] == dow).astype(float)
        rx, ry = _partial_corr_residuals(
            sub.assign(indicator=indicator),
            "indicator", "log_vpd", "log_subscribers"
        )
        pcorr = _partial_correlation(rx, ry)
        dow_stats[dow] = {
            "label":                 _DOW_LABELS[dow],
            "weighted_mean_log_vpd": round(wmean, 4),
            "partial_correlation":   round(pcorr, 4),
            "n_videos":              int(len(cell)),
        }

    if dow_stats:
        scores = {d: s["partial_correlation"] for d, s in dow_stats.items()}
        lo, hi = min(scores.values()), max(scores.values())
        denom  = (hi - lo) if hi > lo else 1.0
        for d in dow_stats:
            dow_stats[d]["score"] = round((scores[d] - lo) / denom, 4)
        best_dow_idx   = max(dow_stats, key=lambda d: dow_stats[d]["score"])
        best_dow_label = _DOW_LABELS[best_dow_idx]
    else:
        best_dow_idx, best_dow_label = None, None

    return dow_stats, best_dow_label


def _cadence_stats(sub: pd.DataFrame) -> dict:
    channel_stats = []
    for channel_id, group in sub.groupby("channel_id"):
        group = group.sort_values("published_at")
        if len(group) < _MIN_CHANNEL_VIDEOS:
            continue
        span_days = (
            group["published_at"].max() - group["published_at"].min()
        ).total_seconds() / 86400
        if span_days < 7:
            continue
        videos_per_week = len(group) / (span_days / 7)
        log_subs = float(np.log1p(group["channel_subscriberCount"].iloc[0]))
        median_vpd = float(group["views_per_day"].median())
        channel_stats.append({
            "videos_per_week": videos_per_week,
            "log_subscribers": log_subs,
            "log_median_vpd":  float(np.log1p(median_vpd)),
        })

    if len(channel_stats) < 3:
        return {}

    cdf = pd.DataFrame(channel_stats)
    cdf = cdf[np.isfinite(cdf).all(axis=1)]

    rx, ry = _partial_corr_residuals(cdf, "videos_per_week", "log_median_vpd", "log_subscribers")
    pcorr  = _partial_correlation(rx, ry)

    return {
        "n_channels":             int(len(cdf)),
        "median_videos_per_week": round(float(cdf["videos_per_week"].median()), 2),
        "p25_videos_per_week":    round(float(cdf["videos_per_week"].quantile(0.25)), 2),
        "p75_videos_per_week":    round(float(cdf["videos_per_week"].quantile(0.75)), 2),
        "partial_correlation":    round(pcorr, 4),
        "interpretation":         (
            "higher cadence associated with better performance (controlling for channel size)"
            if pcorr > 0.05 else
            "lower cadence associated with better performance (controlling for channel size)"
            if pcorr < -0.05 else
            "cadence not meaningfully associated with performance after controlling for channel size"
        ),
    }


def build_publish_time_stats(
    raw_csv: str | Path = _RAW_CSV,
    clean_csv: str | Path = _CLEAN_CSV,
    output_path: str | Path = _OUT_PATH,
) -> dict:
    raw   = pd.read_csv(raw_csv, usecols=[
        "video_id", "video_publishedAt", "channel_id", "channel_subscriberCount"
    ])
    clean = pd.read_csv(clean_csv, usecols=[
        "video_id", "views_per_day", "category_name", "is_short"
    ])

    df = clean.merge(raw, on="video_id", how="inner")
    df = df[df["views_per_day"] > 0].dropna(
        subset=["views_per_day", "video_publishedAt", "channel_subscriberCount"]
    )

    df["published_at"]   = pd.to_datetime(df["video_publishedAt"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df["publish_hour"]   = df["published_at"].dt.hour
    df["publish_dow"]    = df["published_at"].dt.dayofweek
    df["log_vpd"]        = np.log1p(df["views_per_day"])
    df["log_subscribers"] = np.log1p(df["channel_subscriberCount"])
    df = df[np.isfinite(df["log_vpd"]) & np.isfinite(df["log_subscribers"])]

    df["weight"] = 1.0
    for cat in _CATEGORIES:
        mask = df["category_name"] == cat
        q75  = df.loc[mask, "log_vpd"].quantile(0.75)
        df.loc[mask & (df["log_vpd"] >= q75), "weight"] = _TOP_QUARTILE_WEIGHT

    result: dict = {}

    for cat in _CATEGORIES:
        sub = df[df["category_name"] == cat].copy().reset_index(drop=True)

        hour_stats, best_hour     = _hour_stats(sub)
        dow_stats,  best_dow      = _dow_stats(sub)
        cadence                   = _cadence_stats(sub)

        top_hours = sorted(hour_stats, key=lambda h: hour_stats[h]["score"], reverse=True)[:3]
        top_days  = sorted(dow_stats,  key=lambda d: dow_stats[d]["score"],  reverse=True)[:3]

        result[cat] = {
            "n_videos":      int(len(sub)),
            "best_hour_utc": best_hour,
            "best_dow":      best_dow,
            "top_hours_utc": top_hours,
            "top_days":      [_DOW_LABELS[d] for d in top_days],
            "hour_stats":    {str(h): v for h, v in hour_stats.items()},
            "dow_stats":     {str(d): v for d, v in dow_stats.items()},
            "cadence":       cadence,
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    build_publish_time_stats()