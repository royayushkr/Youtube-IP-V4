"""
topic_trend_baseline.py — derives per-topic engagement baselines and monthly
seasonality indices from the training corpus and writes a static lookup JSON.

provides the context needed at inference to express trend strength relative to
a topic's own historical norm rather than as a raw score.

all engagement signals are partial correlations controlling for log_subscribers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_RAW_CSV        = Path("data/processed/combined_videos_raw.csv")
_CLEAN_CSV      = Path("data/processed/combined_videos_clean.csv")
_META_CSV       = Path("data/processed/bertopic_metadata.csv")
_TOPIC_STATS    = Path("data/processed/topic_stats.csv")
_OUT_PATH       = Path("outputs/models/topic_trend_baseline.json")

_MIN_TOPIC_VIDEOS   = 20
_MIN_MONTH_VIDEOS   = 5


def _partial_corr_residuals(df: pd.DataFrame, x_col: str, y_col: str,
                             covariate: str) -> tuple[pd.Series, pd.Series]:
    sub = df[[x_col, y_col, covariate]].copy()
    sub = sub[np.isfinite(sub).all(axis=1)]
    if len(sub) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    c     = sub[covariate].values
    c_dm  = c - c.mean()
    denom = (c_dm ** 2).sum()
    if denom == 0:
        return sub[x_col], sub[y_col]
    beta_x  = (c_dm * sub[x_col].values).sum() / denom
    beta_y  = (c_dm * sub[y_col].values).sum() / denom
    resid_x = pd.Series(sub[x_col].values - beta_x * c_dm, index=sub.index)
    resid_y = pd.Series(sub[y_col].values - beta_y * c_dm, index=sub.index)
    return resid_x, resid_y


def _partial_correlation(resid_x: pd.Series, resid_y: pd.Series) -> float:
    if len(resid_x) < 3 or resid_x.std() == 0 or resid_y.std() == 0:
        return 0.0
    corr = float(np.corrcoef(resid_x, resid_y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _seasonality_index(group: pd.DataFrame) -> dict[int, float]:
    """
    per-month multiplier relative to topic annual mean.
    computed on residuals after partialling out log_subscribers.
    months with fewer than _MIN_MONTH_VIDEOS videos are omitted.
    """
    month_means: dict[int, float] = {}
    for month in range(1, 13):
        cell = group[group["publish_month_num"] == month]
        if len(cell) < _MIN_MONTH_VIDEOS:
            continue
        rx, ry = _partial_corr_residuals(cell, "publish_month_num", "log_vpd", "log_subscribers")
        month_means[month] = float(cell["log_vpd"].mean())

    if not month_means:
        return {}

    annual_mean = float(np.mean(list(month_means.values())))
    if annual_mean == 0:
        return {}

    return {
        m: round(v / annual_mean, 4)
        for m, v in month_means.items()
    }


def _trend_zscore(topic_id: int, category: str, trend_score_recent: float,
                  category_scores: dict[str, list[float]]) -> float | None:
    scores = category_scores.get(category, [])
    if len(scores) < 3:
        return None
    arr  = np.array(scores)
    mean = float(arr.mean())
    std  = float(arr.std())
    if std == 0:
        return None
    z = (trend_score_recent - mean) / std
    return round(float(z), 4) if np.isfinite(z) else None


def build_topic_trend_baseline(
    raw_csv: str | Path     = _RAW_CSV,
    clean_csv: str | Path   = _CLEAN_CSV,
    meta_csv: str | Path    = _META_CSV,
    topic_stats: str | Path = _TOPIC_STATS,
    output_path: str | Path = _OUT_PATH,
) -> dict:
    raw   = pd.read_csv(raw_csv,   usecols=["video_id", "channel_subscriberCount"])
    clean = pd.read_csv(clean_csv, usecols=["video_id", "views_per_day", "publish_month", "is_short"])
    meta  = pd.read_csv(meta_csv,  usecols=["video_id", "topic_id"])
    stats = pd.read_csv(topic_stats)

    df = clean.merge(raw,  on="video_id", how="inner")
    df = df.merge(meta,    on="video_id", how="inner")
    df = df[
        (df["views_per_day"] > 0) &
        (df["is_short"] == False) &
        (df["topic_id"] != -1)
    ].dropna(subset=["views_per_day", "publish_month", "channel_subscriberCount"])

    df["publish_month_num"] = pd.to_datetime(df["publish_month"]).dt.month
    df["log_vpd"]           = np.log1p(df["views_per_day"])
    df["log_subscribers"]   = np.log1p(df["channel_subscriberCount"])
    df["topic_id"]          = df["topic_id"].astype(int)
    df = df[np.isfinite(df["log_vpd"]) & np.isfinite(df["log_subscribers"])].reset_index(drop=True)

    stats_lookup = stats.set_index("topic_id").to_dict("index")

    # build category-level trend_score distributions for z-scoring
    category_scores: dict[str, list[float]] = {}
    for _, row in stats.iterrows():
        cat = row["top_category"]
        category_scores.setdefault(cat, []).append(float(row["trend_score_recent"]))

    result: dict = {}

    for topic_id, group in df.groupby("topic_id"):
        if len(group) < _MIN_TOPIC_VIDEOS:
            continue

        topic_id = int(topic_id)
        group    = group.reset_index(drop=True)
        ts       = stats_lookup.get(topic_id, {})
        category = str(ts.get("top_category", "unknown"))

        log_vpd_vals = group["log_vpd"].values
        baseline_mean = float(np.mean(log_vpd_vals))
        baseline_std  = float(np.std(log_vpd_vals))

        trend_score_recent = float(ts.get("trend_score_recent", 0.0))
        trend_score        = float(ts.get("trend_score", 0.0))
        trajectory         = float(ts.get("trajectory", 1.0))

        z = _trend_zscore(topic_id, category, trend_score_recent, category_scores)

        # partial correlation of publish_month_num vs log_vpd — is there
        # a seasonal component at all after controlling for channel size?
        rx, ry = _partial_corr_residuals(group, "publish_month_num", "log_vpd", "log_subscribers")
        seasonal_pcorr = _partial_correlation(rx, ry)

        result[str(topic_id)] = {
            "category":              category,
            "n_videos":              int(len(group)),
            "baseline_mean_log_vpd": round(baseline_mean, 4),
            "baseline_std_log_vpd":  round(baseline_std, 4),
            "trend_score":           round(trend_score, 4),
            "trend_score_recent":    round(trend_score_recent, 4),
            "trend_zscore":          z,
            "trajectory":            round(trajectory, 4),
            "seasonal_pcorr":        round(seasonal_pcorr, 4),
            "seasonality_index":     _seasonality_index(group),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    build_topic_trend_baseline()