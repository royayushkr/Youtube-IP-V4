"""
title_effectiveness_stats.py — derives per-topic title pattern signals and
optimal duration ranges from the training corpus and writes a static lookup JSON.

all correlations are partial correlations controlling for log_subscribers to
remove channel-size confounding.

run once during training alongside the other stat-derivation scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_RAW_CSV   = Path("data/processed/combined_videos_raw.csv")
_CLEAN_CSV = Path("data/processed/combined_videos_clean.csv")
_META_CSV  = Path("data/processed/bertopic_metadata.csv")
_OUT_PATH  = Path("outputs/models/title_effectiveness_stats.json")

_MIN_TOPIC_VIDEOS = 20
_POSITIVE_KW = {"best", "amazing", "incredible", "winning", "success", "grow", "top", "ultimate"}
_NEGATIVE_KW = {"worst", "fail", "broke", "lost", "mistake", "wrong", "never"}


def _title_features(title: str) -> dict:
    words = title.split()
    word_set = set(w.lower() for w in words)
    sentiment = len(word_set & _POSITIVE_KW) - len(word_set & _NEGATIVE_KW)
    return {
        "wordcount":     len(words),
        "has_number":    int(any(c.isdigit() for c in title)),
        "has_question":  int("?" in title),
        "has_brackets":  int("[" in title or "(" in title),
        "has_caps_word": int(any(w.isupper() and len(w) > 1 for w in words)),
        "sentiment":     max(-2, min(2, sentiment)),
        "char_length":   len(title),
    }


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


def _binary_stats(group: pd.DataFrame, feat: str, high_mask: pd.Series) -> dict | None:
    sub = group[[feat, "log_vpd", "log_subscribers"]].copy()
    sub = sub[np.isfinite(sub).all(axis=1)].reset_index(drop=True)
    if len(sub) < 2 or sub[feat].std() == 0:
        return None
    n1 = sub[feat].sum()
    n0 = len(sub) - n1
    if n1 == 0 or n0 == 0:
        return None
    high = high_mask.reindex(range(len(sub))).fillna(False)
    rx, ry = _partial_corr_residuals(sub, feat, "log_vpd", "log_subscribers")
    pcorr  = _partial_correlation(rx, ry)
    return {
        "partial_correlation": round(pcorr, 4),
        "rate_in_top":         round(float(sub.loc[high,  feat].mean()), 4) if high.any() else 0.0,
        "rate_in_rest":        round(float(sub.loc[~high, feat].mean()), 4) if (~high).any() else 0.0,
        "recommend":           bool(pcorr > 0.02),
    }


def _continuous_stats(group: pd.DataFrame, feat: str, high_mask: pd.Series) -> dict:
    sub = group[[feat, "log_vpd", "log_subscribers"]].copy()
    sub = sub[np.isfinite(sub).all(axis=1)].reset_index(drop=True)
    high = high_mask.reindex(range(len(sub))).fillna(False)
    rx, ry = _partial_corr_residuals(sub, feat, "log_vpd", "log_subscribers")
    pcorr  = _partial_correlation(rx, ry)
    return {
        "partial_correlation": round(pcorr, 4),
        "mean_in_top":         round(float(sub.loc[high,  feat].mean()), 2) if high.any() else 0.0,
        "mean_in_rest":        round(float(sub.loc[~high, feat].mean()), 2) if (~high).any() else 0.0,
        "optimal_range":       _optimal_range(sub[feat], sub["log_vpd"]),
    }


def _optimal_range(feature: pd.Series, target: pd.Series) -> dict:
    try:
        df = pd.DataFrame({"f": feature, "t": target}).dropna()
        df["quartile"] = pd.qcut(df["f"], q=4, duplicates="drop")
        best = df.groupby("quartile", observed=True)["t"].mean().idxmax()
        return {
            "best_quartile_low":  round(float(best.left), 2),
            "best_quartile_high": round(float(best.right), 2),
        }
    except Exception:
        return {}


def _duration_stats(group: pd.DataFrame) -> dict:
    sub = group[group["duration_sec"] > 0][
        ["duration_sec", "log_vpd", "log_subscribers"]
    ].copy().reset_index(drop=True)
    sub = sub[np.isfinite(sub).all(axis=1)].reset_index(drop=True)
    if len(sub) < 2:
        return {}
    high_mask = sub["log_vpd"] >= sub["log_vpd"].median()
    rx, ry    = _partial_corr_residuals(sub, "duration_sec", "log_vpd", "log_subscribers")
    pcorr     = _partial_correlation(rx, ry)
    opt       = _optimal_range(sub["duration_sec"], sub["log_vpd"])
    result    = {
        "partial_correlation": round(pcorr, 4),
        "mean_in_top":         round(float(sub.loc[high_mask,  "duration_sec"].mean()), 2),
        "mean_in_rest":        round(float(sub.loc[~high_mask, "duration_sec"].mean()), 2),
        "mean_in_top_min":     round(float(sub.loc[high_mask,  "duration_sec"].mean()) / 60, 1),
        "mean_in_rest_min":    round(float(sub.loc[~high_mask, "duration_sec"].mean()) / 60, 1),
        "optimal_range":       opt,
    }
    if opt:
        result["optimal_range_min"] = {
            "best_quartile_low":  round(opt["best_quartile_low"]  / 60, 1),
            "best_quartile_high": round(opt["best_quartile_high"] / 60, 1),
        }
    return result


def _compute_entry(group: pd.DataFrame, binary_features: list,
                   continuous_features: list) -> dict:
    group      = group.reset_index(drop=True)
    median_vpd = float(group["log_vpd"].median())
    high       = group["log_vpd"] >= median_vpd
    features: dict = {}
    for feat in binary_features:
        stats = _binary_stats(group, feat, high)
        if stats is not None:
            features[feat] = stats
    for feat in continuous_features:
        features[feat] = _continuous_stats(group, feat, high)
    return features


def build_title_effectiveness_stats(
    raw_csv: str | Path = _RAW_CSV,
    clean_csv: str | Path = _CLEAN_CSV,
    meta_csv: str | Path = _META_CSV,
    output_path: str | Path = _OUT_PATH,
) -> dict:
    raw   = pd.read_csv(raw_csv, usecols=["video_id", "video_title", "channel_subscriberCount"])
    clean = pd.read_csv(clean_csv, usecols=["video_id", "views_per_day", "category_name", "is_short"])
    meta  = pd.read_csv(meta_csv, usecols=["video_id", "topic_id", "duration_sec"])

    df = clean.merge(raw, on="video_id", how="inner")
    df = df.merge(meta, on="video_id", how="inner")
    df = df[
        (df["views_per_day"] > 0) &
        (df["is_short"] == False) &
        (df["topic_id"] != -1)
    ].dropna(subset=["video_title", "views_per_day", "channel_subscriberCount"])

    feats = df["video_title"].apply(_title_features).apply(pd.Series)
    df    = pd.concat([df.reset_index(drop=True), feats], axis=1)
    df["log_vpd"]         = np.log1p(df["views_per_day"])
    df["log_subscribers"] = np.log1p(df["channel_subscriberCount"])
    df = df[
        np.isfinite(df["log_vpd"]) & np.isfinite(df["log_subscribers"])
    ].reset_index(drop=True)
    df["topic_id"] = df["topic_id"].astype(int)

    binary_features     = ["has_number", "has_question", "has_brackets", "has_caps_word"]
    continuous_features = ["wordcount", "char_length", "sentiment"]

    result: dict = {}
    category_fallbacks: dict = {}

    for cat in df["category_name"].dropna().unique():
        sub = df[df["category_name"] == cat]
        if len(sub) < _MIN_TOPIC_VIDEOS:
            continue
        category_fallbacks[cat] = {
            "n_videos": int(len(sub)),
            "features": _compute_entry(sub, binary_features, continuous_features),
            "duration": _duration_stats(sub),
        }

    for topic_id, group in df.groupby("topic_id"):
        if len(group) < _MIN_TOPIC_VIDEOS:
            continue
        cat = group["category_name"].mode().iloc[0]
        result[str(int(topic_id))] = {
            "category": cat,
            "n_videos": int(len(group)),
            "reliable": len(group) >= _MIN_TOPIC_VIDEOS,
            "features": _compute_entry(group, binary_features, continuous_features),
            "duration": _duration_stats(group),
        }

    output = {"topics": result, "category_fallbacks": category_fallbacks}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    return output


if __name__ == "__main__":
    build_title_effectiveness_stats()