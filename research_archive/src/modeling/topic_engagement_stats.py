import json
import numpy as np
import pandas as pd
from pathlib import Path

CLEAN_PATH = Path("data/processed/combined_videos_clean.csv")
TOPIC_PATH = Path("data/processed/bertopic_metadata.csv")
STATS_PATH = Path("data/processed/topic_stats.csv")
OUT_PATH   = Path("outputs/models/topic_engagement_stats.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_VIDEOS = 20


def compute() -> dict:
    df    = pd.read_csv(CLEAN_PATH)
    meta  = pd.read_csv(TOPIC_PATH, usecols=["video_id", "topic_id"])
    topics = pd.read_csv(STATS_PATH, usecols=["topic_id", "trend_score", "top_category"])

    df = df.merge(meta, on="video_id", how="inner")
    df = df.merge(topics, on="topic_id", how="left")
    df = df[df["like_rate"] > 0].dropna(subset=["like_rate", "category_name"])
    df = df[~df["is_short"]].reset_index(drop=True)

    category_stats = (
        df.groupby("category_name")["like_rate"]
        .agg(cat_median="median", cat_mean="mean", cat_std="std")
        .reset_index()
    )

    topic_stats = (
        df.groupby(["topic_id", "category_name"])["like_rate"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            p75=lambda x: np.percentile(x, 75),
            count="count",
        )
        .reset_index()
    )

    topic_stats = topic_stats.merge(category_stats, on="category_name", how="left")
    topic_stats = topic_stats.merge(
        topics[["topic_id", "trend_score", "top_category"]], on="topic_id", how="left"
    )

    low_count = topic_stats["count"] < MIN_VIDEOS
    topic_stats.loc[low_count, "median"] = topic_stats.loc[low_count, "cat_median"]
    topic_stats.loc[low_count, "mean"]   = topic_stats.loc[low_count, "cat_mean"]
    topic_stats["reliable"] = (~low_count).astype(int)

    topic_stats["engagement_percentile"] = topic_stats.groupby("category_name")["median"].rank(pct=True).round(4)

    def minmax(s):
        rng = s.max() - s.min()
        return ((s - s.min()) / rng).round(4) if rng > 0 else pd.Series(0.5, index=s.index)

    topic_stats["engagement_score"]  = topic_stats.groupby("category_name")["median"].transform(minmax)
    topic_stats["global_percentile"] = topic_stats["median"].rank(pct=True).round(4)

    output = {}
    for _, row in topic_stats.iterrows():
        output[int(row["topic_id"])] = {
            "category":              row["category_name"],
            "mean_like_rate":        round(float(row["mean"]), 6),
            "median_like_rate":      round(float(row["median"]), 6),
            "p75_like_rate":         round(float(row["p75"]), 6),
            "std_like_rate":         round(float(row["std"]), 6),
            "n_videos":              int(row["count"]),
            "reliable":              bool(row["reliable"]),
            "engagement_percentile": float(row["engagement_percentile"]),
            "engagement_score":      float(row["engagement_score"]),
            "global_percentile":     float(row["global_percentile"]),
            "trend_score":           float(row["trend_score"]) if pd.notna(row["trend_score"]) else 0.0,
        }

    cat_fallbacks = {}
    for _, row in category_stats.iterrows():
        cat_fallbacks[row["category_name"]] = {
            "median_like_rate":      round(float(row["cat_median"]), 6),
            "engagement_percentile": 0.5,
            "engagement_score":      0.5,
            "global_percentile":     0.5,
        }

    result = {"topics": output, "category_fallbacks": cat_fallbacks}

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Topics scored: {len(output)}")
    print(f"Reliable (>= {MIN_VIDEOS} videos): {sum(v['reliable'] for v in output.values())}")
    print()

    ranked = sorted(output.items(), key=lambda x: x[1]["engagement_percentile"], reverse=True)
    print("Top 10 topics by engagement percentile:")
    for tid, v in ranked[:10]:
        print(f"  topic {tid:>3}  [{v['category']:<18}]  median={v['median_like_rate']:.4f}  pct={v['engagement_percentile']:.2f}  n={v['n_videos']}")
    print()
    print("Bottom 10 topics by engagement percentile:")
    for tid, v in ranked[-10:]:
        print(f"  topic {tid:>3}  [{v['category']:<18}]  median={v['median_like_rate']:.4f}  pct={v['engagement_percentile']:.2f}  n={v['n_videos']}")
    print()
    print(f"Saved -> {OUT_PATH}")

    return result


if __name__ == "__main__":
    compute()