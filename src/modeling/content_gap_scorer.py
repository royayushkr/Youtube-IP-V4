import json
import numpy as np
import pandas as pd
from pathlib import Path

TOPIC_PATH      = Path("data/processed/topic_stats.csv")
ENGAGEMENT_PATH = Path("outputs/models/topic_engagement_stats.json")

MIN_CATEGORY_SHARE = 0.7
MIN_TOPIC_VIDEOS   = 30

# component weights for gap score
W_REACH      = 0.35
W_TREND      = 0.30
W_TRAJECTORY = 0.20
W_ENGAGEMENT = 0.15


def _trajectory_score(trajectory: float) -> float:
    # map trajectory ratio to 0-1: 1.0 = neutral, >1 rising, <1 declining
    t = min(trajectory, 3.0)
    return float(np.clip((t - 0.5) / 2.5, 0.0, 1.0))


def _minmax(series: pd.Series) -> pd.Series:
    rng = series.max() - series.min()
    return (series - series.min()) / rng if rng > 0 else pd.Series(0.5, index=series.index)


def score_gaps(
    covered_topic_ids: list[int],
    channel_category: str,
    topic_stats_path: str = str(TOPIC_PATH),
    engagement_path: str = str(ENGAGEMENT_PATH),
    top_n: int = 10,
) -> list[dict]:
    """
    ranks uncovered topic opportunities for a live creator channel.

    args:
        covered_topic_ids: topic ids assigned to the creator's videos via
                           bertopic_model.transform() at inference time.
                           does not require the channel to exist in the
                           training corpus.
        channel_category:  primary niche category — one of:
                           entertainment, fitness, food, gaming,
                           research_science, tech
        top_n:             number of gap opportunities to return

    returns:
        list of dicts sorted by gap_score descending, ready for gemini
        recommendation layer consumption
    """
    stats = pd.read_csv(topic_stats_path)
    stats = stats[
        (stats["topic_id"] != -1) &
        (stats["dominant_category_share"] >= MIN_CATEGORY_SHARE) &
        (stats["video_count"] >= MIN_TOPIC_VIDEOS)
    ].copy()

    with open(engagement_path) as f:
        eng_data = json.load(f)
    engagement   = eng_data["topics"]
    cat_fallback = eng_data["category_fallbacks"].get(
        channel_category, {"engagement_percentile": 0.5}
    )

    covered    = set(covered_topic_ids) | {-1}
    all_cat    = stats[stats["top_category"] == channel_category].copy()
    candidates = all_cat[~all_cat["topic_id"].isin(covered)].copy()

    if candidates.empty:
        return []

    reach_map = dict(zip(all_cat["topic_id"], _minmax(all_cat["median_views"])))
    trend_map = dict(zip(all_cat["topic_id"], _minmax(all_cat["trend_score"])))

    results = []
    for _, row in candidates.iterrows():
        tid = int(row["topic_id"])

        reach_score = float(reach_map.get(tid, 0.5))
        trend_score = float(trend_map.get(tid, 0.5))
        traj_score  = _trajectory_score(float(row["trajectory"]))
        eng_pct     = float(
            engagement.get(str(tid), {}).get(
                "engagement_percentile", cat_fallback["engagement_percentile"]
            )
        )

        gap_score = (
            W_REACH      * reach_score +
            W_TREND      * trend_score +
            W_TRAJECTORY * traj_score +
            W_ENGAGEMENT * eng_pct
        )

        trajectory_label = (
            "rising"    if row["trajectory"] >= 1.2 else
            "declining" if row["trajectory"] <= 0.8 else
            "stable"
        )

        results.append({
            "topic_id":         tid,
            "topic_label":      row["topic_label"],
            "gap_score":        round(gap_score, 4),
            "reach_score":      round(reach_score, 4),
            "trend_score":      round(trend_score, 4),
            "trajectory_score": round(traj_score, 4),
            "engagement_score": round(eng_pct, 4),
            "median_views":     int(row["median_views"]),
            "trend_raw":        round(float(row["trend_score"]), 2),
            "trend_recent":     round(float(row["trend_score_recent"]), 2),
            "trajectory":       round(float(row["trajectory"]), 4),
            "trajectory_label": trajectory_label,
            "median_like_rate": round(float(row["median_like_rate"]), 6),
            "video_count":      int(row["video_count"]),
            "category":         row["top_category"],
        })

    results.sort(key=lambda x: x["gap_score"], reverse=True)
    return results[:top_n]


if __name__ == "__main__":
    # diagnostic using training corpus channels to validate output shape.
    # at production inference, covered_topic_ids comes from
    # bertopic_model.transform() on the creator's live video titles/descriptions.
    meta = pd.read_csv("data/processed/bertopic_metadata.csv")

    for channel in ["SciShow", "Tasty", "Linus Tech Tips"]:
        ch = meta[meta["channel_title"] == channel]
        if ch.empty:
            continue
        covered  = ch["topic_id"].dropna().astype(int).unique().tolist()
        category = ch["category_name"].mode()[0]
        gaps     = score_gaps(covered_topic_ids=covered, channel_category=category, top_n=5)
        print(f"\n{channel} — top 5 content gaps:")
        for g in gaps:
            print(
                f"  [{g['gap_score']:.3f}] {g['topic_label']:<45} "
                f"views={g['median_views']:>8,}  trend={g['trend_raw']:>5.1f}  "
                f"{g['trajectory_label']}"
            )