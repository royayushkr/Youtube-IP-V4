import time
import random
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError


GEO = "US"
_TIMEFRAMES = {"current": "today 3-m", "recent": "today 1-m"}
_MIN_CATEGORY_SHARE = 0.7
_SLEEP_BASE = 1.2
_SLEEP_JITTER = 0.8
_MAX_TRAJECTORY = 5.0
_MIN_SCORE_FOR_TRAJECTORY = 3.0
_RETRY_WAIT = 60


def _parse_query(topic_label: str) -> str:
    parts = topic_label.split("_")
    chunks = parts[1:] if parts[0].isdigit() else parts
    seen = []
    seen_words = set()
    for chunk in chunks:
        words = chunk.split()
        deduped = [w for w in words if w not in seen_words]
        if deduped:
            seen.append(" ".join(deduped))
            seen_words.update(deduped)
    return " ".join(seen)


def _sleep():
    time.sleep(_SLEEP_BASE + random.uniform(0, _SLEEP_JITTER))


def _query_single(pt: TrendReq, query: str, timeframe: str) -> float:
    try:
        pt.build_payload([query], geo=GEO, timeframe=timeframe)
        df = pt.interest_over_time()
        if df.empty or query not in df.columns:
            return 0.0
        return float(df[query].mean())
    except TooManyRequestsError:
        print(f"429 on '{query}' — waiting {_RETRY_WAIT}s")
        time.sleep(_RETRY_WAIT)
        try:
            pt.build_payload([query], geo=GEO, timeframe=timeframe)
            df = pt.interest_over_time()
            if df.empty or query not in df.columns:
                return 0.0
            return float(df[query].mean())
        except Exception:
            return 0.0
    except Exception:
        return 0.0


def _query_topic(pt: TrendReq, query: str) -> dict:
    scores = {}
    for key, tf in _TIMEFRAMES.items():
        score = _query_single(pt, query, tf)
        _sleep()

        if score == 0.0:
            fallback = query.split()[0]
            if fallback != query:
                score = _query_single(pt, fallback, tf)
                _sleep()

        scores[key] = score
    return scores


def _compute_trajectory(current_score: float, recent_score: float) -> float:
    if current_score < _MIN_SCORE_FOR_TRAJECTORY or recent_score == 0.0:
        return 1.0
    raw = recent_score / current_score
    return round(min(raw, _MAX_TRAJECTORY), 4)


def score_topics(topic_stats_path: str) -> pd.DataFrame:
    df = pd.read_csv(topic_stats_path)
    df = df[df["dominant_category_share"] >= _MIN_CATEGORY_SHARE].copy()

    pt = TrendReq(hl="en-US", tz=360)
    results = []

    for _, row in df.iterrows():
        query = _parse_query(row["topic_label"])
        scores = _query_topic(pt, query)

        current = scores["current"]
        recent = scores["recent"]
        trajectory = _compute_trajectory(current, recent)

        results.append({
            "topic_id": row["topic_id"],
            "topic_label": row["topic_label"],
            "top_category": row["top_category"],
            "pytrends_query": query,
            "trend_score": round(current, 2),
            "trend_score_recent": round(recent, 2),
            "trajectory": trajectory,
        })

    return pd.DataFrame(results)


def get_trend_score_map(topic_stats_path: str) -> dict:
    df = score_topics(topic_stats_path)
    return dict(zip(df["topic_id"], df["trend_score"]))


def update_topic_stats(topic_stats_path: str) -> None:
    stats = pd.read_csv(topic_stats_path)
    scored = score_topics(topic_stats_path)[["topic_id", "trend_score", "trend_score_recent", "trajectory"]]
    for col in ["trend_score", "trend_score_recent", "trajectory"]:
        if col in stats.columns:
            stats = stats.drop(columns=[col])
    stats = stats.merge(scored, on="topic_id", how="left")
    stats["trend_score"] = stats["trend_score"].fillna(0.0)
    stats["trend_score_recent"] = stats["trend_score_recent"].fillna(0.0)
    stats["trajectory"] = stats["trajectory"].fillna(1.0)
    stats.to_csv(topic_stats_path, index=False)
    print(f"Updated {topic_stats_path} with trend scores for {len(scored)} topics")


if __name__ == "__main__":
    import os

    path = os.path.join(
        os.path.dirname(__file__), "../../data/processed/topic_stats.csv"
    )
    update_topic_stats(path)
    df = pd.read_csv(path)
    print(df[["topic_id", "trend_score", "trend_score_recent", "trajectory"]].to_string())