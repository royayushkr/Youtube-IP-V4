# PyTrends Trend Scorer — Methodology & Implementation

---

## Overview

The PyTrends client scores each BERTopic topic cluster against Google Trends data to produce two signals: a `trend_score` (current search interest, 0–100) and a `trajectory` (momentum ratio of recent vs. current interest). These feed into two downstream consumers: `trend_score` is added as a feature in the XGBoost engagement predictor, and both signals are passed to Gemini for conversational recommendation reasoning.

---

## Input

**Source:** `data/processed/topic_stats.csv` — 89 topics output by BERTopic, each with a `topic_label`, `top_category`, and `dominant_category_share`.

**Filtering:** Topics with `dominant_category_share < 0.7` are excluded before querying. These are cross-category blends where the topic label does not cleanly represent a single niche, making PyTrends queries unreliable. This filter reduces the query set from 89 to the coherent subset.

---

## Query Construction

Each topic label follows the format `{topic_id}_{kw1}_{kw2}_{kw3}_{kw4}`, e.g.:

```
49_nuclear fusion_nuclear_fusion_science news
42_quantum physics_quantum_physics_quantum_quantum mechanics
76_season episode_episode_episode check_finale
```

### Parsing Pipeline

1. Split on `_`, drop the leading numeric ID
2. Flatten chunks into individual words
3. Deduplicate at the word level while preserving order
4. Rejoin into a single query string

This converts `"nuclear fusion nuclear fusion science news"` → `"nuclear fusion science news"` and `"season episode episode check finale"` → `"season episode check finale"`.

### Fallback Strategy

If the full query returns a zero score from PyTrends, the client retries with only the first keyword token (e.g. `"nuclear"` for a failed `"nuclear fusion science news"` query). This handles cases where PyTrends has no data for the full multi-word string but recognizes the primary term. Zero scores after fallback are treated as genuine no-signal topics.

---

## Timeframes & Scoring

Two timeframes are queried per topic independently:

| Key | Timeframe | Purpose |
|---|---|---|
| `current` | `today 3-m` | 90-day mean interest -> `trend_score` for XGBoost |
| `recent` | `today 1-m` | 30-day mean interest -> trajectory numerator |

Each query uses `geo="US"` and is issued as a single-keyword payload (no batching). Single-keyword queries avoid cross-keyword normalization artifacts that arise when PyTrends scales all keywords in a batch relative to the highest-volume term.

### Rate Limiting

A randomized sleep of `1.2–2.0` seconds is applied after every API call. At 89 topics × 2 timeframes, with fallback retries on zero-score topics, total runtime is approximately 5–8 minutes. This runs nightly in the Cloud Run retraining job, not in the user session.

---

## Trajectory Computation

```
trajectory = recent_score / current_score
```

**Constraints applied:**

| Condition | Trajectory assigned |
|---|---|
| `current_score < 3.0` | `1.0` (insufficient signal) |
| `recent_score == 0.0` | `1.0` (insufficient signal) |
| Raw ratio `> 5.0` | Capped at `5.0` (noise ceiling) |

The `< 3.0` floor on `current_score` suppresses a PyTrends artifact where sparse low-volume queries consistently return `(current=1.10, recent=3.45)`, producing a spurious ratio of `3.1379`. These are not real trend signals. The `5.0` cap prevents fallback queries that hit an unrelated trending topic from producing extreme ratios.

**Interpretation at the Gemini layer:**

| Trajectory | Label |
|---|---|
| `>= 1.2` | Rising |
| `<= 0.8` | Declining |
| Between | Stable |

---

## Outputs

`score_topics()` returns a DataFrame with one row per scored topic:

| Column | Description |
|---|---|
| `topic_id` | BERTopic topic ID |
| `topic_label` | Raw BERTopic label string |
| `top_category` | Dominant niche category |
| `pytrends_query` | Deduplicated query string sent to PyTrends |
| `trend_score` | 90-day mean interest (0–100) → XGBoost feature |
| `trend_score_recent` | 30-day mean interest (0–100) → trajectory numerator |
| `trajectory` | Momentum ratio, capped and floored as above |

`get_trend_score_map()` returns `{topic_id: trend_score}` — a direct lookup dict for XGBoost feature injection at inference time.

---

## Sample Results (March 11, 2026)

| topic_id | pytrends_query | trend_score | trend_score_recent | trajectory |
|---|---|---|---|---|
| 20 | dark matter black holes space physics time | 81.23 | 82.59 | 1.02 |
| 51 | glasses lenses optics lens | 77.14 | 86.62 | 1.12 |
| 57 | soundtrack song playlist babies | 73.01 | 68.55 | 0.94 |
| 4 | yoga routine exercise stretching | 71.44 | 73.66 | 1.03 |
| 35 | disturbing media internet horror | 71.57 | 76.14 | 1.06 |
| 65 | meals chicken meal korean | 70.51 | 84.00 | 1.19 |
| 24 | security hackers hacking surveillance | 70.26 | 75.66 | 1.08 |
| 34 | beef meat burger cook | 69.10 | 79.10 | 1.14 |
| 21 | horror game haunted halloween ghost | 65.26 | 66.52 | 1.02 |
| 36 | voice singer singing | 7.09 | 41.48 | 5.00 |
| 30 | harry potter disney | 54.20 | 79.24 | 1.46 |
| 3 | fortnite battle royale aim | 31.74 | 80.17 | 2.53 |
| 49 | nuclear fusion science news | 52.19 | 65.59 | 1.26 |
| 42 | quantum physics mechanics | 13.57 | 29.07 | 2.14 |

**Zero-signal topics (persistent zeros after fallback):**

Topics 67, 88, 16, 72, 18, 69, and 76 returned zero across both timeframes and fallback. These are personality-driven or platform-specific clusters (`uncle roger`, `twitch streaming`, `claire anne anna kelly`) where search interest is not captured by Google Trends. These topics receive `trend_score=0.0` and `trajectory=1.0` in the XGBoost feature vector.

---

## Implementation

**File:** `src/data_collection/pytrends_client.py`

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `GEO` | `"US"` | Geographic scope |
| `_MIN_CATEGORY_SHARE` | `0.7` | Minimum topic coherence for querying |
| `_SLEEP_BASE` | `1.2s` | Base delay between API calls |
| `_SLEEP_JITTER` | `0–0.8s` | Randomized jitter to avoid rate limiting |
| `_MAX_TRAJECTORY` | `5.0` | Trajectory noise ceiling |
| `_MIN_SCORE_FOR_TRAJECTORY` | `3.0` | Minimum current score to compute trajectory |