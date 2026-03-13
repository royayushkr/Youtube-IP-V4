# Content Gap Scorer — Methodology & Implementation

---

## Overview

`content_gap_scorer.py` identifies topic opportunities a creator hasn't covered yet, ranked by their predicted value. It is pure computation — no model training, no inference calls. It combines four signals already produced by earlier pipeline components into a single weighted score per uncovered topic.

The scorer works for any creator regardless of whether they exist in the training corpus. Covered topic IDs are derived at inference time by running the creator's live video titles/descriptions through `bertopic_model.transform()` — not from any training data file.

---

## Input Requirements

| Input | Source at inference |
|---|---|
| `covered_topic_ids` | `bertopic_model.transform()` on creator's live channel data |
| `channel_category` | Determined from creator's channel (entertainment, fitness, food, gaming, research_science, tech) |
| `topic_stats.csv` | Static corpus file — per-topic reach and trend signals |
| `topic_engagement_stats.json` | Static model output — per-topic engagement percentiles |

---

## Candidate Topic Filtering

Before scoring, topics are filtered to a clean candidate pool:

| Filter | Value | Reason |
|---|---|---|
| `topic_id != -1` | Always excluded | BERTopic noise bin — not a real topic |
| `dominant_category_share >= 0.70` | Minimum | Topic must clearly belong to creator's niche |
| `video_count >= 30` | Minimum | Enough corpus data for reliable statistics |
| Already covered | Excluded | Topics the creator has already made videos on |

Topics that pass all filters but have no entry in `topic_engagement_stats.json` fall back to their category median engagement percentile from the `category_fallbacks` key.

**Scoreable topics after filters:**

| Category | Scoreable topics |
|---|---|
| entertainment | 18 |
| research_science | 14 |
| tech | 12 |
| gaming | 9 |
| food | 9 |
| fitness | 5 |

---

## Gap Score Formula

```
gap_score = 0.35 × reach_score
          + 0.30 × trend_score
          + 0.20 × trajectory_score
          + 0.15 × engagement_score
```

All four components are normalized to [0, 1] before weighting. Normalization is within-category — scores are relative to other topics in the same niche, not globally.

### Component Definitions

**reach_score (0.35)**
Min-max normalized `median_views` within the creator's category. Represents the typical view ceiling for videos in this topic cluster based on the training corpus. Higher = more reach potential.

**trend_score (0.30)**
Min-max normalized 90-day PyTrends `trend_score` within category. Represents current Google search interest for the topic. Higher = more people are actively searching for this content right now.

**trajectory_score (0.20)**
Derived from `trajectory = trend_score_recent / trend_score` (30-day vs 90-day ratio):

```python
trajectory_score = clip((trajectory - 0.5) / 2.5, 0.0, 1.0)
```

| Trajectory value | Label | Score |
|---|---|---|
| ≥ 1.2 | rising | > 0.28 |
| 0.8–1.2 | stable | ~0.20 |
| ≤ 0.8 | declining | < 0.12 |

Caps at trajectory = 3.0 (score = 1.0). Rewards topics gaining momentum, penalizes topics losing interest.

**engagement_score (0.15)**
`engagement_percentile` from `topic_engagement_stats.json` — within-category rank of the topic's median `like_rate` across the training corpus. Higher = audiences in this topic cluster interact more actively with content. Falls back to category median (0.5) for topics with fewer than 20 corpus videos.

### Weight Rationale

Reach and trend together account for 65% of the score — the primary question is whether a topic can drive views and whether people are actively searching for it now. Trajectory adds a forward-looking adjustment: two topics with equal trend scores rank differently if one is accelerating. Engagement is a secondary quality signal — valuable but not worth sacrificing reach or timing for.

---

## Output Schema

Each entry in the returned list:

| Field | Type | Description |
|---|---|---|
| `topic_id` | int | BERTopic topic ID |
| `topic_label` | str | 4-word BERTopic label (raw — Gemini translates to plain English) |
| `gap_score` | float | Weighted composite score (0–1) |
| `reach_score` | float | Normalized reach component (0–1) |
| `trend_score` | float | Normalized trend component (0–1) |
| `trajectory_score` | float | Normalized trajectory component (0–1) |
| `engagement_score` | float | Engagement percentile component (0–1) |
| `median_views` | int | Median views across corpus videos in this topic |
| `trend_raw` | float | Raw 90-day PyTrends score (0–100) |
| `trend_recent` | float | Raw 30-day PyTrends score (0–100) |
| `trajectory` | float | trend_recent / trend_score ratio |
| `trajectory_label` | str | "rising", "stable", or "declining" |
| `median_like_rate` | float | Median like_rate across corpus videos in this topic |
| `video_count` | int | Videos in training corpus assigned to this topic |
| `category` | str | Dominant niche category |

Results are sorted by `gap_score` descending. `top_n` controls how many are returned (default 10).

---

## Validation Results

Diagnostic run against three training corpus channels to validate output shape. At production inference, `covered_topic_ids` comes from `bertopic_model.transform()` — these channels are used here only because their topic assignments are already known.

```
SciShow — top 5 content gaps:
  [0.300] 29_statquest videos_statistics_statquest_statistical
          views=153,146  trend=3.8  rising

Tasty — top 5 content gaps:
  [0.452] 34_beef_meat_burger_cook
          views=201,945  trend=68.9  stable
  [0.403] 72_uncle_roger_dish_restaurant
          views=2,033,996  trend=1.1  stable
  [0.403] 60_bread_baking_cookbook_recipes
          views=17,954  trend=63.9  stable
  [0.373] 74_cheese_beginners_supplies_recipe
          views=945  trend=52.8  rising
  [0.371] 65_meals_chicken_meal_korean
          views=73,689  trend=69.0  stable

Linus Tech Tips — top 5 content gaps:
  [0.513] 78_native_indian_crash course_crash
          views=55,175  trend=67.3  rising
  [0.503] 50_radio_signal_spectrum_radar
          views=15,182  trend=88.3  stable
  [0.389] 71_developers_developer_programming_python
          views=18,091  trend=60.0  stable
  [0.211] 52_coding_code_programming_web
          views=9,261  trend=46.5  stable
```

**Notes on output:**

SciShow returns only 1 gap because it covers 27 topics already — unusually broad for a single channel. A typical creator with 50–100 videos will have substantially more gap to score against.

Topic 78 for Linus (`native_indian_crash course_crash`) has a noisy BERTopic label but is a legitimate tech cluster (`dominant_category_share` = 0.81, 43 videos) — likely coding crash courses and native app development. The label is a BERTopic artifact. The Gemini recommendation layer translates raw topic labels to plain English before they reach the creator.

---

## Implementation

**File:** `src/modeling/content_gap_scorer.py`

**Dependencies:** `numpy`, `pandas`, `json` — no model loading, no external API calls.

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `MIN_CATEGORY_SHARE` | 0.70 | Minimum dominant_category_share for topic inclusion |
| `MIN_TOPIC_VIDEOS` | 30 | Minimum corpus video count for topic inclusion |
| `W_REACH` | 0.35 | Reach component weight |
| `W_TREND` | 0.30 | Trend component weight |
| `W_TRAJECTORY` | 0.20 | Trajectory component weight |
| `W_ENGAGEMENT` | 0.15 | Engagement component weight |

**Public API:**

```python
score_gaps(
    covered_topic_ids: list[int],
    channel_category: str,
    topic_stats_path: str = "data/processed/topic_stats.csv",
    engagement_path: str = "outputs/models/topic_engagement_stats.json",
    top_n: int = 10,
) -> list[dict]
```

---

## Outputs

| File | Location | Description |
|---|---|---|
| `content_gap_scorer.py` | `src/modeling/` | Scorer implementation |

No model artifacts produced — scorer reads from `topic_stats.csv` and `topic_engagement_stats.json` at call time. Both files are static between nightly retraining runs.