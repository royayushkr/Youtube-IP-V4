# Topic Trend Baseline — Methodology & Implementation

---

## Overview

`topic_trend_baseline.py` derives per-topic engagement baselines and monthly seasonality indices from the training corpus and writes a static lookup JSON. It is pure computation — no model training, no inference calls.

Its purpose is to give the inference orchestrator the context needed to express trend strength relative to a topic's own historical norm. Without a baseline, the orchestrator can only tell Gemini that a topic has a raw trend score — it cannot say whether that score is high or low for that topic. With a baseline, it can say a topic is trending significantly above or below its typical level, which is a materially stronger recommendation signal.

All engagement signals controlling for channel size use partial correlations that regress out `log1p(channel_subscriberCount)` before computing correlations. See individual methodology notes below for where this applies.

---

## Data Sources

| Source | File | Key Columns Used |
|---|---|---|
| Raw video metadata | `data/processed/combined_videos_raw.csv` | `video_id`, `channel_subscriberCount` |
| Clean video metadata | `data/processed/combined_videos_clean.csv` | `video_id`, `views_per_day`, `publish_month`, `is_short` |
| BERTopic output | `data/processed/bertopic_metadata.csv` | `video_id`, `topic_id` |
| Topic stats | `data/processed/topic_stats.csv` | `topic_id`, `top_category`, `trend_score`, `trend_score_recent`, `trajectory` |

**After filtering:** long-form only (`is_short == False`), `views_per_day > 0`, `topic_id != -1`, non-null subscriber counts. 83 of 89 topics meet `_MIN_TOPIC_VIDEOS = 20` after the Shorts filter. The 6 excluded topics are omitted from the output — the inference orchestrator falls back to category-level signals for those topics.

---

## Methodology

### Engagement Baseline

For each topic, `baseline_mean_log_vpd` and `baseline_std_log_vpd` are computed as the mean and standard deviation of `log1p(views_per_day)` across all long-form corpus videos assigned to that topic. These are raw corpus statistics — not partial-correlation adjusted — because they describe the distribution of the target variable itself, not a feature-target relationship.

These values give the inference orchestrator a reference distribution for the topic's typical performance, against which live channel predictions can be contextualized.

### Trend Z-Score

`trend_zscore` expresses how strong a topic's current trend score is relative to the distribution of trend scores within its category. It is computed as:

```
z = (trend_score_recent - category_mean) / category_std
```

where `category_mean` and `category_std` are derived from `trend_score_recent` values across all topics in `topic_stats.csv` that share the same `top_category`. This is a cross-topic z-score within category, not a time-series z-score, because PyTrends data was only collected at a single point in time — there is no topic-level historical time series to standardize against.

A positive z-score means the topic is trending more strongly than typical for its niche. A negative z-score means it is below niche average. The inference orchestrator passes this to Gemini to calibrate recommendation confidence.

Categories with fewer than 3 topics return `trend_zscore: null`.

### Seasonality Index

`seasonality_index` is a per-month multiplier expressing how a topic's typical engagement compares to its own annual mean. A value of 1.2 for month 8 means videos in that topic published in August historically perform 20% above the topic's annual average.

**Method:** For each month (1–12), the mean `log1p(views_per_day)` is computed across all corpus videos in that topic published in that month. The annual mean is the mean of all occupied monthly means. Each month's index is its mean divided by the annual mean.

`_MIN_MONTH_VIDEOS = 5` — months with fewer than 5 videos are omitted from the index entirely rather than returning unstable estimates.

The seasonality index is not partial-correlation adjusted because it is a ratio of within-topic means, not a feature-target correlation. Channel size is partially controlled implicitly because the baseline is topic-specific — large and small channels within a topic both contribute to the monthly mean.

`seasonal_pcorr` is additionally provided as a partial correlation of `publish_month_num` against `log_vpd` controlling for `log_subscribers`. This indicates whether there is a statistically meaningful seasonal component to a topic's performance at all — the inference orchestrator can use this to decide whether to surface seasonality as a signal to Gemini.

---

## Output Schema

`topic_trend_baseline.json` contains one entry per topic keyed by integer topic ID string:

```json
{
  "24": {
    "category": "tech",
    "n_videos": 312,
    "baseline_mean_log_vpd": 7.1204,
    "baseline_std_log_vpd": 1.9831,
    "trend_score": 77.85,
    "trend_score_recent": 85.03,
    "trend_zscore": 1.243,
    "trajectory": 1.0923,
    "seasonal_pcorr": 0.081,
    "seasonality_index": {
      "1": 0.94,
      "2": 0.97,
      "3": 1.08,
      "8": 1.21,
      "9": 1.15
    }
  }
}
```

| Field | Description |
|---|---|
| `category` | Dominant niche category from `topic_stats.csv` |
| `n_videos` | Long-form corpus videos contributing to estimates |
| `baseline_mean_log_vpd` | Mean `log1p(views_per_day)` across topic corpus |
| `baseline_std_log_vpd` | Std of `log1p(views_per_day)` across topic corpus |
| `trend_score` | 90-day mean Google search interest from PyTrends |
| `trend_score_recent` | 30-day mean Google search interest from PyTrends |
| `trend_zscore` | Z-score of `trend_score_recent` within category distribution; `null` if fewer than 3 category topics |
| `trajectory` | `trend_score_recent / trend_score` from `topic_stats.csv` |
| `seasonal_pcorr` | Partial correlation of publish month with `log_vpd` controlling for `log_subscribers` |
| `seasonality_index` | Per-month performance multiplier relative to topic annual mean; omits months with fewer than 5 videos |

---

## Implementation

**File:** `src/modeling/topic_trend_baseline.py`

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `_MIN_TOPIC_VIDEOS` | 20 | Minimum long-form videos for a reliable topic baseline |
| `_MIN_MONTH_VIDEOS` | 5 | Minimum videos per month for a reliable seasonality entry |

**Public API:**

```python
build_topic_trend_baseline(
    raw_csv: str | Path     = "data/processed/combined_videos_raw.csv",
    clean_csv: str | Path   = "data/processed/combined_videos_clean.csv",
    meta_csv: str | Path    = "data/processed/bertopic_metadata.csv",
    topic_stats: str | Path = "data/processed/topic_stats.csv",
    output_path: str | Path = "outputs/models/topic_trend_baseline.json",
) -> dict
```

---

## Outputs

| File | Location | Description |
|---|---|---|
| `topic_trend_baseline.json` | `outputs/models/` | Per-topic engagement baseline, trend z-score, and seasonality index |

No model artifacts produced. Regenerated nightly in the Cloud Run retraining job alongside XGBoost and BERTopic.