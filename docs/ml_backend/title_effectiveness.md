# Title Effectiveness Stats — Methodology & Implementation

---

## Overview

`title_effectiveness_stats.py` derives per-topic title pattern signals and optimal duration ranges from the training corpus and writes a static lookup JSON. It is pure computation — no model training, no inference calls.

All correlations are partial correlations controlling for `log_subscribers` to remove channel-size confounding. Raw title feature correlations against `views_per_day` would reflect channel size differences — large channels perform well regardless of whether a title contains a number. Partialling out subscriber count isolates the content signal.

Output is consumed by the inference orchestrator to give the Gemini recommendation layer concrete, data-backed title and duration strategy per content gap topic.

---

## Data Sources

| Source | File | Key Columns Used |
|---|---|---|
| Raw video metadata | `data/processed/combined_videos_raw.csv` | `video_id`, `video_title`, `channel_subscriberCount` |
| Clean video metadata | `data/processed/combined_videos_clean.csv` | `video_id`, `views_per_day`, `category_name`, `is_short` |
| BERTopic output | `data/processed/bertopic_metadata.csv` | `video_id`, `topic_id`, `duration_sec` |

**After filtering:** long-form only (`is_short == False`), `views_per_day > 0`, `topic_id != -1`, non-null titles and subscriber counts. 83 of 89 topics meet `_MIN_TOPIC_VIDEOS = 20` after the Shorts filter reduces per-topic counts. The 6 excluded topics fall back to their category entry at inference.

---

## Title Features

**Binary features:**

| Feature | Description |
|---|---|
| `has_number` | Title contains any digit |
| `has_question` | Title contains `?` |
| `has_brackets` | Title contains `[` or `(` |
| `has_caps_word` | Title contains an all-caps word of length > 1 |

**Continuous features:**

| Feature | Description |
|---|---|
| `wordcount` | Number of words |
| `char_length` | Character length of raw title |
| `sentiment` | Positive keyword count minus negative keyword count, clipped to [−2, 2] |

Positive keywords: `best, amazing, incredible, winning, success, grow, top, ultimate`
Negative keywords: `worst, fail, broke, lost, mistake, wrong, never`

---

## Methodology

### Partial Correlation

All feature correlations with `log1p(views_per_day)` are computed after regressing out `log1p(channel_subscriberCount)` via OLS projection:
```
resid_x = x - (cov · x / cov · cov) * cov
resid_y = y - (cov · y / cov · cov) * cov
partial_corr = pearson(resid_x, resid_y)
```

### Binary Features

Point-biserial correlation is computed manually on residuals to avoid NaN propagation from index misalignment after group slicing. Features with zero variance within a group are skipped. The `recommend` flag is set when `partial_corr > 0.02`.

### Continuous Features

Pearson correlation on residuals. Features or targets with zero standard deviation return `partial_correlation: 0.0`.

### Optimal Range

For continuous features and duration, the corpus is split into quartiles by feature value and the quartile with the highest mean `log1p(views_per_day)` is returned as `best_quartile_low` / `best_quartile_high`. Duration optimal ranges are additionally converted to minutes.

### Category Fallbacks

Topics with fewer than 20 long-form videos fall back to their category-level entry. Category-level stats are computed across all long-form videos in the category (4,000–6,000 videos per category) and are stable estimates.

---

## Coverage

| Metric | Value |
|---|---|
| Topics scored | 83 |
| Topics falling back to category median | 6 |
| Category fallbacks present | 6 (all categories) |
| Non-finite partial correlations | 0 |

---

## Output Schema
```json
{
  "topics": {
    "24": {
      "category": "tech",
      "n_videos": 279,
      "reliable": true,
      "features": {
        "has_number": {
          "partial_correlation": 0.0812,
          "rate_in_top": 0.312,
          "rate_in_rest": 0.241,
          "recommend": true
        },
        "wordcount": {
          "partial_correlation": 0.1021,
          "mean_in_top": 9.4,
          "mean_in_rest": 7.8,
          "optimal_range": {
            "best_quartile_low": 9.0,
            "best_quartile_high": 22.0
          }
        }
      },
      "duration": {
        "partial_correlation": 0.0731,
        "mean_in_top": 986.2,
        "mean_in_rest": 743.1,
        "mean_in_top_min": 16.4,
        "mean_in_rest_min": 12.4,
        "optimal_range": {
          "best_quartile_low": 686.0,
          "best_quartile_high": 1133.0
        },
        "optimal_range_min": {
          "best_quartile_low": 11.4,
          "best_quartile_high": 18.9
        }
      }
    }
  },
  "category_fallbacks": {
    "tech": {
      "n_videos": 3891,
      "features": { "..." },
      "duration": { "..." }
    }
  }
}
```

**Per-topic fields:**

| Field | Description |
|---|---|
| `category` | Dominant niche category |
| `n_videos` | Long-form videos contributing to estimates |
| `reliable` | `true` if `n_videos >= 20` |
| `features` | Per title feature partial correlation stats |
| `duration` | Duration partial correlation and optimal range |

**Per binary feature:**

| Field | Description |
|---|---|
| `partial_correlation` | Partial correlation with `log1p(views_per_day)` controlling for `log_subscribers` |
| `rate_in_top` | Fraction of above-median videos with this feature present |
| `rate_in_rest` | Fraction of below-median videos with this feature present |
| `recommend` | `true` if partial correlation > 0.02 |

**Per continuous feature:**

| Field | Description |
|---|---|
| `partial_correlation` | Partial correlation with `log1p(views_per_day)` controlling for `log_subscribers` |
| `mean_in_top` | Mean feature value among above-median videos |
| `mean_in_rest` | Mean feature value among below-median videos |
| `optimal_range` | Quartile range with highest mean target |

**Duration fields:**

| Field | Description |
|---|---|
| `partial_correlation` | Partial correlation of `duration_sec` with `log1p(views_per_day)` controlling for `log_subscribers` |
| `mean_in_top_min` | Mean duration in minutes among above-median videos |
| `mean_in_rest_min` | Mean duration in minutes among below-median videos |
| `optimal_range_min` | Optimal duration quartile range in minutes |

---

## Implementation

**File:** `src/modeling/title_effectiveness_stats.py`

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `_MIN_TOPIC_VIDEOS` | 20 | Minimum long-form videos for reliable topic estimate |

**Public API:**
```python
build_title_effectiveness_stats(
    raw_csv: str | Path = "data/processed/combined_videos_raw.csv",
    clean_csv: str | Path = "data/processed/combined_videos_clean.csv",
    meta_csv: str | Path = "data/processed/bertopic_metadata.csv",
    output_path: str | Path = "outputs/models/title_effectiveness_stats.json",
) -> dict
```

---

## Outputs

| File | Location | Description |
|---|---|---|
| `title_effectiveness_stats.json` | `outputs/models/` | Per-topic and per-category title feature and duration partial correlations |

No model artifacts produced. Regenerated nightly in the Cloud Run retraining job alongside XGBoost and BERTopic.