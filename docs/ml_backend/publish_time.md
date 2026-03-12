# Publish Time Stats — Methodology & Implementation

---

## Overview

`publish_time_stats.py` derives per-category optimal publish windows and upload cadence benchmarks from the training corpus and writes a static lookup JSON. It is pure computation — no model training, no inference calls.

All signals are computed as partial correlations controlling for `log_subscribers` to remove channel-size confounding. A large channel can publish at any hour and perform well due to existing audience pull — raw timing averages would reflect when large channels happen to post, not a genuine timing effect.

Output is consumed by the inference orchestrator to add publish timing and cadence signals to the Gemini recommendation context.

---

## Data Sources

| Source | File | Key Columns Used |
|---|---|---|
| Raw video metadata | `data/processed/combined_videos_raw.csv` | `video_id`, `video_publishedAt`, `channel_id`, `channel_subscriberCount` |
| Clean video metadata | `data/processed/combined_videos_clean.csv` | `video_id`, `views_per_day`, `category_name`, `is_short` |

After filtering (`views_per_day > 0`, non-null `video_publishedAt`, non-null `channel_subscriberCount`): all content types included. Publish time and cadence patterns are category-level signals — splitting by content type would reduce cell sizes without meaningful gain.

---

## Methodology

### Publish Time — Partial Correlation Approach

For each category and each hour (0–23 UTC) / day of week, a binary indicator variable is constructed (`1` if the video was published in that hour/day, `0` otherwise) and its partial correlation with `log1p(views_per_day)` is computed after regressing out `log1p(channel_subscriberCount)`.

Residuals are computed via OLS projection:
```
resid_x = x - (cov · x / cov · cov) * cov
resid_y = y - (cov · y / cov · cov) * cov
partial_corr = pearson(resid_x, resid_y)
```

Hours and days are ranked and normalized by `partial_correlation` rather than raw weighted mean. The weighted mean (`_TOP_QUARTILE_WEIGHT = 2.0` for top-25% performers) is retained in the output for reference but does not influence ranking.

### Cadence — Channel-Level Partial Correlation

Upload frequency (`videos_per_week`) is computed per channel from the span between first and last publish date. Channels with fewer than `_MIN_CHANNEL_VIDEOS = 10` videos or a span under 7 days are excluded.

The partial correlation between `videos_per_week` and `log1p(median_views_per_day)` is computed across channels within each category, controlling for `log1p(channel_subscriberCount)`. This answers: after accounting for channel size, do channels that post more frequently perform better?

A plain-English `interpretation` field is derived from the partial correlation value:

| Threshold | Interpretation |
|---|---|
| `pcorr > 0.05` | Higher cadence associated with better performance |
| `pcorr < -0.05` | Lower cadence associated with better performance |
| Between | Cadence not meaningfully associated with performance |

### Minimum Cell Threshold

`_MIN_CELL_VIDEOS = 20` — hours or days with fewer than 20 videos are omitted from output entirely.

---

## Results

| Category | Best hour (UTC) | Best day | Cadence partial corr | Median vpw |
|---|---|---|---|---|
| entertainment | 18 | sunday | 0.2034 | 3.02 |
| fitness | 15 | sunday | 0.3806 | 1.84 |
| food | 18 | tuesday | 0.2604 | 1.71 |
| gaming | 23 | friday | 0.2931 | 3.13 |
| research_science | 15 | sunday | 0.5242 | 0.50 |
| tech | 15 | saturday | 0.1728 | 1.65 |

All six categories show positive cadence partial correlation after controlling for subscribers — cadence genuinely associates with performance independent of channel size across every niche. Research/science is the strongest signal (0.52) despite the lowest median cadence, reflecting that consistency matters more than volume in that niche. Tech is the weakest (0.17), consistent with content quality being the primary driver there.

---

## Output Schema

`publish_time_stats.json` contains one entry per category:
```json
{
  "tech": {
    "n_videos": 4294,
    "best_hour_utc": 15,
    "best_dow": "saturday",
    "top_hours_utc": [15, 14, 16],
    "top_days": ["saturday", "tuesday", "wednesday"],
    "hour_stats": {
      "15": {
        "weighted_mean_log_vpd": 8.1234,
        "partial_correlation": 0.0312,
        "n_videos": 312,
        "score": 1.0
      }
    },
    "dow_stats": {
      "5": {
        "label": "saturday",
        "weighted_mean_log_vpd": 8.0921,
        "partial_correlation": 0.0289,
        "n_videos": 641,
        "score": 1.0
      }
    },
    "cadence": {
      "n_channels": 87,
      "median_videos_per_week": 1.65,
      "p25_videos_per_week": 0.81,
      "p75_videos_per_week": 2.94,
      "partial_correlation": 0.1728,
      "interpretation": "higher cadence associated with better performance (controlling for channel size)"
    }
  }
}
```

| Field | Description |
|---|---|
| `n_videos` | Total videos in category after filtering |
| `best_hour_utc` | Single best publish hour (UTC) by partial correlation |
| `best_dow` | Single best day of week by partial correlation |
| `top_hours_utc` | Top 3 hours by normalized partial correlation score |
| `top_days` | Top 3 days by normalized partial correlation score |
| `hour_stats` | Per-hour weighted mean, partial correlation, video count, normalized score |
| `dow_stats` | Per-day weighted mean, partial correlation, video count, normalized score, label |
| `cadence.partial_correlation` | Channel-level partial correlation of cadence with performance |
| `cadence.interpretation` | Plain-English cadence signal for Gemini context |

All times are UTC. The Gemini translation layer is responsible for converting to the creator's local timezone if known.

---

## Implementation

**File:** `src/modeling/publish_time_stats.py`

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `_MIN_CELL_VIDEOS` | 20 | Minimum videos per hour/day cell |
| `_MIN_CHANNEL_VIDEOS` | 10 | Minimum videos per channel for cadence computation |
| `_TOP_QUARTILE_WEIGHT` | 2.0 | Sample weight for weighted mean (reference only, not used for ranking) |

**Public API:**
```python
build_publish_time_stats(
    raw_csv: str | Path = "data/processed/combined_videos_raw.csv",
    clean_csv: str | Path = "data/processed/combined_videos_clean.csv",
    output_path: str | Path = "outputs/models/publish_time_stats.json",
) -> dict
```

---

## Outputs

| File | Location | Description |
|---|---|---|
| `publish_time_stats.json` | `outputs/models/` | Per-category hourly, day-of-week, and cadence performance lookup |

No model artifacts produced. Regenerated nightly in the Cloud Run retraining job alongside XGBoost and BERTopic.