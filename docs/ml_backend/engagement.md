# Engagement Modeling — Methodology & Implementation

---

## Overview

Engagement modeling adds a second optimization axis alongside the reach model (`xgboost_reach_trainer.py`). Where the reach model predicts `log1p(views_per_day)`, engagement modeling captures audience quality: how actively a topic's viewers interact with content rather than passively consuming it.

`like_rate` and `views_per_day` are negatively correlated (Spearman = −0.054 and −0.153 respectively). High-view videos attract passive audiences; high-engagement topics attract invested ones. These are distinct signals that move in opposite directions and must be modeled separately.

The engagement layer consists of two components:

1. **XGBoost engagement model (Shorts only)** — `xgboost_engagement_trainer.py` trains on `log1p(like_rate)` using the same feature set as the reach model. Viable for Shorts (OOF r² = 0.25) but not for long-form (OOF r² = 0.086).
2. **Per-topic engagement percentiles (long-form)** — `topic_engagement_stats.py` computes topic-level `like_rate` statistics from the training corpus and produces a lookup table of engagement percentiles by topic. No model, pure corpus statistics.

At inference, the two signals combine as:

| Content type | Reach signal | Engagement signal |
|---|---|---|
| Long-form | `xgboost_reach_longform` prediction | Topic engagement percentile from `topic_engagement_stats.json` |
| Shorts | `xgboost_reach_shorts` prediction | `xgboost_engagement_shorts` prediction |

---

## Why Long-form Engagement Modeling Fails

The long-form engagement model was trained and evaluated. OOF r² = 0.086 across 20,718 videos is not usable as a ranking signal.

**Root cause:** `like_rate` variance across long-form videos is extremely low (std = 0.024, range 0.0001–0.693). Within-channel variance is even lower: most channels produce videos with consistent engagement rates regardless of topic. The model cannot find content features that reliably explain engagement rate differences because they largely don't exist at the video level.

**Feature importance confirms this:** `log_subscribers` (28.9%) and `is_hd` (23.0%) dominate — both are channel-level or era-level constants, not content signals. Six features have zero importance. The model is learning channel identity proxies rather than content characteristics.

**Why Shorts is different:** Shorts are algorithm-driven rather than subscriber-pull. Cold traffic from non-subscribers means engagement rate is more dependent on content quality than audience loyalty. `trend_score` is the top feature at 23%; timely Shorts content drives genuine engagement rather than passive views from an existing audience.

---

## Data Sources

| Source | File | Key Columns Used |
|---|---|---|
| Clean video metadata | `data/processed/combined_videos_clean.csv` | `video_id`, `like_rate`, `comment_rate`, `is_short`, `category_name` |
| Raw video metadata | `data/processed/combined_videos_raw.csv` | `video_publishedAt`, `snapshot_utc`, `views`, `channel_subscriberCount`, `definition`, `video_description`, `video_tags`, `video_title` |
| BERTopic output | `data/processed/bertopic_metadata.csv` | `video_id`, `topic_id`, `bertopic_token_count`, `is_sparse_text`, `duration_sec` |
| PyTrends scores | `data/processed/topic_stats.csv` | `topic_id`, `trend_score`, `top_category` |

**Note:** `is_short` is pulled from `combined_videos_clean.csv` exclusively. Both `combined_videos_raw.csv` and `bertopic_metadata.csv` contain `is_short`. The authoritative source is the clean file. The merge excludes `is_short` from the metadata pull to prevent column collision.

**After filtering** (dropping `like_rate ≤ 0`, `video_age_days ≤ 3`, null `log_subscribers`): 26,738 videos.

| Split | Videos |
|---|---|
| Long-form | 20,718 |
| Shorts | 6,020 |

---

## Target Variable

```
y = log1p(like_rate)
```

where:
```
like_rate = likes / views
```

`log1p` compresses the right skew. At inference, predictions are inverted with `expm1`.

**Distribution of `log1p(like_rate)`:**

| Stat | Long-form | Shorts |
|---|---|---|
| Mean | 0.0373 | 0.0334 |
| Std | 0.0239 | 0.0249 |
| Median | 0.0344 | 0.0292 |
| Max | 0.6931 | 0.6931 |

**`log_subscribers` vs `log1p(like_rate)` Spearman correlation: −0.12** — substantially weaker than the reach model's −0.54. Content features carry proportionally more weight here, particularly for Shorts.

**Mean `like_rate` by category:**

| Category | Mean like_rate |
|---|---|
| research_science | 0.0412 |
| food | 0.0387 |
| entertainment | 0.0359 |
| tech | 0.0356 |
| gaming | 0.0356 |
| fitness | 0.0354 |

`research_science` leads — dedicated science audiences engage more actively than passive entertainment or gaming viewers.

---

## XGBoost Engagement Model (Shorts)

Feature set, validation strategy, and hyperparameter tuning are identical to `xgboost_reach_trainer.py`. See the reach model documentation for methodology details.

### Results

| Fold | RMSE | R² |
|---|---|---|
| 1 | 0.0286 | 0.1276 |
| 2 | 0.0217 | 0.3773 |
| 3 | 0.0199 | −0.4329 |
| 4 | 0.0211 | 0.4265 |
| 5 | 0.0156 | 0.3979 |
| **OOF** | **0.0218** | **0.2507** |

Fold 3 at r² = −0.43 indicates one channel group is badly out of distribution — a small number of outlier Shorts creators dominate that fold's validation set. OOF r² = 0.25 is meaningful enough to use as a secondary ranking signal for Shorts, with the caveat that predictions are less reliable for channels whose content style deviates significantly from the training distribution.

**Best params:** `n_estimators=500, learning_rate=0.0109, max_depth=5, subsample=0.825, colsample_bytree=0.934, min_child_weight=7, gamma=0.002, reg_alpha=0.000102, reg_lambda=0.000536`

**Top features:** `trend_score` (23.1%), `log_subscribers` (9.9%), `category_name` (8.6%), `bertopic_token_count` (8.3%), `log_age_days` (7.2%), `title_has_number` (6.1%), `is_sparse_text` (4.7%), `topic_id` (4.5%)

`trend_score` leading at 23% is the key finding. For Shorts, topic timing is the primary driver of audience engagement quality - not channel size, not content style. This has direct implications for the recommendation layer: trending Shorts topics should be surfaced with an explicit engagement quality signal, not just a reach prediction.

### Long-form Results (Not Used)

| Fold | RMSE | R² |
|---|---|---|
| 1 | 0.0295 | 0.1510 |
| 2 | 0.0225 | 0.0549 |
| 3 | 0.0184 | −0.0717 |
| 4 | 0.0210 | −0.0781 |
| 5 | 0.0208 | 0.1389 |
| **OOF** | **0.0227** | **0.0861** |

**Top features:** `log_subscribers` (28.9%), `is_hd` (23.0%), `topic_id` (19.4%), `log_age_days` (14.3%), `log_duration` (14.3%) — six features at zero importance. Not used in production.

---

## Per-Topic Engagement Percentiles (Long-form)

`topic_engagement_stats.py` replaces the failed long-form engagement model with a simpler, more reliable approach: compute median `like_rate` per topic from the training corpus, normalize within category, and produce a percentile lookup table.

### Methodology

1. Filter to long-form videos (`is_short == False`) with `like_rate > 0`
2. Compute per-topic `mean`, `median`, `std`, `p75`, and `count` of `like_rate`
3. Topics with fewer than 20 videos fall back to their category median — marked `reliable: false`
4. Compute `engagement_percentile` as within-category percentile rank on median `like_rate`
5. Compute `engagement_score` as within-category min-max normalized score (0–1)
6. Compute `global_percentile` across all topics regardless of category

### Coverage

| Metric | Value |
|---|---|
| Topics scored | 90 |
| Reliable (≥ 20 videos) | 29 |
| Topics falling back to category median | 61 |

The low reliable count (29/90) reflects the `is_short` filter reducing per-topic video counts. Category median fallbacks are stable estimates — category-level `like_rate` variance is low and consistent across the training corpus.

### Top and Bottom Topics by Engagement Percentile

**Top 10:**

| Topic | Category | Median like_rate | Percentile | Videos |
|---|---|---|---|---|
| 50 | tech | 0.0663 | 1.00 | 96 |
| 78 | tech | 0.0622 | 0.99 | 35 |
| 19 | tech | 0.0461 | 0.97 | 34 |
| 76 | gaming | 0.0516 | 0.97 | 42 |
| 77 | tech | 0.0444 | 0.96 | 60 |
| 24 | tech | 0.0444 | 0.94 | 279 |
| 18 | tech | 0.0424 | 0.93 | 20 |
| 48 | tech | 0.0416 | 0.92 | 88 |
| 83 | tech | 0.0401 | 0.90 | 40 |
| 82 | tech | 0.0368 | 0.89 | 42 |

**Bottom 10:**

| Topic | Category | Median like_rate | Percentile | Videos |
|---|---|---|---|---|
| 22 | tech | 0.0296 | 0.08 | 188 |
| 63 | entertainment | 0.0184 | 0.08 | 51 |
| 70 | tech | 0.0267 | 0.07 | 46 |
| 12 | tech | 0.0259 | 0.06 | 427 |
| 31 | gaming | 0.0203 | 0.05 | 56 |
| 51 | tech | 0.0233 | 0.04 | 90 |
| 79 | tech | 0.0134 | 0.03 | 46 |
| 65 | food | 0.0241 | 0.02 | 24 |
| 52 | tech | 0.0090 | 0.01 | 79 |
| 62 | entertainment | 0.0014 | 0.01 | 50 |

Tech dominates both ends — niche-specific tech content (cybersecurity, privacy, engineering) drives strong engagement while generic consumer tech and unboxing content produces passive viewers. Topic 62 (`star wars credits writers producer original`) and topic 52 (`coding code programming web`) sit at near-zero engagement — entertainment credit content and programming tutorials both attract viewers who watch without interacting.

### Output Schema

Each topic entry in `topic_engagement_stats.json`:

| Field | Description |
|---|---|
| `category` | Dominant niche category |
| `mean_like_rate` | Mean `like_rate` across topic videos |
| `median_like_rate` | Median `like_rate` — primary signal, robust to outliers |
| `p75_like_rate` | 75th percentile `like_rate` |
| `std_like_rate` | Standard deviation |
| `n_videos` | Videos contributing to estimate |
| `reliable` | `true` if n_videos ≥ 20, `false` if backed by category median |
| `engagement_percentile` | Within-category percentile rank (0–1) |
| `engagement_score` | Within-category min-max normalized score (0–1) |
| `global_percentile` | Percentile rank across all topics (0–1) |
| `trend_score` | PyTrends 90-day mean interest, joined from `topic_stats.csv` |

Category-level fallback entries are stored under `category_fallbacks` for unknown topics at inference time, each with `engagement_percentile: 0.5`, `engagement_score: 0.5`, and `global_percentile: 0.5`.

---

## Implementation

**Files:**

| File | Location | Purpose |
|---|---|---|
| `xgboost_engagement_trainer.py` | `src/modeling/` | Trains Shorts engagement model |
| `topic_engagement_stats.py` | `src/modeling/` | Computes per-topic engagement percentiles |

**Key constants (`xgboost_engagement_trainer.py`):**

| Constant | Value | Purpose |
|---|---|---|
| `MIN_LIKE_RATE` | `0.0` | Minimum like_rate threshold (strict positive) |
| `MIN_AGE_DAYS` | `3` | Minimum video age to exclude fresh uploads |
| `N_TRIALS` | `60` | Optuna hyperparameter search trials |
| `N_CV_FOLDS` | `5` | GroupKFold cross-validation folds |

**Key constants (`topic_engagement_stats.py`):**

| Constant | Value | Purpose |
|---|---|---|
| `MIN_VIDEOS` | `20` | Minimum videos for reliable topic estimate |

---

## Outputs

| File | Location | Description |
|---|---|---|
| `xgboost_engagement_shorts.json` | `outputs/models/` | Trained Shorts engagement booster |
| `xgboost_engagement_shorts_meta.json` | `outputs/models/` | Feature names, target, best params |
| `xgboost_engagement_longform.json` | `outputs/models/` | Trained long-form booster — not used in production |
| `xgboost_engagement_longform_meta.json` | `outputs/models/` | Feature names, target, best params |
| `topic_engagement_stats.json` | `outputs/models/` | Per-topic engagement percentiles, 90 topics + category fallbacks |

`xgboost_engagement_longform.json` is saved but excluded from the inference orchestrator. It is retained for reference and potential future use if per-channel engagement data accumulates sufficient within-channel variance to make the model viable.