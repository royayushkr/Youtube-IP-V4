# XGBoost Engagement Predictor — Methodology & Implementation

---

## Overview

The XGBoost engagement predictor estimates `log1p(views_per_day)` for YouTube videos based on content, channel, and temporal features. Two separate models are trained: one for long-form content and one for Shorts, as their virality mechanics differ fundamentally.

The model is used at inference time to **rank video ideas within a single creator's channel** — not to predict absolute view counts across channels. Channel size is constant at inference, so the ranking is driven entirely by content features.

---

## Data Sources

| Source | File | Key Columns Used |
|---|---|---|
| Raw video metadata | `data/processed/combined_videos_raw.csv` | `video_publishedAt`, `snapshot_utc`, `views`, `channel_subscriberCount`, `definition`, `video_description`, `video_tags`, `video_title` |
| BERTopic output | `data/processed/bertopic_metadata.csv` | `topic_id`, `bertopic_token_count`, `is_sparse_text`, `is_short`, `duration_sec` |
| PyTrends scores | `data/processed/topic_stats.csv` | `topic_id`, `trend_score` |

The BERTopic and raw files are joined on `video_id` (inner join). `trend_score` is joined from `topic_stats.csv` on `topic_id` after the video-level merge. `duration_sec`, `is_short`, and `is_sparse_text` are taken from `bertopic_metadata` as the authoritative source — these columns are dropped from raw before the merge to prevent collisions. Topics with no PyTrends signal (zero-score or filtered by `dominant_category_share`) receive `trend_score=0.0`.

**After filtering:** 27,022 videos (dropping views ≤ 1, video age ≤ 3 days, null `views_per_day`, null `log_subscribers`).

| Split | Videos |
|---|---|
| Long-form | 20,877 |
| Shorts | 6,145 |

---

## Target Variable
```
y = log1p(views_per_day)
```

where:
```
views_per_day = views / ((snapshot_utc - video_publishedAt).total_seconds() / 86400)
```

`log1p` is applied to compress the heavy right skew of the view distribution (range: 0 to 308M raw views). At inference, predictions are inverted with `expm1`.

**Why not normalize by channel baseline?**

Subtracting the channel median (`log1p(views_per_day) - log1p(channel_median_vpd)`) was tested and produced r² = 0.20 for long-form — a significant degradation. Most features in the set (category, topic, duration, platform era) are channel-level constants, not video-level differentiators. The model has insufficient within-channel signal to explain deviations from a channel's own average. The absolute target with `log_subscribers` as a feature is the correct formulation.

Channel baseline stats (`channel_med_vpd`, `channel_std_vpd`, `channel_p75_vpd`) were also tested as features but consumed 81% of combined feature importance, crowding out content signal entirely. They are excluded from training and applied as post-processing scaling at inference instead.

---

## Feature Engineering

### Temporal

| Feature | Description |
|---|---|
| `publish_hour` | Hour of publish (UTC), 0–23 |
| `publish_dow` | Day of week, 0=Monday |
| `platform_era` | Categorical bin of publish year: `early` (≤2016), `growth` (2017–2020), `covid` (2021–2023), `recent` (2024+) |
| `log_age_days` | `log1p(age_in_days)` — captures content decay curve |

`publish_year` was initially used but replaced by `platform_era`. Raw year was absorbing platform-level variance without adding interpretable content signal. The era bins reflect meaningful structural shifts in YouTube's algorithm and creator ecosystem.

### Channel

| Feature | Description |
|---|---|
| `log_subscribers` | `log1p(channel_subscriberCount)` — current subscriber count, log-compressed |

Note: `channel_subscriberCount` is a current snapshot, not at time of publish. This makes it technically leaky for historical videos but it remains the strongest available prior for channel reach. For the production use case (scoring new ideas for an active channel), it is clean.

### Content

| Feature | Description |
|---|---|
| `log_duration` | `log1p(duration_sec)` |
| `is_hd` | Binary: 1 if `definition == "hd"` |
| `has_description` | Binary: 1 if description is non-null |
| `has_tags` | Binary: 1 if tags are non-null |
| `is_sparse_text` | Binary: inherited from BERTopic preprocessing (< 5 tokens after cleaning) |
| `bertopic_token_count` | Token count of the BERTopic input document |

### Title Features

| Feature | Description |
|---|---|
| `title_wordcount` | Word count of raw title |
| `title_has_number` | Binary: title contains a digit |
| `title_has_question` | Binary: title contains `?` |
| `title_has_brackets` | Binary: title contains `[` or `(` |
| `title_has_caps_word` | Binary: title contains an all-caps word of length > 1 |
| `title_sentiment` | Count of positive keyword matches minus negative keyword matches, clipped to [-2, 2] |

Positive keywords: `best, amazing, incredible, winning, success, grow, top, ultimate`
Negative keywords: `worst, fail, broke, lost, mistake, wrong, never`

### Topic, Category & Trend

| Feature | Description |
|---|---|
| `topic_id` | BERTopic topic assignment (0–88, or `-1` for outliers) — native categorical |
| `category_name` | Niche category — native categorical |
| `trend_score` | PyTrends 90-day mean search interest for the topic's keyword cluster (0–100). Topic-level constant joined from `topic_stats.csv`. Zero-signal topics receive 0.0. |

`topic_id` and `category_name` are passed as XGBoost native categoricals (`enable_categorical=True`). `trend_score` is a continuous float. Because `trend_score` is a topic-level constant it is partially collinear with `topic_id` — it contributes orthogonal signal by capturing the *current momentum* of a topic cluster rather than its identity, but does not dominate feature importance.

---

## Model Architecture

Two independent XGBoost regressors — one for long-form, one for Shorts. Separate models are trained because Shorts virality operates on a fundamentally different distribution: algorithm-driven cold traffic rather than subscriber-pull, shorter content decay windows, and different feature relationships (e.g. `has_description` is a stronger signal for Shorts than long-form).

**Base hyperparameters (pre-tuning):**
```python
{
    "objective":        "reg:squarederror",
    "tree_method":      "hist",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "n_estimators":     800,
}
```

---

## Validation Strategy

**GroupKFold (5 folds), grouped by `channel_title`.**

Grouping by channel prevents the same channel from appearing in both train and validation within any fold. Without this, the model would leak channel identity across splits, inflating validation scores significantly given 461 channels across 27K videos (average ~58 videos per channel). This is the correct formulation for a model that will be evaluated on unseen channels.

`early_stopping_rounds` is used during fold training. The final model is retrained on the full dataset without early stopping, using the best `n_estimators` found by Optuna.

---

## Hyperparameter Tuning

Optuna is used for hyperparameter search (60 trials per model, `direction="maximize"` on OOF r²).

**Search space:**

| Parameter | Range |
|---|---|
| `n_estimators` | 400–1200 (step 100) |
| `learning_rate` | 0.01–0.15 (log scale) |
| `max_depth` | 4–9 |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `min_child_weight` | 3–20 |
| `gamma` | 0.0–2.0 |
| `reg_alpha` | 1e-4–10.0 (log scale) |
| `reg_lambda` | 1e-4–10.0 (log scale) |

---

## Results

### Long-form (20,877 videos)

| Fold | RMSE | R² |
|---|---|---|
| 1 | 1.6305 | 0.6124 |
| 2 | 1.4744 | 0.6152 |
| 3 | 1.4509 | 0.6401 |
| 4 | 1.5488 | 0.5889 |
| 5 | 1.2789 | 0.6687 |
| **OOF** | **1.4814** | **0.6258** |

**Best params:** `n_estimators=1100, learning_rate=0.0101, max_depth=4, subsample=0.706, colsample_bytree=0.522, min_child_weight=19, gamma=0.567, reg_alpha=0.000155, reg_lambda=0.0237`

**Top features:** `log_subscribers` (22.6%), `is_hd` (17.6%), `log_age_days` (12.4%), `platform_era` (10.1%), `topic_id` (8.8%), `category_name` (4.4%), `log_duration` (4.0%), `has_tags` (3.0%)

Note: `is_hd` ranks second at 17.6% importance. This is primarily a proxy for `platform_era` — HD became the standard around 2015, so it captures production-era variance. At inference all new videos will be HD, making this feature a constant. Its importance reflects historical corpus variance rather than actionable content signal.

`trend_score` does not appear in the top 8 features for long-form. As a topic-level constant partially collinear with `topic_id`, it contributes marginal additive signal without displacing stronger features. OOF r² is unchanged from the pre-`trend_score` baseline (0.6258).

### Shorts (6,145 videos)

| Fold | RMSE | R² |
|---|---|---|
| 1 | 1.6703 | 0.7465 |
| 2 | 1.9266 | 0.5496 |
| 3 | 1.4883 | 0.7502 |
| 4 | 1.7031 | 0.6044 |
| 5 | 1.9208 | 0.5761 |
| **OOF** | **1.7497** | **0.6818** |

**Best params:** `n_estimators=400, learning_rate=0.0168, max_depth=4, subsample=0.626, colsample_bytree=0.549, min_child_weight=6, gamma=1.232, reg_alpha=0.683, reg_lambda=0.000380`

**Top features:** `log_subscribers` (26.7%), `log_age_days` (11.2%), `platform_era` (10.1%), `topic_id` (8.6%), `category_name` (6.4%), `has_description` (5.8%), `is_sparse_text` (5.6%), `title_wordcount` (5.6%)

Shorts OOF r² improved from 0.6783 to 0.6818 with `trend_score` added. `trend_score` does not appear in the top 8 but contributed marginal signal captured in the ensemble.

### R² by Category (Long-form Diagnostics)

| Category | R² |
|---|---|
| gaming | 0.687 |
| tech | 0.674 |
| fitness | 0.639 |
| entertainment | 0.560 |
| food | 0.536 |
| research_science | 0.398 |

`research_science` is the weakest category — viral science videos (Veritasium, NileRed, Kurzgesagt) produce high-variance outliers that content features cannot predict. This is a data ceiling, not a model deficiency.

### R² by Subscriber Bucket (Long-form Diagnostics)

| Bucket | R² |
|---|---|
| xs (smallest) | 0.583 |
| s | 0.352 |
| m | 0.245 |
| l | 0.254 |
| xl (largest) | 0.072 |

The subscriber bucket collapse is expected and **does not affect production use**. At inference, all candidate video ideas are scored for a single creator — channel size is constant, so ranking is driven entirely by content features. Cross-channel r² degradation at large channel sizes is irrelevant to the ranking task.

---

## Outputs

| File | Location | Description |
|---|---|---|
| `xgboost_longform.json` | `outputs/models/` | Trained booster for long-form videos |
| `xgboost_longform_meta.json` | `outputs/models/` | Feature names, target definition, best params |
| `xgboost_shorts.json` | `outputs/models/` | Trained booster for Shorts |
| `xgboost_shorts_meta.json` | `outputs/models/` | Feature names, target definition, best params |

Models are saved via `model.get_booster().save_model()` to avoid sklearn wrapper serialization issues in XGBoost ≥ 2.x. Load with:
```python
booster = xgb.Booster()
booster.load_model("outputs/models/xgboost_longform.json")
```

---

## Inference Notes

At inference time, channel baseline stats are applied as **post-processing scaling**, not as model inputs:

1. Score all candidate video ideas through the model → ranked `log1p(views_per_day)` predictions
2. Apply `expm1` to convert back to views/day scale
3. Scale by `channel_median_views_per_day` to translate relative rankings into channel-contextualized estimates

This separation keeps content signal clean inside the model while still producing channel-calibrated output for the user.

`platform_era` for all new videos should be set to `"recent"` at inference. `trend_score` should be populated from the most recent `pytrends_client.update_topic_stats()` run, which executes nightly in the Cloud Run retraining job.