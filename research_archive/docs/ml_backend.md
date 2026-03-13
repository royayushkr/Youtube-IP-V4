# ML Backend Documentation

---

## Overview

The ML backend is a multi-model pipeline that takes a creator's live YouTube channel data and produces ranked content recommendations, visual thumbnail blueprints, and content gap opportunities — all grounded in corpus-derived statistics and predictive models trained on 28K videos across 6 niches.

The pipeline is designed around a strict separation between training time and inference time. Training artifacts (models, JSON stat lookups) are produced once and updated nightly. At inference, only live YouTube API data and pre-built artifacts are used. Training data files are never read at inference time.

**Any creator can use this product regardless of whether their channel exists in the training corpus.** All inference paths work on live channel data.

---

## Corpus

28,037 videos across 6 niche categories, collected from ~460 channels via the YouTube Data API v3.

| Category | Videos |
|---|---|
| research_science | 6,779 |
| entertainment | 4,690 |
| tech | 4,294 |
| food | 4,126 |
| gaming | 4,093 |
| fitness | 4,055 |

**Key processed files:**

| File | Contents |
|---|---|
| `data/processed/combined_videos_raw.csv` | Raw YouTube API metadata, 28K videos |
| `data/processed/combined_videos_clean.csv` | Engineered features — `views_per_day`, `like_rate`, `is_short`, publish time fields |
| `data/processed/bertopic_metadata.csv` | Per-video `topic_id`, `is_sparse_text`, `bertopic_token_count`, `category_name` |
| `data/processed/topic_stats.csv` | Per-topic `trend_score`, `median_views`, `trajectory`, `top_category` |

**Column collision warning:** `is_short` and `duration_sec` exist in both `combined_videos_clean.csv` and `bertopic_metadata.csv`. Always use `is_short` from the clean file. When merging with `bertopic_metadata.csv`, pull only `["video_id", "topic_id", "bertopic_token_count", "is_sparse_text"]`.

---

## Model Inventory

| Artifact | File | Type | Used at inference |
|---|---|---|---|
| BERTopic model | `outputs/models/bertopic_model/` | Trained model | Yes |
| XGBoost reach — longform | `outputs/models/xgboost_longform.json` | Trained model | Yes |
| XGBoost reach — shorts | `outputs/models/xgboost_shorts.json` | Trained model | Yes |
| XGBoost engagement — shorts | `outputs/models/xgboost_engagement_shorts.json` | Trained model | Yes |
| XGBoost engagement — longform | `outputs/models/xgboost_engagement_longform.json` | Trained model | No — R²=0.086, discarded |
| CLIP niche blueprints | `outputs/models/niche_blueprints.json` | Corpus statistics | Yes |
| Topic engagement stats | `outputs/models/topic_engagement_stats.json` | Corpus statistics | Yes |
| Publish time stats | `outputs/models/publish_time_stats.json` | Corpus statistics | Yes |
| Title effectiveness stats | `outputs/models/title_effectiveness_stats.json` | Corpus statistics | Yes |
| Topic trend baseline | `outputs/models/topic_trend_baseline.json` | Corpus statistics | Yes |

---

## Components

### 1. BERTopic — Topic Modeling

**Script:** `src/modeling/bertopic_trainer.py`
**Doc:** `bertopic.md`

Clusters the corpus into 89 topics using BERTopic on video titles, descriptions, and tags. Topic `-1` is the noise/outlier bin and is excluded from all downstream scoring.

At inference, `topic_model.transform(docs)` assigns topic IDs to a creator's live video library. This is the foundation of all downstream gap scoring and reach prediction — topic assignment is the first step in the inference chain for any creator.

Topics are labeled with a 4-word string derived from the most representative terms (e.g., `16_twitch_streaming_gaming channel_stream`). Labels are used in Gemini context construction, not for ML purposes.

---

### 2. PyTrends — Trend Scoring

**Script:** `src/data_collection/pytrends_client.py`
**Doc:** `pytrends.md`

Scores each BERTopic topic on Google search interest using PyTrends. Produces `data/processed/topic_stats.csv` with `trend_score` (90-day mean), `trend_score_recent` (30-day mean), and `trajectory` (their ratio).

`trajectory > 1.2` = rising topic. `trajectory < 0.8` = declining topic. Trajectory is used as a component in the content gap scorer and carried through to the topic trend baseline.

At inference, PyTrends is also called live to get real-time search interest for the creator's identified topics — the corpus `trend_score` provides the baseline context; the live call provides the current signal.

---

### 3. XGBoost Reach Models

**Script:** `src/modeling/xgboost_reach_trainer.py`
**Doc:** `xgboost.md`

Two separate models predicting `log1p(views_per_day)` — one for long-form content, one for Shorts. Trained with GroupKFold cross-validation grouped by channel to prevent data leakage.

| Model | R² |
|---|---|
| Long-form | 0.626 |
| Shorts | 0.682 |

20 features including `log_subscribers`, `log_age_days`, `topic_id`, `trend_score`, `category_name`, and title pattern features. `topic_id` and `category_name` must be encoded as `category` dtype at inference.

At inference: load booster from JSON, engineer features from live channel data, call `predict()`, return `expm1(prediction)` to recover views/day. Predictions are scaled relative to the creator's own channel median to produce personalized forecasts rather than raw corpus-scale outputs.

---

### 4. XGBoost Engagement Models

**Script:** `src/modeling/xgboost_engagement_trainer.py`
**Doc:** `engagement.md`

Predicts `log1p(like_rate)`. Two models were trained — only the Shorts model is used in production.

**Shorts model (R²=0.25):** `trend_score` is the dominant feature, indicating Shorts engagement is driven primarily by topic timing rather than channel size. Used at inference for Shorts content gap scoring.

**Long-form model (R²=0.086, not used):** Dominated by `log_subscribers` and `is_hd` — the model learns channel size proxies rather than content signal. Replaced by corpus-derived topic engagement percentiles for long-form.

---

### 5. Topic Engagement Stats

**Script:** `src/modeling/topic_engagement_stats.py`
**Doc:** `engagement.md`

Replaces the failed long-form engagement model. Computes per-topic `like_rate` percentiles from the corpus as a lookup table. Topics with fewer than 20 long-form videos fall back to their category median.

At inference: look up a gap topic's `engagement_percentile` (within-category rank, 0–1) and `global_percentile` (across all topics) by `topic_id`. These percentiles are passed to Gemini as the engagement signal for long-form recommendations.

---

### 6. CLIP Visual Analyzer

**Script:** `src/modeling/clip_analyzer.py`
**Doc:** `clip.md`

Uses OpenAI CLIP (ViT-B/32) to analyze 27,890 training thumbnails and derive per-category visual blueprints — data-derived specifications of which visual features correlate with engagement in each niche.

73 visual axes are scored per thumbnail by projecting CLIP embeddings onto concept axes via positive/negative text prompt pairs. The correlation of each axis score with `log1p(views_per_day)` is computed within each category, weighted 2× for top-quartile performers.

At inference, the same CLIP model analyzes the creator's own thumbnail library and blends their personal visual patterns with the niche blueprint. The blend weight ramps from 20% channel influence at 20 videos to 100% at 100+ videos. The resulting blueprint drives Gemini's thumbnail prompt construction and is used post-generation to rank candidate thumbnails.

---

### 7. Content Gap Scorer

**Script:** `src/modeling/content_gap_scorer.py`
**Doc:** `content_gap.md`

Pure computation — no model. Takes a creator's covered topic IDs and ranks uncovered topics in their category by a weighted opportunity score:

```
gap_score = 0.35 × reach_score
          + 0.30 × trend_score
          + 0.20 × trajectory_score
          + 0.15 × engagement_score
```

All components are min-max normalized within the creator's category before weighting. Topics below `dominant_category_share < 0.70` or `video_count < 30` are filtered out before scoring.

The gap scorer's output — ranked topic opportunities — is the primary input to Gemini's recommendation generation. The top gaps are what the orchestrator asks Gemini to generate video ideas around.

---

### 8. Publish Time Stats

**Script:** `src/modeling/publish_time_stats.py`
**Doc:** `publish_time_stats.md`

Derives per-category optimal publish windows and upload cadence benchmarks. All signals are computed as partial correlations controlling for `log_subscribers` to remove channel-size confounding — raw timing averages would reflect when large channels happen to post, not a genuine timing effect.

Hour and day-of-week rankings are based on partial correlation of publish time indicators with `log1p(views_per_day)`, not raw weighted means. Cadence (videos/week) is similarly expressed as a partial correlation across channels within each category.

At inference, the orchestrator loads this file and passes the relevant category's timing signals and cadence interpretation directly to Gemini context.

---

### 9. Title Effectiveness Stats

**Script:** `src/modeling/title_effectiveness_stats.py`
**Doc:** `title_effectiveness_stats.md`

Derives per-topic title pattern signals and optimal duration ranges. Seven features are computed per title: four binary (`has_number`, `has_question`, `has_brackets`, `has_caps_word`) and three continuous (`wordcount`, `char_length`, `sentiment`). Duration is analyzed separately.

All feature-to-performance correlations are partial correlations controlling for `log_subscribers`. Raw correlations would reflect channel size, not genuine title signal — large channels perform well regardless of title format. Topics with fewer than 20 long-form videos fall back to their category-level entry.

At inference, the orchestrator looks up the gap topic's title feature signals and duration optimal range and includes them in the Gemini prompt to ground title generation in corpus evidence.

---

### 10. Topic Trend Baseline

**Script:** `src/modeling/topic_trend_baseline.py`
**Doc:** `topic_trend_baseline.md`

Provides the reference distribution needed to contextualize live trend scores. Without a baseline, the orchestrator can report that a topic has a trend score of 77 — but cannot say whether that is high or low for that topic.

Three signals are derived per topic:

- **Engagement baseline:** mean and std of `log1p(views_per_day)` across the topic corpus, giving the orchestrator a performance reference distribution
- **Trend z-score:** `trend_score_recent` standardized against the distribution of trend scores within the topic's category — expresses how strongly a topic is trending relative to its niche peers
- **Seasonality index:** per-month performance multiplier relative to the topic's annual mean, derived from corpus publish timestamps and `views_per_day`

All three are passed to Gemini context, enabling recommendations like "this topic is trending 1.4 standard deviations above its niche average and historically peaks in September."

---

## Statistical Validity Notes

Several components produce corpus statistics rather than trained models. Where these statistics involve correlating a feature with a performance outcome, channel-size confounding is a validity threat — large channels outperform small ones regardless of the feature being studied. This is addressed consistently across all stat-derivation scripts.

**Partial correlation is applied in:** publish time stats (hour/day/cadence), title effectiveness stats (all title features and duration), topic trend baseline (seasonality partial correlation).

**Not applied in:** topic engagement stats (like_rate percentiles are descriptive within-topic distributions, not feature correlations), CLIP blueprints (top-quartile weighting provides implicit size adjustment), topic trend baseline engagement baseline (describes the target distribution itself, not a feature relationship).

All partial correlations use OLS residualization: both the feature and target are regressed against `log1p(channel_subscriberCount)` and the residuals are correlated. This is equivalent to the standard partial correlation formula but computed without scipy to avoid index misalignment issues in grouped operations.

---

## Inference Chain (Planned)

**Script:** `src/modeling/inference.py` — not yet built.

The orchestrator chains all components on live creator channel data fetched from the YouTube Data API at runtime. Input is a `channel_data` dict with the creator's videos, metadata, and channel stats. Output is a structured dict ready for Gemini prompt construction.

**Sequence:**

1. **BERTopic** — assign topic IDs to creator's video library via `topic_model.transform()`
2. **XGBoost reach** — predict `views_per_day` per gap topic using live channel features
3. **XGBoost engagement / topic stats** — predict `like_rate` (Shorts) or look up engagement percentile (long-form)
4. **Content gap scorer** — rank uncovered topics by `gap_score`
5. **CLIP** — analyze creator's thumbnails, blend with niche blueprint
6. **Stat lookups** — load publish time, title effectiveness, and trend baseline signals for gap topics
7. **Gemini** — structured context from all above passed to Gemini 2.5 Pro for recommendation generation, translated to plain English by Gemini 2.0 Flash

All models are loaded once per process via a model cache and reused across requests. CLIP and BERTopic cold start takes 10–20 seconds — `--min-instances=1` on Cloud Run keeps one instance warm at all times.

---

## Retraining

All models and stat artifacts are regenerated nightly by a Cloud Run Job triggered by Cloud Scheduler. The job pulls fresh YouTube API data, retrains XGBoost and BERTopic, recomputes all stat lookups, and pushes updated artifacts to GCS. The Streamlit application loads artifacts from GCS at startup via `st.cache_resource`.

---

## Documentation Index

| File | Component |
|---|---|
| `bertopic.md` | BERTopic topic modeling |
| `pytrends.md` | PyTrends trend scoring |
| `xgboost.md` | XGBoost reach models |
| `engagement.md` | XGBoost engagement models + topic engagement stats |
| `clip.md` | CLIP visual analyzer + niche blueprints |
| `content_gap.md` | Content gap scorer |
| `publish_time_stats.md` | Publish time and cadence stats |
| `title_effectiveness_stats.md` | Title pattern and duration stats |
| `topic_trend_baseline.md` | Topic engagement baseline and trend z-scores |
