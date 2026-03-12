# CLIP Analyzer — Methodology & Implementation

---

## Overview

The CLIP analyzer derives data-driven visual blueprints from thumbnail performance data. It scores thumbnails across 73 visual axes using OpenAI's CLIP ViT-B/32 model, correlates axis scores against `views_per_day`, and produces per-niche blueprints that describe which visual features drive engagement in each content category.

The analyzer serves three roles in the production pipeline:

1. **Blueprint derivation (training-time)** — scores all 27,890 training corpus thumbnails and derives per-category blueprints saved to `niche_blueprints.json`
2. **Channel analysis (inference-time)** — scores a live creator's own thumbnails and derives a personalized blueprint, blended with the niche baseline
3. **Post-generation verification** — scores generated thumbnail candidates against the blueprint and returns them ranked by alignment

---

## Data Sources

| Source | File | Key Columns Used |
|---|---|---|
| Raw video metadata | `data/processed/combined_videos_raw.csv` | `video_id`, `thumb_maxres_url`, `thumb_standard_url`, `thumb_high_url` |
| Clean video metadata | `data/processed/combined_videos_clean.csv` | `video_id`, `category_name`, `views_per_day` |

Thumbnail URL priority chain: `thumb_maxres_url → thumb_standard_url → thumb_high_url`. `thumb_high_url` has zero nulls across 28,037 videos and serves as the guaranteed fallback.

**After filtering** (dropping `views_per_day ≤ 0` and videos with no resolvable URL): 27,919 thumbnails attempted, **27,890 successfully scored** (29 failed downloads — deleted or unavailable videos).

---

## Visual Axes

73 axes are scored per thumbnail, organized into 7 categories. Each axis is defined by a positive/negative text prompt pair. CLIP scores each image against both prompts and the result is normalized to [0, 1] where 1 = fully positive.

### Face & Person (9 axes)
`face_present`, `multiple_faces`, `close_crop_face`, `mouth_open`, `eye_contact`, `person_full_body`, `emotional_expression`, `negative_emotion`, `pointing_gesture`

### Composition & Framing (11 axes)
`rule_of_thirds`, `subject_left`, `subject_right`, `subject_centered`, `close_crop`, `wide_shot`, `object_prominent`, `multiple_objects`, `split_screen`, `before_after_layout`, `collage_layout`

### Color & Tone (12 axes)
`warm_tones`, `cool_tones`, `high_saturation`, `low_saturation`, `high_contrast`, `low_contrast`, `dark_background`, `bright_background`, `neon_colors`, `monochromatic`, `dramatic_lighting`, `golden_hour`

### Text & Graphics (9 axes)
`text_overlay`, `large_text`, `minimal_text`, `arrow_or_shape_overlay`, `brand_logo_present`, `number_in_text`, `question_in_text`, `emoji_or_symbol`, `progress_bar_or_meter`

### Background & Setting (10 axes)
`clean_background`, `busy_background`, `outdoor_setting`, `indoor_setting`, `studio_setting`, `nature_background`, `urban_background`, `blurred_background`, `underwater_setting`, `space_setting`

### Content-Specific (13 axes)
`food_visible`, `technology_visible`, `vehicle_visible`, `animal_visible`, `money_visible`, `fire_or_explosion`, `luxury_items`, `sports_action`, `gaming_content`, `science_experiment`, `travel_destination`, `crowd_or_audience`, `screen_recording`

### Thumbnail Style (9 axes)
`high_production_value`, `amateur_style`, `meme_format`, `clickbait_style`, `minimalist_style`, `illustrated_or_animated`, `cinematic_style`, `retro_vintage_style`, `thumbnail_vs_layout`

### Low-Confidence Axes

8 axes involve spatial or layout reasoning that CLIP handles unreliably. These are downweighted by 0.5× in all correlation computations:

```python
_LOW_CONFIDENCE_AXES = {
    "rule_of_thirds", "split_screen", "before_after_layout",
    "subject_left", "subject_right", "subject_centered",
    "collage_layout", "blurred_background",
}
```

---

## Scoring Method

Each image is scored against all 73 axis prompt pairs simultaneously. For each axis:

```
score = clip(( sim(image, pos_prompt) - sim(image, neg_prompt) + 2.0 ) / 4.0, 0.0, 1.0)
```

where `sim` is cosine similarity in CLIP embedding space. This produces a score in [0, 1] where 0.5 is neutral (equal similarity to both prompts), values above 0.5 indicate alignment with the positive prompt, and values below indicate alignment with the negative prompt.

Text prompts are encoded once at startup and cached. Image encoding runs in batches of 64.

---

## Download Strategy

Thumbnails are downloaded asynchronously in chunks of 500 to bound peak memory usage. Within each chunk, up to 32 concurrent connections are held open. After scoring, image objects are discarded and only the score array (float32, negligible size) is retained before the next chunk begins.

**Runtime (March 11, 2026, CPU, M-series Mac):**

| Stage | Duration |
|---|---|
| Model download (ViT-B/32, 338MB) | ~12s |
| Download + score 27,890 thumbnails | ~18 min |
| Blueprint derivation (all categories) | <1s |
| **Total** | **~18 min** |

This runs once during training and nightly in the Cloud Run retraining job. It does not run in the user session.

---

## Blueprint Derivation

For each category, axis scores are correlated against `log1p(views_per_day)` using a weighted Pearson correlation. Videos in the top quartile of `log1p(views_per_day)` are weighted 2× to bias the blueprint toward high-performing thumbnails rather than the median.

```
sample_weight = 2.0 if log1p(views_per_day) >= 75th percentile else 1.0
```

Each axis entry in the blueprint contains:

| Field | Description |
|---|---|
| `mean_score` | Mean CLIP score across all videos in category |
| `top_quartile_mean` | Mean CLIP score among top-25% performers |
| `correlation` | Weighted Pearson correlation with `log1p(views_per_day)`, scaled by axis weight |
| `direction` | `"positive"` or `"negative"` |
| `axis_weight` | `1.0` for standard axes, `0.5` for low-confidence axes |
| `n_videos` | Number of videos scored in this category |

A `_global` blueprint is also derived across all categories as a fallback.

---

## Results

### Score Distribution

Mean scores cluster tightly around 0.5 across all categories (std ~0.005). This is expected CLIP behavior — the model regresses toward neutral on ambiguous axes. The correlation signal, not the absolute score, is the operative measure.

| Category | Min score | Max score | Std |
|---|---|---|---|
| entertainment | 0.4798 | 0.5115 | 0.0054 |
| fitness | 0.4786 | 0.5104 | 0.0058 |
| food | 0.4786 | 0.5100 | 0.0058 |
| gaming | 0.4791 | 0.5117 | 0.0054 |
| research_science | 0.4843 | 0.5109 | 0.0047 |
| tech | 0.4817 | 0.5127 | 0.0054 |

### Correlation Strength (Global)

| Threshold | Axes |
|---|---|
| \|correlation\| > 0.20 | 0 |
| \|correlation\| > 0.10 | 18 |
| \|correlation\| > 0.05 | 47 |
| \|correlation\| ≤ 0.05 | 26 |
| Max | 0.1833 |
| Median | 0.0689 |

Correlations in the 0.05–0.20 range are meaningful at this scale (27K thumbnails, top-quartile weighted). No individual axis dominates — the blueprint derives its power from combining signal across many axes.

### Top Axes by Category

**Entertainment**

| Direction | Axis | Correlation |
|---|---|---|
| + | `minimal_text` | +0.1464 |
| + | `luxury_items` | +0.1128 |
| + | `thumbnail_vs_layout` | +0.0907 |
| − | `animal_visible` | −0.1634 |
| − | `text_overlay` | −0.1572 |
| − | `retro_vintage_style` | −0.1442 |

**Fitness**

| Direction | Axis | Correlation |
|---|---|---|
| + | `high_production_value` | +0.1861 |
| + | `close_crop` | +0.1809 |
| + | `money_visible` | +0.1591 |
| − | `nature_background` | −0.1531 |
| − | `food_visible` | −0.1362 |
| − | `animal_visible` | −0.1361 |

**Food**

| Direction | Axis | Correlation |
|---|---|---|
| + | `high_production_value` | +0.1955 |
| + | `multiple_objects` | +0.1855 |
| + | `travel_destination` | +0.1692 |
| − | `gaming_content` | −0.2310 |
| − | `monochromatic` | −0.2237 |
| − | `indoor_setting` | −0.2114 |

**Gaming**

| Direction | Axis | Correlation |
|---|---|---|
| + | `high_production_value` | +0.2047 |
| + | `technology_visible` | +0.2004 |
| + | `eye_contact` | +0.1986 |
| − | `meme_format` | −0.2464 |
| − | `text_overlay` | −0.2294 |
| − | `amateur_style` | −0.2059 |

**Research & Science**

| Direction | Axis | Correlation |
|---|---|---|
| + | `emotional_expression` | +0.1287 |
| + | `clean_background` | +0.1197 |
| + | `pointing_gesture` | +0.1151 |
| − | `food_visible` | −0.3053 |
| − | `animal_visible` | −0.2786 |
| − | `large_text` | −0.2651 |

**Tech**

| Direction | Axis | Correlation |
|---|---|---|
| + | `high_contrast` | +0.1723 |
| + | `travel_destination` | +0.1625 |
| + | `dramatic_lighting` | +0.1511 |
| − | `monochromatic` | −0.2264 |
| − | `question_in_text` | −0.1948 |
| − | `meme_format` | −0.1743 |

### Cross-Category Universals

19 axes hold consistent direction across all 6 categories:

**Always positive (recommend in every niche):**
`high_production_value`, `minimal_text`, `emotional_expression`, `negative_emotion`, `clickbait_style`, `close_crop_face`

**Always negative (avoid in every niche):**
`text_overlay`, `large_text`, `meme_format`, `monochromatic`, `animal_visible`, `food_visible`, `question_in_text`, `brand_logo_present`, `amateur_style`, `multiple_faces`, `person_full_body`, `cool_tones`, `gaming_content`

Notable finding: `meme_format` is strongly negative in gaming (−0.2464) despite being a format commonly associated with the niche. Top-performing gaming thumbnails have moved toward polished production aesthetics. `food_visible` is the strongest single negative signal in `research_science` (−0.3053), reflecting audience sensitivity to lifestyle bleed-in from unrelated niches.

---

## Channel-Level Blueprint (Inference)

At inference time, `analyze_channel_thumbnails()` downloads and scores a creator's own thumbnails, then blends the channel-specific correlation with the niche baseline. Blend weight scales with sample size:

```
blend_weight = min(1.0, n_channel_videos / 100.0)
blended_correlation = blend_weight * channel_corr + (1 - blend_weight) * niche_corr
```

At 20 videos (minimum threshold): 20% channel signal, 80% niche. At 100+ videos: 100% channel signal. If fewer than 20 valid images are downloadable, the niche blueprint is returned unchanged with `_source: "niche"`.

---

## Post-Generation Verification

`score_candidates()` accepts a list of PIL images (the 3 generated thumbnail candidates) and scores each against the blueprint. Alignment is computed as a weighted sum across all axes:

```
contribution = corr * score          (if corr > 0)
contribution = |corr| * (1 - score)  (if corr < 0)
alignment = sum(contribution * |corr| * axis_weight) / sum(|corr| * axis_weight)
```

Candidates are returned sorted by `alignment_score` descending. The top candidate is passed to finalization.

---

## Implementation

**File:** `src/modeling/clip_analyzer.py`

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `MODEL_NAME` | `"ViT-B/32"` | CLIP model variant |
| `BATCH_SIZE` | `64` | Images per CLIP forward pass |
| `DOWNLOAD_CONCURRENCY` | `32` | Max concurrent thumbnail download connections |
| `CHUNK_SIZE` | `500` | Thumbnails downloaded and scored per memory chunk |
| `MIN_CHANNEL_VIDEOS` | `20` | Minimum channel thumbnails required for channel-level blueprint |
| `TOP_QUARTILE_WEIGHT` | `2.0` | Sample weight multiplier for top-25% performers in correlation |
| `_LOW_CONFIDENCE_WEIGHT` | `0.5` | Correlation downweight for spatially ambiguous axes |

**Public API:**

| Function | Path | Description |
|---|---|---|
| `build_niche_blueprints()` | Training | Downloads corpus thumbnails, scores all axes, saves `niche_blueprints.json` |
| `analyze_channel_thumbnails()` | Inference | Scores creator's thumbnails, returns blended blueprint |
| `score_candidates()` | Inference | Ranks generated thumbnail candidates by blueprint alignment |
| `get_blueprint_summary()` | Inference | Returns condensed summary for Gemini translation layer |

---

## Outputs

| File | Location | Description |
|---|---|---|
| `niche_blueprints.json` | `outputs/models/` | Per-category and global blueprints, 73 axes each |

`niche_blueprints.json` is a training artifact. It is pushed to GCS alongside `bertopic_model`, `xgboost_longform.json`, and `xgboost_shorts.json` during the nightly retraining job and loaded at inference time via `analyze_channel_thumbnails()`.