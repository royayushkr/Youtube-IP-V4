import asyncio
import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
import clip

warnings.filterwarnings("ignore")

MODEL_NAME = "ViT-B/32"
BATCH_SIZE = 64
DOWNLOAD_CONCURRENCY = 32
CHUNK_SIZE = 500
MIN_CHANNEL_VIDEOS = 20
TOP_QUARTILE_WEIGHT = 2.0
BLUEPRINT_PATH = Path("outputs/models/niche_blueprints.json")

_LOW_CONFIDENCE_AXES = {
    "rule_of_thirds", "split_screen", "before_after_layout",
    "subject_left", "subject_right", "subject_centered",
    "collage_layout", "blurred_background",
}
_LOW_CONFIDENCE_WEIGHT = 0.5

AXES: dict[str, tuple[str, str]] = {
    # face & person
    "face_present":            ("a thumbnail with a human face clearly visible", "a thumbnail with no people or faces"),
    "multiple_faces":          ("a thumbnail showing two or more people", "a thumbnail with one person or no people"),
    "close_crop_face":         ("an extreme close-up of a face filling the frame", "a wide shot where the face is small"),
    "mouth_open":              ("a person with mouth wide open in surprise or excitement", "a person with mouth closed or neutral expression"),
    "eye_contact":             ("a person looking directly into the camera", "a person looking away from the camera"),
    "person_full_body":        ("a full body shot showing the entire person", "a headshot or partial body"),
    "emotional_expression":    ("a person showing intense surprise, excitement, or joy", "a person with a calm or neutral expression"),
    "negative_emotion":        ("a person showing fear, anger, disgust, or shock", "a person smiling or showing positive emotion"),
    "pointing_gesture":        ("a person pointing with their finger toward something", "a person with arms at sides or not pointing"),
    # composition & framing
    "rule_of_thirds":          ("a photo where the subject is placed at the intersection of thirds", "a photo where the subject is dead center"),
    "subject_left":            ("the main subject positioned on the left side of the frame", "the main subject on the right or center"),
    "subject_right":           ("the main subject positioned on the right side of the frame", "the main subject on the left or center"),
    "subject_centered":        ("the main subject perfectly centered in the frame", "the main subject off-center"),
    "close_crop":              ("a tightly cropped image with subject filling the frame", "a wide shot with lots of empty space"),
    "wide_shot":               ("a wide establishing shot showing full environment", "a tight close-up shot"),
    "object_prominent":        ("a prominent object or product as the main focal point, no face", "a person as the main subject"),
    "multiple_objects":        ("many different objects cluttered in the frame", "a single clean focal object"),
    "split_screen":            ("a thumbnail divided into two halves showing different things", "a single unified image"),
    "before_after_layout":     ("a before and after comparison layout", "a single scene with no comparison"),
    "collage_layout":          ("a collage of multiple smaller images arranged together", "a single full-bleed image"),
    # color & tone
    "warm_tones":              ("a photo with warm orange, red, and yellow color tones", "a photo with cool blue and green color tones"),
    "cool_tones":              ("a photo with cool blue, teal, and purple color tones", "a photo with warm orange and red tones"),
    "high_saturation":         ("a highly saturated vivid colorful image", "a desaturated muted dull image"),
    "low_saturation":          ("a desaturated muted low-color image", "a vivid highly saturated colorful image"),
    "high_contrast":           ("a high contrast image with bright highlights and deep shadows", "a flat low contrast image with even tones"),
    "low_contrast":            ("a flat low contrast image with soft even lighting", "a dramatic high contrast image"),
    "dark_background":         ("a thumbnail with a dark or black background", "a thumbnail with a bright or white background"),
    "bright_background":       ("a thumbnail with a bright white or very light background", "a thumbnail with a dark background"),
    "neon_colors":             ("neon glowing colors, electric blues, pinks, and greens", "natural muted or realistic colors"),
    "monochromatic":           ("a single color palette, black and white or monochrome", "a full color image with multiple hues"),
    "dramatic_lighting":       ("dramatic cinematic lighting with strong directional light and shadows", "flat even ambient lighting"),
    "golden_hour":             ("golden hour warm natural sunlight photography", "artificial indoor lighting or flat daylight"),
    # text & graphics
    "text_overlay":            ("a thumbnail with bold text words overlaid on the image", "a thumbnail with no text"),
    "large_text":              ("massive bold text taking up a large portion of the thumbnail", "small or no text on the image"),
    "minimal_text":            ("a thumbnail with very little or no text", "a thumbnail covered in large text"),
    "arrow_or_shape_overlay":  ("arrows, circles, or boxes drawn over the image highlighting something", "a clean image with no graphic overlays"),
    "brand_logo_present":      ("a channel logo or watermark visible in the corner", "no logo or watermark present"),
    "number_in_text":          ("text overlay containing a number like 10 or 100 or a statistic", "text with no numbers"),
    "question_in_text":        ("text overlay phrased as a question with a question mark", "text that is a statement not a question"),
    "emoji_or_symbol":         ("emoji symbols or special characters overlaid on the thumbnail", "plain text or no text overlays"),
    "progress_bar_or_meter":   ("a progress bar, meter, or ranking chart visible", "no data visualization elements"),
    # background & setting
    "clean_background":        ("a clean simple uncluttered background", "a busy cluttered complex background"),
    "busy_background":         ("a very busy cluttered complex background with many elements", "a clean simple background"),
    "outdoor_setting":         ("an outdoor location, nature, street, or exterior environment", "an indoor setting"),
    "indoor_setting":          ("an indoor setting, room, studio, or interior environment", "an outdoor location"),
    "studio_setting":          ("a professional studio backdrop or solid color background", "a natural real-world environment"),
    "nature_background":       ("trees, forests, mountains, water, or natural landscape background", "urban or indoor background"),
    "urban_background":        ("city streets, buildings, urban environment background", "nature or indoor background"),
    "blurred_background":      ("a bokeh blurred background with sharp subject in foreground", "a sharp in-focus background"),
    "underwater_setting":      ("underwater ocean or pool setting with blue water", "above water or dry land setting"),
    "space_setting":           ("outer space stars planets or space environment", "earth-based terrestrial setting"),
    # content-specific
    "food_visible":            ("food, meals, cooking, or beverages prominently shown", "no food or drinks visible"),
    "technology_visible":      ("computers, phones, gadgets, or electronics prominently shown", "no technology or devices visible"),
    "vehicle_visible":         ("cars, trucks, motorcycles, or other vehicles prominently shown", "no vehicles visible"),
    "animal_visible":          ("animals, pets, or wildlife prominently featured", "no animals present"),
    "money_visible":           ("money, cash, gold, or wealth symbols prominently shown", "no money or wealth symbols"),
    "fire_or_explosion":       ("fire, explosions, flames, or destruction shown", "no fire or explosive elements"),
    "luxury_items":            ("luxury goods, mansions, supercars, or high-end lifestyle", "ordinary everyday items and settings"),
    "sports_action":           ("athletic action, sports gameplay, physical movement", "static non-athletic content"),
    "gaming_content":          ("video game footage, gaming setup, or esports content", "non-gaming real world content"),
    "science_experiment":      ("scientific equipment, lab setting, or experiment in progress", "non-scientific everyday content"),
    "travel_destination":      ("iconic travel location, tourist destination, or exotic place", "ordinary local non-destination setting"),
    "crowd_or_audience":       ("a large crowd, audience, or group of many people", "a small group or individual"),
    "screen_recording":        ("a computer or phone screen being shown or recorded", "no screen or device display"),
    # thumbnail style
    "high_production_value":   ("a professional high production quality polished thumbnail", "an amateur low quality unpolished image"),
    "amateur_style":           ("an amateur casual unpolished low production value image", "a professional polished thumbnail"),
    "meme_format":             ("an internet meme format with impact font or reaction image", "a standard non-meme thumbnail"),
    "clickbait_style":         ("an extreme clickbait thumbnail with shocked face and sensational text", "a calm informational non-clickbait thumbnail"),
    "minimalist_style":        ("a clean minimalist thumbnail with lots of whitespace and simple design", "a busy maximalist design"),
    "illustrated_or_animated": ("illustrated, cartoon, animated, or graphic art style", "photographic real-world image"),
    "cinematic_style":         ("cinematic widescreen movie poster style composition", "casual snapshot or vlog style"),
    "retro_vintage_style":     ("retro vintage old-school aesthetic with grain or faded colors", "modern clean contemporary look"),
    "thumbnail_vs_layout":     ("a versus battle comparison layout with two things facing off", "a single subject with no comparison"),
}

AXIS_NAMES = list(AXES.keys())


@lru_cache(maxsize=1)
def _load_model() -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model, preprocess, device


@lru_cache(maxsize=1)
def _encode_text_prompts() -> tuple[torch.Tensor, torch.Tensor]:
    model, _, device = _load_model()
    pos_prompts = [AXES[ax][0] for ax in AXIS_NAMES]
    neg_prompts = [AXES[ax][1] for ax in AXIS_NAMES]
    with torch.no_grad():
        pos_tokens = clip.tokenize(pos_prompts).to(device)
        neg_tokens = clip.tokenize(neg_prompts).to(device)
        pos_feats = model.encode_text(pos_tokens).float()
        neg_feats = model.encode_text(neg_tokens).float()
        pos_feats /= pos_feats.norm(dim=-1, keepdim=True)
        neg_feats /= neg_feats.norm(dim=-1, keepdim=True)
    return pos_feats, neg_feats


def _pick_url(row: pd.Series) -> Optional[str]:
    for col in ("thumb_maxres_url", "thumb_standard_url", "thumb_high_url"):
        val = row.get(col)
        if pd.notna(val) and isinstance(val, str) and val.startswith("http"):
            return val
    return None


async def _fetch_one(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> Optional[bytes]:
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception:
            pass
    return None


async def _download_chunk(urls: list[str]) -> list[Optional[bytes]]:
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=DOWNLOAD_CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_one(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)


def _bytes_to_pil(data: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _score_images(images: list[Image.Image]) -> np.ndarray:
    model, preprocess, device = _load_model()
    pos_feats, neg_feats = _encode_text_prompts()
    scores = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            img_feats = model.encode_image(tensors).float()
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
        sim_pos = (img_feats @ pos_feats.T).cpu().numpy()
        sim_neg = (img_feats @ neg_feats.T).cpu().numpy()
        scores.append(np.clip((sim_pos - sim_neg + 2.0) / 4.0, 0.0, 1.0))
    return np.vstack(scores).astype(np.float32)


def _axis_weight(axis: str) -> float:
    return _LOW_CONFIDENCE_WEIGHT if axis in _LOW_CONFIDENCE_AXES else 1.0


def _derive_blueprint(scores: np.ndarray, vpd: np.ndarray) -> dict:
    log_vpd = np.log1p(vpd)
    q75 = np.percentile(log_vpd, 75)
    sample_weights = np.where(log_vpd >= q75, TOP_QUARTILE_WEIGHT, 1.0)
    blueprint = {}
    for i, axis in enumerate(AXIS_NAMES):
        axis_scores = scores[:, i]
        w = sample_weights
        w_mean = np.average(axis_scores, weights=w)
        vpd_mean = np.average(log_vpd, weights=w)
        cov = np.average((axis_scores - w_mean) * (log_vpd - vpd_mean), weights=w)
        std_s = np.sqrt(np.average((axis_scores - w_mean) ** 2, weights=w))
        std_v = np.sqrt(np.average((log_vpd - vpd_mean) ** 2, weights=w))
        corr = float(cov / (std_s * std_v + 1e-8))
        top_q_mask = log_vpd >= q75
        top_q_mean = float(axis_scores[top_q_mask].mean()) if top_q_mask.sum() > 0 else float(axis_scores.mean())
        blueprint[axis] = {
            "mean_score":        float(axis_scores.mean()),
            "top_quartile_mean": top_q_mean,
            "correlation":       round(corr * _axis_weight(axis), 4),
            "direction":         "positive" if corr > 0 else "negative",
            "axis_weight":       _axis_weight(axis),
            "n_videos":          int(len(axis_scores)),
        }
    return blueprint


def _download_and_score_chunked(urls: list[str]) -> tuple[np.ndarray, list[int]]:
    all_scores, valid_indices, scored, total = [], [], 0, len(urls)
    for chunk_start in range(0, total, CHUNK_SIZE):
        chunk_urls = urls[chunk_start : chunk_start + CHUNK_SIZE]
        raw_bytes = asyncio.run(_download_chunk(chunk_urls))
        images, local_valid = [], []
        for local_idx, data in enumerate(raw_bytes):
            if data:
                img = _bytes_to_pil(data)
                if img:
                    images.append(img)
                    local_valid.append(chunk_start + local_idx)
        if images:
            all_scores.append(_score_images(images))
            valid_indices.extend(local_valid)
            scored += len(images)
        print(f"Progress: {min(chunk_start + CHUNK_SIZE, total)}/{total} urls processed, {scored} scored")
    scores = np.vstack(all_scores) if all_scores else np.empty((0, len(AXIS_NAMES)), dtype=np.float32)
    return scores, valid_indices


def build_niche_blueprints(
    raw_csv: str = "data/processed/combined_videos_raw.csv",
    clean_csv: str = "data/processed/combined_videos_clean.csv",
    output_path: str = str(BLUEPRINT_PATH),
) -> dict:
    raw = pd.read_csv(raw_csv)
    clean = pd.read_csv(clean_csv)[["video_id", "category_name", "views_per_day"]].dropna()
    df = clean.merge(
        raw[["video_id", "thumb_maxres_url", "thumb_standard_url", "thumb_high_url"]],
        on="video_id", how="inner"
    )
    df = df[df["views_per_day"] > 0].reset_index(drop=True)
    urls = [_pick_url(row) for _, row in df.iterrows()]
    valid_url_mask = [u is not None for u in urls]
    valid_urls = [u for u in urls if u is not None]
    valid_orig_indices = [i for i, v in enumerate(valid_url_mask) if v]
    print(f"Scoring {len(valid_urls)} thumbnails in chunks of {CHUNK_SIZE}...")
    scores, scored_local_indices = _download_and_score_chunked(valid_urls)
    orig_indices = [valid_orig_indices[i] for i in scored_local_indices]
    scored_df = df.iloc[orig_indices].copy().reset_index(drop=True)
    blueprints = {}
    for category, group in scored_df.groupby("category_name"):
        idx = group.index.values
        blueprints[category] = _derive_blueprint(scores[idx], group["views_per_day"].values)
    blueprints["_global"] = _derive_blueprint(scores, scored_df["views_per_day"].values)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(blueprints, f, indent=2)
    print(f"Niche blueprints saved → {output_path}")
    return blueprints


def analyze_channel_thumbnails(
    video_ids: list[str],
    views_per_day: list[float],
    thumbnail_urls: list[str],
    category: str,
    blueprint_path: str = str(BLUEPRINT_PATH),
) -> dict:
    with open(blueprint_path) as f:
        niche_blueprints = json.load(f)
    niche_bp = niche_blueprints.get(category, niche_blueprints.get("_global"))
    valid = [
        (url, vpd) for url, vpd in zip(thumbnail_urls, views_per_day)
        if isinstance(url, str) and url.startswith("http") and vpd > 0
    ]
    if len(valid) < MIN_CHANNEL_VIDEOS:
        niche_bp["_source"] = "niche"
        niche_bp["_n_channel_videos"] = len(valid)
        return niche_bp
    urls, vpd_vals = zip(*valid)
    scores, valid_local = _download_and_score_chunked(list(urls))
    good_vpd = np.array([vpd_vals[i] for i in valid_local])
    if len(scores) < MIN_CHANNEL_VIDEOS:
        niche_bp["_source"] = "niche"
        niche_bp["_n_channel_videos"] = len(scores)
        return niche_bp
    channel_bp = _derive_blueprint(scores, good_vpd)
    channel_bp["_source"] = "channel"
    channel_bp["_n_channel_videos"] = len(scores)
    blend_weight = min(1.0, len(scores) / 100.0)
    for axis in AXIS_NAMES:
        if axis in channel_bp and axis in niche_bp:
            channel_bp[axis]["blended_correlation"] = round(
                blend_weight * channel_bp[axis]["correlation"] + (1 - blend_weight) * niche_bp[axis]["correlation"], 4
            )
    return channel_bp


def score_candidates(
    candidate_images: list[Image.Image],
    blueprint: dict,
) -> list[dict]:
    scores = _score_images(candidate_images)
    results = []
    for i, axis_scores in enumerate(scores):
        alignment, total_weight, axis_detail = 0.0, 0.0, {}
        for j, axis in enumerate(AXIS_NAMES):
            if axis not in blueprint:
                continue
            bp = blueprint[axis]
            corr = bp.get("blended_correlation", bp["correlation"])
            weight = abs(corr) * bp["axis_weight"]
            score = float(axis_scores[j])
            contribution = corr * score if corr > 0 else abs(corr) * (1.0 - score)
            alignment += contribution * weight
            total_weight += weight
            axis_detail[axis] = round(score, 4)
        results.append({
            "candidate_index": i,
            "alignment_score": round(alignment / (total_weight + 1e-8), 4),
            "axis_scores":     axis_detail,
        })
    results.sort(key=lambda x: x["alignment_score"], reverse=True)
    return results


def get_blueprint_summary(blueprint: dict) -> dict:
    axis_entries = [
        (axis, data) for axis, data in blueprint.items()
        if isinstance(data, dict) and "correlation" in data
    ]
    axis_entries.sort(
        key=lambda x: abs(x[1].get("blended_correlation", x[1]["correlation"])),
        reverse=True
    )
    top_positive = [ax for ax, d in axis_entries if d.get("blended_correlation", d["correlation"]) > 0.05][:8]
    top_negative = [ax for ax, d in axis_entries if d.get("blended_correlation", d["correlation"]) < -0.05][:4]
    return {
        "source":    blueprint.get("_source", "unknown"),
        "n_videos":  blueprint.get("_n_channel_videos", blueprint.get(AXIS_NAMES[0], {}).get("n_videos", 0)),
        "recommend": top_positive,
        "avoid":     top_negative,
        "top_axes":  {ax: blueprint[ax] for ax in top_positive + top_negative if ax in blueprint},
    }