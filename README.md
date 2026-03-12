# YouTube IP V3

YouTube IP V3 is a Streamlit application for channel benchmarking, content analysis, and AI-assisted planning. It combines prebuilt cross-channel datasets with live YouTube API pulls so you can analyze performance patterns, generate recommendations, and prototype creator assets in one interface.

## What The App Includes

- Channel Analysis for portfolio-level trends across the bundled datasets
- Recommendations for publish timing, title patterns, and keyword angles
- Ytuber Creator Suite for live channel audits, outlier discovery, competitor benchmarking, SEO scoring, content planning, and AI generation
- Gemini and OpenAI integrations for text and thumbnail workflows

## Repository Layout

```text
.
├── dashboard/                 # Streamlit UI and page views
├── data/youtube api data/     # Bundled CSV datasets used by the analytics views
├── docs/                      # Architecture and project brief
├── outputs/                   # Generated thumbnails and derived artifacts
├── scripts/                   # Dataset-building and API smoke-test scripts
├── src/                       # Partial package scaffolding
├── streamlit_app.py           # Root Streamlit Cloud entrypoint
└── requirements.txt           # Python dependencies
```

## Local Setup

### Prerequisites

- Python 3.10+
- `YOUTUBE_API_KEYS` or `YOUTUBE_API_KEY` for live channel analysis
- `GEMINI_API_KEYS` or `GEMINI_API_KEY` for Gemini generation
- `OPENAI_API_KEYS` or `OPENAI_API_KEY` if you want OpenAI / ChatGPT fallback text and image generation

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Secrets For Local Development

```bash
cp .env.example .env
```

Populate `.env` with:

- `YOUTUBE_API_KEYS`
- `GEMINI_API_KEYS`
- `OPENAI_API_KEYS`

Comma-separated lists are supported for local development, for example:

```bash
YOUTUBE_API_KEYS=key_1,key_2
GEMINI_API_KEYS=key_1,key_2
OPENAI_API_KEYS=key_1,key_2
```

Single-key fallbacks also work with `YOUTUBE_API_KEY`, `GEMINI_API_KEY`, and `OPENAI_API_KEY`.

### Run The App

```bash
streamlit run streamlit_app.py
```

The original module entrypoint also works:

```bash
streamlit run dashboard/app.py
```

## Dashboard Pages

### Channel Analysis

Located in `dashboard/views/channel_analysis.py`.

Uses the committed CSV datasets under `data/youtube api data/` to compare channels, inspect upload trends, review top-performing videos, and visualize engagement patterns.

### Recommendations

Located in `dashboard/views/recommendations.py`.

Uses dataset-backed performance patterns to suggest keyword angles, title length targets, publish timing, and reference videos. It also includes the thumbnail studio for Gemini/OpenAI image generation.

### Ytuber

Located in `dashboard/views/ytuber.py`.

Uses live YouTube API pulls for a single channel, rotates across the configured background API-key pools, and exposes:

- AI Studio
- Overview
- Channel Audit
- Keyword Intel
- Outliers Finder
- Title and SEO Lab
- Competitor Benchmark
- Content Planner

Ytuber now links out to the standalone Outlier Finder page instead of hosting the full research workflow inside the channel workspace.

### Outlier Finder

Located in `dashboard/views/outlier_finder.py`.

Outlier Finder is a standalone sidebar feature for niche research. It scans the public cohort returned by the official YouTube Data API, applies stricter filtering for language, duration, subscriber range, and freshness, then ranks videos by age-adjusted overperformance.

It is designed for practical research, not exhaustive YouTube-wide ranking. Results are limited by official API search coverage, public metrics, language heuristics, and your configured API quota.

## Streamlit Deployment

This repo is ready to deploy from GitHub to Streamlit Community Cloud.

### Streamlit Cloud App Settings

- Repo: `royayushkr/Youtube-IP-V3`
- Branch: `main`
- Main file path: `streamlit_app.py`

### Required Secrets

Add these in the Streamlit app Secrets panel:

```toml
YOUTUBE_API_KEYS = ["your_youtube_key_1", "your_youtube_key_2"]
GEMINI_API_KEYS = ["your_gemini_key_1", "your_gemini_key_2"]
OPENAI_API_KEYS = ["your_openai_key_1", "your_openai_key_2"]
```

You can also use single-key fallbacks with `YOUTUBE_API_KEY`, `GEMINI_API_KEY`, and `OPENAI_API_KEY`.
You can copy `.streamlit/secrets.toml.example` for local reference.

### Deployment Notes

- `streamlit_app.py` is the recommended root entrypoint for deployment.
- `dashboard/app.py` remains the main application module.
- Channel Analysis and Recommendations work from the committed datasets.
- The Ytuber page uses the configured key pools to serve live channel data without exposing raw keys in the UI.
- Outliers Finder uses the same YouTube key pool, caches niche scans for one hour, and caches channel baselines for six hours to control quota burn.
- Outliers Finder also offers structured Gemini/OpenAI research cards when AI keys are configured.
- Thumbnail and text generation features use the configured Gemini and/or OpenAI key pools.
- OpenAI thumbnail generation uses the official Images API flow and now exposes model, size, quality, background, and output-format controls in AI Studio.

## Supporting Files

- `.streamlit/config.toml` controls the app theme
- `.env.example` documents local environment variables
- `docs/ARCHITECTURE.md` describes the high-level data flow
- `scripts/` contains dataset builders and API smoke tests

## License

MIT License. See `LICENSE`.
