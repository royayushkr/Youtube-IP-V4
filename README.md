# YouTube IP V4

YouTube IP V4 is a Streamlit application for YouTube research, benchmarking, live channel analysis, outlier discovery, creator workflow tooling, and AI-assisted planning. It combines bundled CSV datasets with live YouTube Data API requests and optional Gemini/OpenAI generation so one app can cover historical benchmarking, channel diagnostics, idea research, creator support, and creative asset prototyping.

Live app:

- [youtube-ip-v4.streamlit.app](https://youtube-ip-v4.streamlit.app/)

Deployment target depends on the repo and branch you choose. Use the deployment section below for the current Streamlit setup.

This README documents the current deployed app as it exists in this repository, including:

- what the product does
- which files power each feature
- how the app is wired together
- what data sources and API keys it uses
- how to run it locally
- how to deploy it on Streamlit Community Cloud
- what parts of the repo are active versus legacy scaffolding
- how the retrieval-first assistant caches and reuses answers before calling AI

## Product Overview

The app currently exposes seven sidebar destinations:

| Page | Purpose | Main File |
| --- | --- | --- |
| `Channel Analysis` | Portfolio-level analytics across the bundled datasets | `dashboard/views/channel_analysis.py` |
| `Recommendations` | Dataset-backed publishing guidance and thumbnail generation | `dashboard/views/recommendations.py` |
| `Ytuber` | Live creator workspace for one channel at a time | `dashboard/views/ytuber.py` |
| `Channel Insights` | Persisted channel snapshots with public analysis and optional owner-only analytics overlays via Google OAuth | `dashboard/views/channel_insights.py` |
| `Outlier Finder` | Standalone niche research and outlier-video discovery | `dashboard/views/outlier_finder.py` |
| `Tools` | Standalone utility workspace for YouTube metadata and asset downloads | `dashboard/views/tools.py` |
| `Deployment` | Run/deploy notes shown inside the app | `dashboard/app.py` |

In addition to the sidebar pages, the app now includes a **global Assistant** in the sidebar. It is available across the product and is designed to answer product-help, troubleshooting, metric-interpretation, and creator-workflow questions with a retrieval-first stack before it escalates to Gemini or OpenAI.

At a high level, the app is designed for three use cases:

1. Analyze existing cross-channel datasets to understand benchmark patterns.
2. Pull live stats for a public channel and turn them into creator-focused diagnostics.
3. Persist public-channel snapshots and compare topic, format, and outlier patterns over time.
4. Generate strategy and creative suggestions with Gemini or OpenAI using the same public data.
5. Export public YouTube assets such as thumbnails, transcripts, audio, and video from one utility page.

## What The App Includes

### 1. Channel Analysis

`Channel Analysis` is the dataset-backed analytics view.

It can:

- load one category dataset or all committed datasets together
- filter by channel and published-date range
- show KPI summaries for videos, channels, views, average views, and median engagement
- surface top channels by total views
- chart monthly upload trends
- list best-performing videos
- compare publishing-day performance
- visualize views versus engagement

Code:

- `dashboard/views/channel_analysis.py`
- `dashboard/components/visualizations.py`

Data source:

- committed CSV files in `data/youtube api data/`

### 2. Recommendations

`Recommendations` turns the same bundled datasets into lightweight strategy guidance.

It can:

- benchmark a selected category or all categories
- compute a high-performing sample from the top quartile of videos
- suggest publish timing and title length targets
- extract keyword angles from strong titles
- show reference videos to model
- generate thumbnail concepts with Gemini or OpenAI

Code:

- `dashboard/views/recommendations.py`
- `src/llm_integration/thumbnail_generator.py`

### 3. Ytuber

`Ytuber` is the live creator workspace for one public channel.

It can:

- resolve a handle, channel name, or channel ID
- pull fresh channel and recent-video metadata from the YouTube Data API
- cache channel fetches in the local CSV-backed dataset
- compute a channel overview and audit
- generate keyword intelligence from recent uploads
- score titles and descriptions in `Title And SEO Lab`
- benchmark competitors and generate comparative recommendations
- plan content around day/hour performance patterns
- run `AI Studio` for titles, ideas, scripts, clips, and thumbnail generation
- hand off into the standalone `Outlier Finder`

Key modules inside the page:

- `AI Studio`
- `Overview`
- `Channel Audit`
- `Keyword Intel`
- `Outliers Finder` shortcut
- `Title And SEO Lab`
- `Competitor Benchmark`
- `Content Planner`

Code:

- `dashboard/views/ytuber.py`
- `src/utils/api_keys.py`
- `src/llm_integration/thumbnail_generator.py`

### 4. Channel Insights

`Channel Insights` is the recurring creator-intelligence workflow for tracked channels.

It can:

- add a public channel by URL, handle, or channel ID
- connect a Google account for owner-only YouTube Analytics access during the session
- store tracked channels in a local SQLite database
- persist dated channel snapshots on refresh
- compare current public performance against prior snapshots
- blend in owner-only watch time, average percentage viewed, thumbnail impressions, and thumbnail click-through signals when the authenticated account owns the tracked channel
- surface rising and weak themes from recent public uploads
- compare Shorts versus long-form and duration buckets
- identify outliers and underperformers within the channel
- recommend what to double down on, what to avoid, and what to test next
- generate grounded video-direction suggestions from actual channel data
- optionally use a beta BERTopic-backed topic mode when an external model manifest and bundle are configured

Code:

- `dashboard/views/channel_insights.py`
- `src/services/public_channel_service.py`
- `src/services/channel_snapshot_store.py`
- `src/services/channel_insights_service.py`
- `src/services/google_oauth_service.py`
- `src/services/topic_analysis_service.py`
- `src/services/youtube_owner_analytics_service.py`
- `src/services/channel_idea_service.py`
- `src/services/model_artifact_service.py`
- `src/services/topic_model_runtime.py`
- `src/utils/channel_parser.py`

Storage:

- `outputs/channel_insights/channel_insights.db`
- optional BERTopic runtime cache: `outputs/models/runtime/`

Google OAuth setup for owner metrics:

- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REDIRECT_URI`

Scopes requested:

- Google Sign-In
- YouTube Read-Only
- YouTube Analytics Read-Only

### 5. Outlier Finder

`Outlier Finder` is a standalone niche-research page in the sidebar. It is designed to find videos that are overperforming relative to channel size, age, peers, or channel baseline within the scanned cohort returned by the official YouTube API.

It supports:

- niche / keyword search
- timeframe filters
- region and language filters
- language strictness
- duration preference
- minimum views
- subscriber bucket and explicit min/max subscriber filters
- include/exclude hidden subscriber counts
- exact-phrase versus broad matching
- exclude keywords
- bounded search depth and baseline-enrichment settings

Its results-first workflow is:

1. `Top Outliers In This Scan`
2. `Breakout Snapshot`
3. `AI Research`
4. `How This Works`

The page also includes:

- sortable outlier results
- explanation strings for why each video is an outlier
- score and scan summary cards
- breakout charts for age, duration, title pattern, and language quality
- structured AI report cards via Gemini/OpenAI
- an inline methodology section explaining metrics and caveats

Code:

- `dashboard/views/outlier_finder.py`
- `src/services/outliers_finder.py`
- `src/services/outlier_ai.py`

### 6. Tools

`Tools` is a standalone utility page for public YouTube asset retrieval.

It supports:

- single-video metadata preview
- batch URL processing
- public playlist preview with selected-item operations
- thumbnail preview and export
- transcript language discovery and `.txt` export
- audio download
- video download
- quality/format selection for single videos
- profile-based audio/video choices for batch and playlist workflows

The page is designed around three modes:

- `Single`
- `Batch`
- `Playlist`

Code:

- `dashboard/views/tools.py`
- `src/services/youtube_tools.py`
- `src/services/transcript_service.py`
- `src/utils/file_utils.py`

### 7. Assistant

The sidebar `Assistant` is a retrieval-first help and creator-support layer.

It can:

- answer product usage questions
- explain metrics and caveats
- suggest creator workflows
- help troubleshoot missing results, failed exports, or unavailable AI features
- reuse prior high-confidence answers before making any paid AI call
- fall back to Gemini or OpenAI only when retrieval is insufficient

Core implementation:

- `dashboard/components/assistant_panel.py`
- `src/services/assistant_service.py`
- `src/services/retrieval_service.py`
- `src/services/cache_service.py`
- `src/services/assistant_knowledge.py`
- `src/utils/text_normalization.py`

Knowledge and storage:

- `data/assistant/*.json`
- `outputs/assistant/assistant_cache.db`

## Current Runtime Architecture

### App Entrypoints

There are two Streamlit entrypoints:

- `streamlit_app.py`
  - root deployment entrypoint used by Streamlit Cloud
  - simply imports `dashboard.app`
- `dashboard/app.py`
  - real application shell
  - configures Streamlit page settings
  - injects the shared theme
  - renders the sidebar
  - routes to each page

### Shared UI Layer

- `dashboard/components/sidebar.py`
  - branded sidebar navigation using `streamlit-option-menu`
- `dashboard/components/theme.py`
  - shared app theme, CSS tokens, page widths, button styling, and general chrome
- `dashboard/components/visualizations.py`
  - reusable Plotly chart helpers, dataframe styling, keyword chips, KPI rows, and section headers
- `dashboard/components/assistant_panel.py`
  - sidebar assistant UI, starter prompts, answer cards, and feedback controls

### Active Service Layer

The current active backend logic is concentrated in a small number of files:

- `src/utils/api_keys.py`
  - reads API keys from environment variables and Streamlit secrets
  - supports single-key and pooled-key modes
  - rotates keys per provider in session state
  - retries operations across configured keys

- `src/services/outliers_finder.py`
  - core outlier-search request and scoring engine
  - YouTube API orchestration for search, videos, channels, and baseline fetches
  - language confidence heuristics
  - duration and age bucketing
  - peer percentile and baseline-based scoring
  - cache wrappers for niche scans and channel baselines

- `src/services/outlier_ai.py`
  - converts outlier results into structured AI research cards
  - calls Gemini or OpenAI
  - expects JSON output and falls back gracefully if parsing fails

- `src/services/public_channel_service.py`
  - shared public-channel fetch layer reused by `Ytuber` and `Channel Insights`
  - resolves handles / channel IDs
  - reuses the local CSV-backed cache plus live YouTube Data API refreshes

- `src/services/channel_snapshot_store.py`
  - SQLite-backed persistence for tracked channels and dated channel snapshots

- `src/services/channel_insights_service.py`
  - channel refresh orchestration
  - baseline computation
  - topic/format/outlier insight payload generation

- `src/services/topic_analysis_service.py`
  - title-pattern classification
  - heuristic topic clustering
  - duration and timing aggregations

- `src/services/channel_idea_service.py`
  - grounded “double down / avoid / test next” suggestions
  - optional AI explanation layer on top of structured metrics

- `src/services/assistant_service.py`
  - top-level assistant orchestration
  - intent detection
  - page-context snapshots
  - exact-cache -> semantic-retrieval -> knowledge -> hybrid -> LLM routing

- `src/services/retrieval_service.py`
  - local TF-IDF retrieval over curated knowledge and cached historical answers

- `src/services/cache_service.py`
  - SQLite-backed answer cache and feedback storage

- `src/services/assistant_knowledge.py`
  - JSON knowledge loading for FAQs, metric definitions, troubleshooting, and workflow guidance

- `src/llm_integration/thumbnail_generator.py`
  - Gemini and OpenAI image-generation wrapper
  - used by the Recommendations page and `Ytuber -> AI Studio`

### Data Flow

There are two main data flows in the app:

#### A. Dataset-backed analytics

```text
Bundled CSV datasets
-> pandas loading/cleaning in page views
-> dashboard/components/visualizations.py
-> Channel Analysis / Recommendations UI
```

#### B. Live API-backed creator workflows

```text
Streamlit secrets / env vars
-> src/utils/api_keys.py
-> YouTube API or Gemini/OpenAI calls
-> page-specific transformations in Ytuber / Outlier Finder
-> charts, result cards, and AI panels in the Streamlit UI
```

#### C. Retrieval-first assistant workflow

```text
User question
-> query normalization
-> exact cache lookup (SQLite)
-> semantic similarity search (TF-IDF over cached answers)
-> structured knowledge retrieval (JSON knowledge base)
-> hybrid deterministic response when possible
-> Gemini/OpenAI only when retrieval is insufficient
-> cache new answer + collect helpful / not-helpful feedback
```

## Repository Map

This is the practical repository layout, not just the nominal one:

```text
.
├── dashboard/
│   ├── app.py                       # Main Streamlit router
│   ├── components/
│   │   ├── sidebar.py               # Sidebar navigation
│   │   ├── theme.py                 # Shared dark/purple theme
│   │   └── visualizations.py        # Plotly + dataframe helpers
│   └── views/
│       ├── channel_analysis.py      # Dataset analytics page
│       ├── channel_insights.py      # Persisted public-channel insights page
│       ├── recommendations.py       # Recommendations + thumbnail studio
│       ├── ytuber.py                # Live creator workspace
│       ├── outlier_finder.py        # Standalone niche research page
│       └── tools.py                 # Standalone YouTube tools page
├── data/
│   └── youtube api data/            # Bundled CSV datasets used by the app
│   └── assistant/                   # Curated assistant knowledge records
├── docs/
│   ├── ARCHITECTURE.md              # Runtime-first architecture note
│   └── PROJECT_BRIEF.md             # Original project brief
├── outputs/
│   ├── assistant/                   # SQLite cache for assistant answers/feedback
│   ├── channel_insights/            # SQLite snapshot store for tracked channels
│   └── thumbnails/                  # Generated thumbnail outputs (gitignored)
├── research_archive/                # Historical research code, notebooks, docs, and datasets
├── scripts/
│   ├── yt_api_smoketest.py          # Rich YouTube API smoke test
│   ├── build_*_dataset.py           # Dataset builder scripts
│   └── available_data_constraints.md
├── src/
│   ├── services/                    # Active outlier, tools, and AI service layer
│   ├── utils/                       # API-key management, file helpers, and shared utilities
│   └── llm_integration/             # Thumbnail generation wrapper
├── tests/
│   ├── integration/                 # Integration tests
│   └── unit/                        # Unit tests
├── streamlit_app.py                 # Root Streamlit Cloud entrypoint
├── requirements.txt                 # Python dependencies
└── .streamlit/config.toml           # Theme config
```

## What Is Active Versus Archived Research Material

This repo has evolved over time. The currently deployed app does **not** use every folder equally.

### Actively used by the app today

- `dashboard/`
- `src/services/`
- `src/services/public_channel_service.py`
- `src/services/channel_snapshot_store.py`
- `src/services/channel_insights_service.py`
- `src/services/topic_analysis_service.py`
- `src/services/channel_idea_service.py`
- `src/utils/api_keys.py`
- `src/utils/channel_parser.py`
- `src/utils/file_utils.py`
- `src/llm_integration/thumbnail_generator.py`
- `dashboard/components/assistant_panel.py`
- `src/services/assistant_service.py`
- `src/services/retrieval_service.py`
- `src/services/cache_service.py`
- `src/services/assistant_knowledge.py`
- `src/utils/text_normalization.py`
- `data/youtube api data/`
- `data/assistant/`
- `outputs/channel_insights/`
- `outputs/assistant/`
- `tests/unit/test_outliers_finder.py`
- `tests/unit/test_outlier_ai.py`
- `tests/integration/test_pipeline.py`
- `tests/unit/test_text_normalization.py`
- `tests/unit/test_cache_service.py`
- `tests/unit/test_retrieval_service.py`
- `tests/unit/test_assistant_service.py`
- `tests/integration/test_assistant_flow.py`

## Assistant Retrieval And Caching Flow

The Assistant is intentionally retrieval-first to reduce token cost and improve speed.

### Layer order

1. **Exact cache lookup**
   - normalized question + page scope + context mode
   - reuses answers younger than 30 days when confidence is strong and feedback is not negative
2. **Semantic cache lookup**
   - local TF-IDF cosine similarity across prior resolved answers
   - allows near-duplicate reuse without paying for embeddings
3. **Structured knowledge retrieval**
   - uses curated JSON knowledge files for product help, metrics, troubleshooting, and workflows
4. **Hybrid deterministic answer**
   - combines cached answers and knowledge into a structured response without calling an LLM
5. **LLM fallback**
   - Gemini first, OpenAI fallback
   - used only when retrieval is insufficient, especially for creator strategy or contextual interpretation

### Storage

- Cache DB: `outputs/assistant/assistant_cache.db`
- Knowledge files: `data/assistant/*.json`

### What gets stored

- original query
- normalized query
- page scope
- context mode
- answer text
- answer source type (`exact_cache`, `semantic_cache`, `knowledge`, `hybrid`, `llm`)
- confidence
- source references
- related questions
- page-context snapshot
- model/provider metadata when generation occurs
- helpful / not-helpful feedback counts

### Current limitations

- SQLite persistence is local and may not survive all Streamlit Cloud redeploys
- TF-IDF retrieval is deliberately lightweight and cheaper than embeddings, but it is weaker on heavy paraphrases
- the strongest context support today is on `Outlier Finder`, `Ytuber`, and `Tools`
- creator-strategy answers may still require AI fallback when product knowledge is not enough
- Channel Insights supports owner-only analytics only when Google OAuth is configured and the connected Google account actually owns the tracked channel
- OAuth state is session-scoped in V1, so users may need to reconnect after a Streamlit restart or redeploy
- There is still no authenticated YouTube Analytics scheduling or background refresh worker in Streamlit alone
- BERTopic beta mode is optional, off by default, and falls back to heuristic topics whenever artifact setup or runtime loading fails

### Archived for reference

Historical research material now lives under `research_archive/`, including:

- legacy ML and data-collection code
- research notebooks
- processed modeling datasets
- ML backend notes and supporting docs
- extra raw datasets not needed by the deployed Streamlit app

This keeps the runtime tree app-focused without deleting the older research work.

### Branch guardrail: `asher` is not a deployable feature branch

The `asher` branch is intentionally **not** merged into `youtube-ip-v4` or deployment `main`.

Current reason:

- its branch diff only adds Git LFS tracking and BERTopic model artifact pointers under `outputs/models/`
- it does **not** add active Streamlit runtime code, routes, services, tests, or deployment fixes
- those branch-specific model artifacts are still **not** merged into V4; the deployed app only supports BERTopic through the new optional external bundle workflow and would become heavier and more fragile if the `asher` Git LFS artifacts were merged directly

Deployment rule:

- keep `youtube-ip-v4` as the source of truth for routing, entrypoints, secrets, and Streamlit deployability
- do not introduce Git LFS model artifacts into the deploy branch
- model-backed topic support must remain **optional**, externally hosted, graceful when artifacts are missing, and inactive during app boot

## Bundled Data Assets

The repository currently ships with four CSV datasets under `data/youtube api data/`.

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| `entertainment_channels_videos.csv` | 101,554 | 54 |
| `gaming_channels_videos.csv` | 95,534 | 54 |
| `research_science_channels_videos.csv` | 221,325 | 54 |
| `tech_channels_videos.csv` | 125,693 | 54 |

Total bundled rows: **544,106**

These datasets power:

- `Channel Analysis`
- the dataset-backed parts of `Recommendations`
- parts of the `Ytuber` page when appending live fetches into the working CSV-backed flow

## Secrets, Environment Variables, And API-Key Pools

The app supports both single keys and pooled keys.

Supported provider groups:

- `youtube`
- `gemini`
- `openai`

### Preferred pooled-key format

Environment variables:

```bash
YOUTUBE_API_KEYS=key_1,key_2
GEMINI_API_KEYS=key_1,key_2
OPENAI_API_KEYS=key_1,key_2
```

Streamlit secrets:

```toml
YOUTUBE_API_KEYS = ["key_1", "key_2"]
GEMINI_API_KEYS = ["key_1", "key_2"]
OPENAI_API_KEYS = ["key_1", "key_2"]
```

### Supported single-key fallbacks

- `YOUTUBE_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

### Optional Google OAuth settings for owner analytics

`Channel Insights` can also use session-scoped Google OAuth for owner-only YouTube Analytics metrics.

Set these in environment variables or Streamlit secrets:

- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REDIRECT_URI`
- `MODEL_ARTIFACTS_ENABLED` (optional)
- `MODEL_ARTIFACTS_MANIFEST_URL` (optional)

Example Streamlit secrets:

```toml
GOOGLE_OAUTH_CLIENT_ID = "your-google-oauth-client-id"
GOOGLE_OAUTH_CLIENT_SECRET = "your-google-oauth-client-secret"
GOOGLE_OAUTH_REDIRECT_URI = "https://your-app.streamlit.app/"

# Optional BERTopic beta settings for Channel Insights
# MODEL_ARTIFACTS_ENABLED = true
# MODEL_ARTIFACTS_MANIFEST_URL = "https://raw.githubusercontent.com/royayushkr/Youtube-IP-V4/main/data/model_manifests/bertopic_manifest_2026.03.27.json"
# MODEL_ARTIFACTS_CACHE_DIR = "outputs/models/runtime"
# MODEL_ARTIFACTS_DOWNLOAD_TIMEOUT_SECONDS = 300
# MODEL_ARTIFACTS_MAX_SIZE_MB = 512
```

Important:

- the redirect URI must exactly match one of the authorized redirect URIs in your Google Cloud OAuth client
- V1 stores Google OAuth credentials in session state, not a long-lived encrypted credential store

### How key rotation works

`src/utils/api_keys.py` does the following:

- reads values from Streamlit secrets first, then environment variables
- accepts JSON-style lists, comma-separated strings, line-delimited strings, or indexed secret names
- deduplicates the final list
- stores a session-level cursor for each provider
- retries operations across all configured keys when failures are retryable
- leaves Channel Insights on heuristic topics when the optional BERTopic settings are absent

This matters most for:

- live YouTube fetches in `Ytuber`
- outlier scans in `Outlier Finder`
- Gemini/OpenAI generation in `AI Studio`, `Recommendations`, and Outlier AI reports

## Local Development

### Prerequisites

- Python 3.10 or newer
- `ffmpeg` for merged video downloads and MP3 conversion in the `Tools` page
- valid YouTube Data API credentials for live features
- Gemini and/or OpenAI credentials for AI features

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are running locally outside Streamlit Community Cloud, make sure `ffmpeg` is on your system path when you want:

- merged video downloads in `Tools`
- MP3 conversion in `Tools`

### Configure local secrets

Copy:

```bash
cp .env.example .env
```

Then populate:

- `YOUTUBE_API_KEYS`
- `GEMINI_API_KEYS`
- `OPENAI_API_KEYS`

Example:

```bash
YOUTUBE_API_KEYS=your_youtube_key_1,your_youtube_key_2
GEMINI_API_KEYS=your_gemini_key_1,your_gemini_key_2
OPENAI_API_KEYS=your_openai_key_1,your_openai_key_2
```

Local Streamlit-style secrets are also supported via `.streamlit/secrets.toml`.

Reference file:

- `.streamlit/secrets.toml.example`

Optional BERTopic beta env vars are also supported:

- `MODEL_ARTIFACTS_ENABLED`
- `MODEL_ARTIFACTS_MANIFEST_URL`
- `MODEL_ARTIFACTS_CACHE_DIR`
- `MODEL_ARTIFACTS_DOWNLOAD_TIMEOUT_SECONDS`
- `MODEL_ARTIFACTS_MAX_SIZE_MB`

### Run the app

Preferred:

```bash
streamlit run streamlit_app.py
```

Alternate:

```bash
streamlit run dashboard/app.py
```

## Streamlit Community Cloud Deployment

This repo is structured to deploy directly from GitHub to Streamlit Community Cloud.

### Streamlit app settings

- Live app: [youtube-ip-v4.streamlit.app](https://youtube-ip-v4.streamlit.app/)
- Repo: `royayushkr/Youtube-IP-V4`
- Branch: `main`
- Main file path: `streamlit_app.py`

### Where To Add Secrets In Streamlit Cloud

You can add secrets in either place:

1. During first deploy:
   - `New App` -> `Advanced Settings` -> `Secrets`
2. After the app already exists:
   - open the app in Streamlit Community Cloud
   - `Manage App` -> `Settings` -> `Secrets`

If you are deploying locally, use `.streamlit/secrets.toml` instead.

### Recommended Streamlit Secrets Block

Paste this into the Streamlit `Secrets` editor and replace the placeholder values:

```toml
YOUTUBE_API_KEYS = ["your_youtube_key_1", "your_youtube_key_2"]
GEMINI_API_KEYS = ["your_gemini_key_1", "your_gemini_key_2"]
OPENAI_API_KEYS = ["your_openai_key_1", "your_openai_key_2"]

GOOGLE_OAUTH_CLIENT_ID = "your-google-oauth-client-id"
GOOGLE_OAUTH_CLIENT_SECRET = "your-google-oauth-client-secret"
GOOGLE_OAUTH_REDIRECT_URI = "https://your-app-name.streamlit.app/"

# Optional BERTopic beta settings
# MODEL_ARTIFACTS_ENABLED = true
# MODEL_ARTIFACTS_MANIFEST_URL = "https://raw.githubusercontent.com/royayushkr/Youtube-IP-V4/main/data/model_manifests/bertopic_manifest_2026.03.27.json"
```

Single-key fallbacks still work if needed.

### Google OAuth Setup For Channel Insights

`Channel Insights` can optionally connect a Google account and use owner-only YouTube Analytics metrics during the session.

That flow needs all three of these secrets:

- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REDIRECT_URI`

Important setup rules:

- the redirect URI must exactly match one of the authorized redirect URIs in your Google Cloud OAuth client
- for Streamlit Community Cloud, the safest V1 redirect URI is usually your deployed app root URL, for example:
  - `https://your-app-name.streamlit.app/`
- if you test locally, also add your local URL to Google Cloud, for example:
  - `http://localhost:8501/`

Suggested Google Cloud setup:

1. Open Google Cloud Console
2. Create or select a project
3. Enable:
   - `YouTube Data API v3`
   - `YouTube Analytics API`
4. Create an OAuth client for a web application
5. Add the authorized redirect URIs you will use
6. Copy the client ID and client secret into Streamlit secrets

What the current V4 app requests:

- Google Sign-In
- YouTube Read-Only
- YouTube Analytics Read-Only

What V4 does not request:

- upload permissions
- channel management permissions
- destructive account scopes

### Local Secrets Example

If you are running locally, create `.streamlit/secrets.toml` and use:

```toml
YOUTUBE_API_KEYS = ["your_youtube_key_1", "your_youtube_key_2"]
GEMINI_API_KEYS = ["your_gemini_key_1", "your_gemini_key_2"]
OPENAI_API_KEYS = ["your_openai_key_1", "your_openai_key_2"]

GOOGLE_OAUTH_CLIENT_ID = "your-google-oauth-client-id"
GOOGLE_OAUTH_CLIENT_SECRET = "your-google-oauth-client-secret"
GOOGLE_OAUTH_REDIRECT_URI = "http://localhost:8501/"

# Optional BERTopic beta settings
# MODEL_ARTIFACTS_ENABLED = true
# MODEL_ARTIFACTS_MANIFEST_URL = "https://raw.githubusercontent.com/royayushkr/Youtube-IP-V4/main/data/model_manifests/bertopic_manifest_2026.03.27.json"
```

### Optional BERTopic Beta For Channel Insights

`Channel Insights` now supports two topic modes:

- `Heuristic Topics`
- `Model-Backed Topics (Beta)`

The beta mode is deployment-safe by design:

- it is off by default
- it uses an external manifest and GitHub Release asset instead of Git LFS or repo-committed model binaries
- it downloads the bundle only when a user explicitly requests the beta mode during a refresh
- it falls back to heuristic topics if the manifest, download, checksum, load, or transform step fails

#### Required optional settings

- `MODEL_ARTIFACTS_ENABLED=true`
- `MODEL_ARTIFACTS_MANIFEST_URL=https://.../bertopic_manifest_<version>.json`

#### Optional advanced settings

- `MODEL_ARTIFACTS_CACHE_DIR`
- `MODEL_ARTIFACTS_DOWNLOAD_TIMEOUT_SECONDS`
- `MODEL_ARTIFACTS_MAX_SIZE_MB`

#### Manifest schema

The BERTopic manifest must contain:

- `bundle_version`
- `artifact_url`
- `sha256`
- `size_bytes`
- `model_type` (`bertopic_global`)
- `bertopic_version`
- `python_version`
- `load_subpath`

#### Maintainer packaging workflow

Use the release packaging script:

The current starter manifest in this repo lives at:

- `data/model_manifests/bertopic_manifest_2026.03.27.json`

That manifest currently points at the public BERTopic artifact served from:

- `https://github.com/matt-foor/purdue-youtube-ip/raw/asher/outputs/models/bertopic_model`

If you want to replace that with your own hosted bundle later, the maintainer packaging script is:

```bash
python3 scripts/package_bertopic_release.py \
  --model-path /path/to/bertopic_model.pkl \
  --output-dir dist/bertopic_release \
  --bundle-version 2026.03.27 \
  --bertopic-version 0.16.4 \
  --repo royayushkr/Youtube-IP-V4 \
  --tag v0.1.0-beta \
  --github-token "$GITHUB_TOKEN"
```

The script:

- packages the local BERTopic artifact into a zip bundle
- verifies the bundle with `BERTopic.load(...)` unless `--skip-verify` is used
- writes a checksum-based manifest JSON
- uploads the bundle and manifest to the selected GitHub Release when repo/tag/token are supplied

Important rule:

- do not enable deployed BERTopic inference until the packaged bundle loads cleanly in a fresh environment without relying on hidden local caches or surprise model downloads

### Theme

The live app theme is defined in `.streamlit/config.toml`:

- `primaryColor = "#8B5CF6"`
- `backgroundColor = "#090B14"`
- `secondaryBackgroundColor = "#141A31"`
- `textColor = "#F7F8FC"`

### Extra System Package For Tools

This repo now includes a `packages.txt` file with:

```text
ffmpeg
```

Streamlit Community Cloud uses that file to install the system dependency required for merged video downloads and audio conversion in the `Tools` page.

## Outlier Finder Methodology Summary

Outlier Finder is one of the most custom parts of the app, so it deserves a direct summary here.

### What it measures

The outlier score is a weighted mix of:

- channel-baseline lift
- peer percentile
- engagement percentile
- recency boost

### Key derived metrics

- `Views Per Day`
  - views divided by video age in days
- `Views Per Subscriber`
  - views normalized by channel subscriber count when public
- `Peer Percentile`
  - performance relative to the scanned cohort
- `Baseline Component`
  - how far the video is running above the channel's recent baseline
- `Language Confidence`
  - heuristic score based on metadata and title script

### Practical constraints

- results come from the scanned cohort returned by YouTube search, not the entire platform
- YouTube search is ranked and sampled
- subscriber counts may be hidden or rounded
- language filtering is heuristic, not guaranteed
- public-only workflows do not include owner-only metrics like watch time, average percentage viewed, thumbnail impressions, or thumbnail CTR; those require the Channel Insights Google OAuth flow

### Current cache behavior

- niche query cache: 1 hour
- channel baseline cache: 6 hours

## AI Integrations

### Outlier AI Research

`src/services/outlier_ai.py` converts outlier results into structured research cards with:

- executive headline
- key takeaway
- confidence label and notes
- breakout themes
- title patterns
- repeatable angles
- notable anomalies
- next steps
- warnings

Provider support:

- Gemini
- OpenAI

### Thumbnail Generation

`src/llm_integration/thumbnail_generator.py` supports:

- Gemini image generation
- OpenAI image generation via the Images API

It exposes controls for:

- model
- count
- size
- quality
- background
- output format

Generated files are saved under `outputs/thumbnails/`. The directory is intentionally kept out of Git so generated images do not accumulate in the repo.

## Tools Page Notes

The `Tools` page is intentionally scoped to public YouTube content and temporary downloads.

### Supported V1 modes

- `Single`
  - exact metadata preview
  - exact transcript-language selection
  - exact audio/video format selection where available
- `Batch`
  - newline-separated public URLs
  - per-item statuses
  - per-item downloads
- `Playlist`
  - public playlist preview
  - selected-item processing
  - per-item downloads

### Dependencies used by Tools

- `yt-dlp`
  - metadata extraction
  - format listing
  - audio/video downloads
  - playlist expansion
- `youtube-transcript-api`
  - transcript language discovery
  - transcript retrieval
  - transcript export
- `ffmpeg`
  - merged video downloads
  - MP3 conversion

### Important delivery constraint

`st.download_button` keeps file data in memory for the connected session. For that reason, the app blocks very large in-app downloads instead of trying to stream arbitrarily large files through Streamlit.

### Known Tools limitations

- public URLs only
- no auth/cookies in V1
- private, members-only, or region-restricted videos may fail
- batch and playlist downloads are sequential, not parallel
- batch and playlist modes use quality profiles instead of per-video exact format IDs
- transcript summarization is not included in V1

## Scripts

The `scripts/` directory includes the repo's operational utilities.

### `scripts/yt_api_smoketest.py`

A richer smoke test for the public YouTube Data API. It checks:

- channel discovery
- channel details
- uploads playlist traversal
- video details
- video categories
- sample comments

Use it when validating that a YouTube API key is working and returning the expected response shapes.

### `scripts/build_*_dataset.py`

These scripts build the CSV datasets for different categories:

- `build_category_dataset.py`
- `build_fitness_dataset.py`
- `build_research_dataset.py`

They are useful if you want to refresh or regenerate the bundled datasets outside the Streamlit app.

### `scripts/available_data_constraints.md`

Documents what the public YouTube API can and cannot provide, and how those limitations should influence product design and interpretation.

## Tests

The current test suite includes:

- `tests/unit/test_outliers_finder.py`
  - verifies scoring behavior, ordering, scan quality summaries, and presentational helpers
- `tests/unit/test_outlier_ai.py`
  - verifies JSON extraction, report mapping, and fallback behavior
- `tests/integration/test_pipeline.py`
  - verifies outlier search flow with mocked API responses and advanced filters
- `tests/unit/test_text_processing.py`
- `tests/unit/test_data_collection.py`
- `tests/unit/test_youtube_tools.py`
  - verifies URL validation, playlist shaping, format curation, and batch error handling
- `tests/unit/test_transcript_service.py`
  - verifies transcript option normalization and transcript file export
- `tests/unit/test_file_utils.py`
  - verifies temp-file helpers and filename sanitization
- `tests/integration/test_tools_flow.py`
  - verifies playlist and batch orchestration for the new Tools page

Run:

```bash
python3 -m pytest
```

## Known Limitations

This app is intentionally pragmatic, not a full YouTube intelligence platform with first-party creator analytics.

Important limitations:

- all live research is limited to public YouTube metadata
- all `Tools` exports are limited to public YouTube content and Streamlit-friendly in-memory delivery
- YouTube API search quota is expensive, especially `search.list`
- `yt-dlp` and transcript retrieval behavior can change when YouTube changes extraction behavior
- Outlier Finder is not an exhaustive rank tracker
- language, geography, and subscriber-based filters are best-effort
- some legacy folders in `src/` are still placeholders and do not reflect the live dashboard architecture

## Supporting Documentation

- `docs/ARCHITECTURE.md`
  - original high-level architecture note
- `docs/PROJECT_BRIEF.md`
  - original academic project brief
- `CONTRIBUTING.md`
  - contribution guidelines
- `SECURITY.md`
  - private reporting guidance for vulnerabilities
- `LICENSE`
  - MIT license

## Contribution And Maintenance Notes

If you change behavior or configuration:

- update the relevant view/service code
- update tests if the behavior is observable
- update this README if setup, deployment, or feature scope changed

For UI changes, include screenshots in pull requests as noted in `CONTRIBUTING.md`.

## License

MIT License. See `LICENSE`.
