# YouTube IP V4

YouTube IP V4 is the fuller Streamlit branch for YouTube benchmarking, live creator workflows, tracked channel analysis, thumbnail generation, asset tools, and retrieval-first product help.

Live app:

- [youtube-ip-v4.streamlit.app](https://youtube-ip-v4.streamlit.app/)

## Branch And Deploy Targets

- Original repo branch tag: `youtube-ip-v4`
- Original repo: `matt-foor/purdue-youtube-ip`
- Deploy repo: `royayushkr/Youtube-IP-V4`
- Deploy branch: `main`
- PR branch reference: [youtube-ip-v4](https://github.com/matt-foor/purdue-youtube-ip/tree/youtube-ip-v4)

## Sidebar Navigation

1. `Channel Analysis`
2. `Channel Insights`
3. `Recommendations`
4. `Outlier Finder`
5. `Ytuber`
6. `Tools`
7. `Deployment`

This branch also includes a global sidebar `Assistant`.

## What Each Page Solves

| Page | Problem Solved | Main Inputs And Services | Main Outputs |
| --- | --- | --- | --- |
| `Channel Analysis` | Benchmark bundled YouTube datasets by category, channel, and time range | CSV files in `data/youtube api data/`, pandas, visualization helpers | KPI cards, trend charts, top channels, top videos |
| `Channel Insights` | Track a real channel over time and understand what is working now | `public_channel_service`, `channel_snapshot_store`, `channel_insights_service`, optional Google OAuth, optional BERTopic | Snapshot history, topic trends, format patterns, outliers, next-topic ideas |
| `Recommendations` | Turn benchmark patterns into lighter strategy guidance and thumbnail prompts | Bundled datasets plus `thumbnail_generator.py` | dataset guidance, example videos, thumbnail concepts |
| `Outlier Finder` | Discover overperforming videos in a niche | `outliers_finder.py`, `outlier_ai.py`, YouTube Data API | scored outlier tables, breakout charts, AI research cards |
| `Ytuber` | Work on one live channel with creator-focused AI tooling | YouTube Data API, pooled API keys, thumbnail generator, outlier handoff | AI Studio outputs, audit views, keyword insights, planner views |
| `Tools` | Export or inspect public YouTube assets | `youtube_tools.py`, `transcript_service.py`, `yt-dlp`, `ffmpeg` | metadata previews, transcript exports, audio/video downloads, thumbnail downloads |
| `Deployment` | Show deployment and setup instructions inside the app | static app guidance in `dashboard/app.py` | deploy notes and secrets setup |

## End-To-End Data Pipeline

```mermaid
flowchart TD
    A["GitHub repo data<br/>data/youtube api data/*.csv"] --> B["streamlit_app.py"]
    U["User actions in Streamlit"] --> B
    B --> C["dashboard/app.py router"]
    C --> D["dashboard/components/sidebar.py"]
    D --> E["Channel Analysis / Channel Insights / Recommendations / Outlier Finder / Ytuber / Tools / Deployment"]

    S["Streamlit secrets / env vars"] --> F["src/utils/api_keys.py"]
    F --> G["YouTube Data API v3"]
    F --> H["Gemini / OpenAI"]
    O["Google OAuth + YouTube Analytics"] --> I["Owner analytics services"]

    A --> J["Channel Analysis + Recommendations dataset flows"]
    G --> K["Ytuber / Channel Insights / Outlier Finder / Tools live API flows"]
    H --> L["Thumbnail generation, AI reports, AI Studio, Assistant"]

    J --> M["pandas transforms + visualization helpers"]
    K --> M
    L --> M

    K --> N["Channel Insights internal pipeline"]
    N --> N1["load_public_channel_workspace(...)"]
    N1 --> N2["ensure_public_channel_frame(...)"]
    N2 --> N3["add_channel_video_features(...)"]
    N3 --> N4["_apply_requested_topic_mode(...)"]
    N4 --> N5["assign_topic_labels(...)"]
    N4 --> N6["apply_optional_topic_model(...)"]
    N6 -->|failure| N5
    N5 --> N7["primary_topic + topic_labels + topic_source"]
    N6 --> N7
    O --> N8["fetch_owner_channel_analytics(...) + _merge_owner_video_metrics(...)"]
    N7 --> N8
    N7 --> N9["_score_videos(...)"]
    N8 --> N9
    N9 --> N10["topic / duration / title / timing metrics"]
    N10 --> N11["outliers + recommendations + snapshot payload"]
    N11 --> Q["Rendered Streamlit UI"]

    M --> P["Tables, charts, cards, downloads, prompts"]
    P --> Q
```

## Live API Extraction Flow

```mermaid
flowchart LR
    A["User enters query, channel, or URL"] --> B["Page view"]
    B --> C["src/utils/api_keys.py"]
    C --> D["Provider key selection"]
    D --> E["YouTube Data API request"]
    E --> F["Service-layer normalization"]
    F --> G["pandas dataframes + scored payloads"]
    G --> H["dashboard/components/visualizations.py"]
    H --> I["Charts, tables, cards, and actions"]

    J["Optional Google OAuth session"] --> K["YouTube Analytics overlay"]
    K --> F
```

In V4, `Channel Insights` can optionally merge owner-only analytics when Google OAuth is configured and the signed-in Google account actually owns the tracked channel.

## How Channel Insights Topic Modes Are Integrated

`Channel Insights` always starts with the same public-channel workspace. The split between `Heuristic Topics` and `Model-Backed Topics (Beta)` happens only after `add_channel_video_features(...)` has created the base feature frame.

```mermaid
flowchart TD
    A["Channel Insights UI"] --> B["refresh_channel_insights(...)"]
    B --> C["load_public_channel_workspace(...)"]
    C --> D["ensure_public_channel_frame(...)"]
    D --> E["add_channel_video_features(...)"]
    E --> F["_apply_requested_topic_mode(...)"]
    F --> G["assign_topic_labels(...)"]
    F --> H["apply_optional_topic_model(...)"]
    H -->|artifact missing / invalid / load failed / transform failed| G
    G --> I["primary_topic + topic_labels + topic_source='heuristic'"]
    H --> J["model_topic_id + model_topic_label_raw + model_topic_label"]
    J --> K["primary_topic + topic_labels + topic_source='bertopic_global'"]
    I --> L["optional owner overlay in V4"]
    K --> L
    I --> M["_score_videos(...)"]
    L --> M
    M --> N["build_topic_metrics(...)"]
    M --> O["build_duration_metrics(...)"]
    M --> P["build_title_pattern_metrics(...)"]
    M --> Q["build_publish_day_metrics(...) + build_publish_hour_metrics(...)"]
    N --> R["_outlier_and_underperformer_tables(...)"]
    O --> S["_build_summary(...)"]
    P --> S
    Q --> S
    R --> T["build_grounded_idea_bundle(...) + maybe_generate_ai_overlay(...)"]
    S --> U["store_channel_snapshot(...)"]
    T --> U
    U --> V["Overview / Topic Trends / Formats / Outliers / Next Topics / History tabs"]
```

The important piece is that heuristic and beta are not two separate products. They are two topic-assignment modes inside the same refresh path. Once the dataframe has `primary_topic`, `topic_labels`, and `topic_source`, the downstream metrics, summaries, recommendations, and tabs are shared.

### What The Topic Labels Feed

- `primary_topic` becomes the grouping key for `build_topic_metrics(...)`.
- `topic_labels` and `topic_source` stay attached to each video row and are persisted in the snapshot payload.
- duration, title-pattern, publish-day, and publish-hour metrics are built from the same enriched dataframe after topic assignment.
- outlier and underperformer tables use `performance_score`, but the explanation strings shown in the UI use `primary_topic`.
- next-topic recommendations and the optional AI overlay run after the metric tables exist, so both topic modes feed the same downstream recommendation layer.

### Heuristic Vs Model-Backed Topics

- `Heuristic Topics`
  - tokenizes the title, tags, and a description excerpt
  - normalizes tokens and filters stopwords / weak tokens
  - weights tokens using a log-scaled `views_per_day` signal
  - is always available and stays the default
- `Model-Backed Topics (Beta)`
  - preprocesses text for BERTopic inference
  - requires the manifest plus the external model bundle
  - only runs when the user explicitly requests beta mode
  - falls back to heuristics on any artifact, load, or transform failure

### Heuristic Topic Derivation

```mermaid
flowchart LR
    A["video_title + video_tags + first 180 chars of video_description"] --> B["tokenize_topic_text(...)"]
    B --> C["normalize_topic_token(...)"]
    C --> D["remove stopwords + short tokens + punctuation noise"]
    D --> E["weight tokens with log1p(views_per_day + 1)"]
    E --> F["build top topic token pool"]
    F --> G["assign up to 2 topic_labels per video"]
    G --> H["set primary_topic from first label"]
```

### BERTopic Beta Preprocessing

```mermaid
flowchart LR
    A["video_title"] --> B["title duplicated"]
    C["video_description"] --> D["strip boilerplate + truncate to 600 chars"]
    E["video_tags"] --> F["normalize tags"]
    B --> G["build_bertopic_inference_text(...)"]
    D --> G
    F --> G
    G --> H["remove standalone digits"]
    H --> I["compute bertopic_token_count"]
    I --> J["flag is_sparse_text when token count is low"]
    J --> K["BERTopic transform(...)"]
    K --> L["model_topic_id + model_topic_label_raw + model_topic_label + topic_source"]
```

## Model-Backed Topics In Channel Insights

Plain-language topic modes:

- `Heuristic Topics` = built-in keyword and rule grouping from titles, descriptions, and tags
- `Model-Backed Topics` = optional BERTopic semantic grouping loaded only when beta mode is explicitly requested

```mermaid
flowchart LR
    A["Streamlit secrets"] --> B["MODEL_ARTIFACTS_ENABLED"]
    A --> C["MODEL_ARTIFACTS_MANIFEST_URL"]
    C --> D["model_artifact_service.py"]
    D --> E["Manifest JSON"]
    E --> F["artifact_url + sha256 + bundle_version"]
    F --> G["Download on explicit beta refresh only"]
    G --> H["outputs/models/runtime/<bundle_version>/"]
    H --> I["topic_model_runtime.py"]
    I --> J["channel_insights_service.py"]
    J --> K["Channel Insights UI"]
    J --> L["SQLite snapshot metadata"]
    D --> M["Fallback to heuristics on missing / invalid artifact"]
    M --> J
```

Current V4 manifest URL:

- `https://raw.githubusercontent.com/royayushkr/Youtube-IP-V4/main/data/model_manifests/bertopic_manifest_2026.03.27.json`

## Streamlit Secrets

```toml
YOUTUBE_API_KEYS = ["your_youtube_key_1", "your_youtube_key_2"]
GEMINI_API_KEYS = ["your_gemini_key_1", "your_gemini_key_2"]
OPENAI_API_KEYS = ["your_openai_key_1", "your_openai_key_2"]

GOOGLE_OAUTH_CLIENT_ID = "your-google-oauth-client-id"
GOOGLE_OAUTH_CLIENT_SECRET = "your-google-oauth-client-secret"
GOOGLE_OAUTH_REDIRECT_URI = "https://your-app-name.streamlit.app/"

MODEL_ARTIFACTS_ENABLED = true
MODEL_ARTIFACTS_MANIFEST_URL = "https://raw.githubusercontent.com/royayushkr/Youtube-IP-V4/main/data/model_manifests/bertopic_manifest_2026.03.27.json"
MODEL_ARTIFACTS_CACHE_DIR = "outputs/models/runtime"
MODEL_ARTIFACTS_DOWNLOAD_TIMEOUT_SECONDS = 300
MODEL_ARTIFACTS_MAX_SIZE_MB = 512
```

## V4 Vs V5

| Area | V4 | V5 |
| --- | --- | --- |
| Sidebar Assistant | Present | Removed |
| Google OAuth | Present | Removed |
| Channel Insights | Public + optional owner overlays | Public-only |
| Page 3 label | `Recommendations` | `Thumbnails` |
| Ytuber | Present | Present |
| Tools | Present | Present |
| Deployment page | Present | Present |
| BERTopic beta | Optional | Optional |

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

More detailed technical flow:

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment And Versions](docs/DEPLOYMENT_AND_VERSIONS.md)
