# V4 Deployment, Versions, And Model Flow

## Branch And Repo Targets

| Item | Value |
| --- | --- |
| Original repo branch tag | `youtube-ip-v4` |
| Original repo | `matt-foor/purdue-youtube-ip` |
| Deploy repo | `royayushkr/Youtube-IP-V4` |
| Deploy branch | `main` |
| PR branch reference | [youtube-ip-v4](https://github.com/matt-foor/purdue-youtube-ip/tree/youtube-ip-v4) |

## Navigation Order

1. `Channel Analysis`
2. `Channel Insights`
3. `Recommendations`
4. `Outlier Finder`
5. `Ytuber`
6. `Tools`
7. `Deployment`

This branch also includes the global sidebar `Assistant`.

## Streamlit Deployment Flow

```mermaid
flowchart TD
    A["GitHub repo branch<br/>youtube-ip-v4"] --> B["Streamlit app config"]
    B --> C["streamlit_app.py"]
    C --> D["dashboard/app.py"]
    D --> E["Sidebar navigation"]
    E --> F["Page views"]
    G["Streamlit secrets"] --> H["Runtime services"]
    H --> F
    F --> I["Charts, tables, cards, prompts, downloads"]
```

## Secrets And Live API Flow

```mermaid
flowchart LR
    A["Streamlit secrets"] --> B["YOUTUBE_API_KEYS"]
    A --> C["GEMINI_API_KEYS / OPENAI_API_KEYS"]
    A --> D["GOOGLE_OAUTH_*"]
    A --> E["MODEL_ARTIFACTS_*"]

    B --> F["src/utils/api_keys.py"]
    C --> F
    F --> G["YouTube Data API / Gemini / OpenAI"]
    D --> H["Google OAuth + YouTube Analytics"]
    G --> I["Service-layer transforms"]
    H --> I
    I --> J["Rendered Streamlit UI"]
```

## Model-Backed Topic Deployment

The deploy-time settings only enable the beta path. The normal `Channel Insights` pipeline still starts with the public workspace and branches inside `_apply_requested_topic_mode(...)`.

### Channel Insights Topic Pipeline

```mermaid
flowchart TD
    A["Channel Insights UI"] --> B["refresh_channel_insights(...)"]
    B --> C["load_public_channel_workspace(...)"]
    C --> D["ensure_public_channel_frame(...)"]
    D --> E["add_channel_video_features(...)"]
    E --> F["_apply_requested_topic_mode(...)"]
    F --> G["assign_topic_labels(...)"]
    F --> H["apply_optional_topic_model(...)"]
    H -->|failure| G
    G --> I["heuristic primary_topic + topic_labels + topic_source"]
    H --> J["model_topic_id + model_topic_label_raw + model_topic_label"]
    J --> K["model-backed primary_topic + topic_labels + topic_source"]
    I --> L["optional owner overlay in V4"]
    K --> L
    I --> M["_score_videos(...)"]
    L --> M
    M --> N["topic / duration / title / timing metrics"]
    N --> O["outliers + recommendations + summary payload"]
    O --> P["store_channel_snapshot(...)"]
    P --> Q["Overview / Topic Trends / Formats / Outliers / Next Topics / History"]
```

### Topic Mode Explanation

- `Heuristic Topics` = built-in token and rule grouping from title, tags, and a description excerpt
- `Model-Backed Topics` = optional BERTopic semantic grouping loaded from the external artifact path

### Heuristic Topic Derivation

```mermaid
flowchart LR
    A["title + tags + short description excerpt"] --> B["tokenize_topic_text(...)"]
    B --> C["normalize_topic_token(...)"]
    C --> D["drop stopwords + short tokens"]
    D --> E["weight by log1p(views_per_day + 1)"]
    E --> F["build top token pool"]
    F --> G["assign topic_labels and primary_topic"]
```

### BERTopic Beta Preprocessing And Artifact Flow

```mermaid
flowchart LR
    A["MODEL_ARTIFACTS_ENABLED"] --> B["beta mode can be requested"]
    C["MODEL_ARTIFACTS_MANIFEST_URL"] --> D["model_artifact_service.py"]
    D --> E["Manifest JSON"]
    E --> F["artifact_url + sha256 + bundle_version"]
    F --> G["download only on explicit beta refresh"]
    G --> H["outputs/models/runtime/<bundle_version>/"]
    H --> I["topic_model_runtime.py"]
    I --> J["build_bertopic_inference_text(...)"]
    J --> K["duplicate title"]
    J --> L["strip boilerplate description"]
    J --> M["normalize tags"]
    K --> N["remove standalone digits"]
    L --> N
    M --> N
    N --> O["bertopic_token_count + is_sparse_text"]
    O --> P["BERTopic transform(...)"]
    P --> Q["model_topic_id + raw label + human label + topic_source"]
    D --> R["fallback to heuristics if artifact is missing or invalid"]
```

### Streamlit Secrets Block

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

| Area | V4 (`youtube-ip-v4`) | V5 (`youtube-ip-v5`) |
| --- | --- | --- |
| Sidebar Assistant | Present | Removed |
| Google OAuth | Present | Removed |
| Channel Insights | Public + optional owner overlays | Public-only |
| Page 3 label | `Recommendations` | `Thumbnails` |
| Ytuber | Present | Present |
| Tools | Present | Present |
| Deployment | Present | Present |
| BERTopic beta | Optional | Optional |
