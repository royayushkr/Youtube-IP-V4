# YouTube IP V4 Architecture

## Sidebar Navigation

1. `Channel Analysis`
2. `Channel Insights`
3. `Recommendations`
4. `Outlier Finder`
5. `Ytuber`
6. `Tools`
7. `Deployment`

V4 also includes a global sidebar `Assistant`.

## Full Runtime And Data Pipeline

```mermaid
flowchart TD
    A["GitHub committed CSVs<br/>data/youtube api data/*.csv"] --> B["streamlit_app.py"]
    U["User actions"] --> B
    B --> C["dashboard/app.py"]
    C --> D["dashboard/components/sidebar.py"]
    D --> E["Page views"]

    S["Streamlit secrets / env"] --> F["src/utils/api_keys.py"]
    F --> G["YouTube Data API v3"]
    F --> H["Gemini / OpenAI"]
    O["Google OAuth + YouTube Analytics"] --> I["youtube_owner_analytics_service.py"]

    A --> J["Channel Analysis / Recommendations"]
    G --> K["Ytuber / Channel Insights / Outlier Finder / Tools"]
    H --> L["Recommendations / Ytuber / Outlier Finder / Assistant"]

    J --> M["pandas transforms + service payloads"]
    K --> M
    L --> M

    K --> N["Channel Insights service path"]
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
    N10 --> N11["summary + outliers + recommendations + snapshots"]

    M --> P["dashboard/components/visualizations.py"]
    N11 --> P
    P --> Q["Charts, cards, tables, downloads, AI outputs"]
```

## Page Problem Map

| Page | Problem Solved | Main Services / Inputs | Main UI Outputs | Interlinks |
| --- | --- | --- | --- | --- |
| `Channel Analysis` | benchmark bundled datasets | CSVs, pandas, visualization helpers | KPI cards, trend charts, ranked tables | shares benchmark context with `Recommendations` |
| `Channel Insights` | analyze one tracked channel over time | `public_channel_service`, `channel_snapshot_store`, `channel_insights_service`, optional owner analytics | topic trends, format analysis, outliers, next-topic ideas | can inform `Outlier Finder` themes |
| `Recommendations` | convert benchmark patterns into guidance and thumbnail concepts | bundled datasets, `thumbnail_generator.py` | sample videos, heuristic guidance, thumbnail outputs | overlaps with thumbnail generation used in `Ytuber` |
| `Outlier Finder` | find niche winners | `outliers_finder.py`, `outlier_ai.py`, YouTube API | scored outlier tables, breakout snapshot, AI research | receives handoff from `Ytuber` and `Channel Insights` |
| `Ytuber` | run a live creator AI workspace | YouTube API, pooled API keys, thumbnail generator | AI Studio, audit views, keyword and planner outputs | can hand off into `Outlier Finder` |
| `Tools` | export public YouTube assets | `youtube_tools.py`, `transcript_service.py`, `yt-dlp`, `ffmpeg` | metadata previews, transcript/audio/video/thumbnail downloads | standalone utility surface |
| `Deployment` | explain setup and deployment | static instructions in app shell | repo, branch, secrets, deploy notes | operational reference only |
| `Assistant` | answer help/troubleshooting/product questions | retrieval services, knowledge base, Gemini/OpenAI fallback | answer cards, related questions, feedback controls | available across all pages |

## Live API Extraction Flow

```mermaid
flowchart LR
    A["User enters channel, keyword, or URL"] --> B["Page view"]
    B --> C["src/utils/api_keys.py"]
    C --> D["Selected provider key"]
    D --> E["YouTube Data API request"]
    E --> F["Service-layer normalization"]
    F --> G["pandas dataframes / scored payloads"]
    G --> H["dashboard/components/visualizations.py"]
    H --> I["Rendered Streamlit UI"]

    J["Optional Google OAuth session"] --> K["YouTube Analytics request"]
    K --> F
```

In V4, `Channel Insights` may merge owner-only metrics only when Google OAuth is configured and the signed-in Google account owns the tracked channel.

## Channel Insights Topic Integration

The base `Channel Insights` dataframe is built the same way regardless of topic mode:

1. `load_public_channel_workspace(...)`
2. `ensure_public_channel_frame(...)`
3. `add_channel_video_features(...)`
4. `_apply_requested_topic_mode(...)`

After that, both topic modes feed the same downstream metrics, scoring, outlier detection, idea generation, and snapshot persistence.

```mermaid
flowchart TD
    A["dashboard/views/channel_insights.py"] --> B["refresh_channel_insights(...)"]
    B --> C["load_public_channel_workspace(...)"]
    C --> D["ensure_public_channel_frame(...)"]
    D --> E["add_channel_video_features(...)"]
    E --> F["_apply_requested_topic_mode(...)"]
    F --> G["assign_topic_labels(...)"]
    F --> H["apply_optional_topic_model(...)"]
    H -->|failure| G
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
    U --> V["Overview / Topic Trends / Formats / Outliers / Next Topics / History"]
```

### Topic Outputs That Persist

- `primary_topic` is the row-level theme key used in topic metrics and UI explanations.
- `topic_labels` stores the per-video label list used for grouping and later inspection.
- `topic_source` records whether the row came from heuristics or BERTopic beta.
- summary JSON and insight payloads persist:
  - `topic_mode_requested`
  - `topic_mode_used`
  - `topic_model_status`
  - `topic_model_bundle_version`
  - `topic_model_failure_reason`

## Model-Backed Topic Flow

```mermaid
flowchart LR
    A["Streamlit secrets"] --> B["MODEL_ARTIFACTS_ENABLED"]
    A --> C["MODEL_ARTIFACTS_MANIFEST_URL"]
    C --> D["src/services/model_artifact_service.py"]
    D --> E["Manifest JSON"]
    E --> F["artifact_url + sha256 + bundle_version"]
    F --> G["Download on explicit beta refresh only"]
    G --> H["outputs/models/runtime/<bundle_version>/"]
    H --> I["src/services/topic_model_runtime.py"]
    I --> J["src/services/channel_insights_service.py"]
    J --> K["dashboard/views/channel_insights.py"]
    D --> L["Fallback to heuristic topics"]
    L --> J
```

Topic modes:

- `Heuristic Topics` uses built-in keyword and rule grouping
- `Model-Backed Topics` uses optional BERTopic semantic grouping

### Heuristic Topic Derivation

```mermaid
flowchart LR
    A["video_title + video_tags + short video_description excerpt"] --> B["tokenize_topic_text(...)"]
    B --> C["normalize_topic_token(...)"]
    C --> D["drop stopwords + short tokens"]
    D --> E["weight tokens using log1p(views_per_day + 1)"]
    E --> F["build top token pool"]
    F --> G["assign topic_labels"]
    G --> H["set primary_topic from first label"]
```

### BERTopic Beta Preprocessing

```mermaid
flowchart LR
    A["video_title"] --> B["duplicate title"]
    C["video_description"] --> D["strip boilerplate + truncate"]
    E["video_tags"] --> F["normalize tags"]
    B --> G["build_bertopic_inference_text(...)"]
    D --> G
    F --> G
    G --> H["remove standalone digits"]
    H --> I["compute bertopic_token_count"]
    I --> J["flag is_sparse_text"]
    J --> K["BERTopic transform(...)"]
    K --> L["model_topic_id + raw label + human label + topic_source"]
```

## Branch Notes

- V4 keeps the global `Assistant`
- V4 keeps Google OAuth and owner-only analytics overlays in `Channel Insights`
- V4 keeps the page label `Recommendations`
- BERTopic is optional and never required at app boot
