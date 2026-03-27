# System Architecture

This repository now follows a runtime-first layout.

## Active Runtime Surface

The deployed Streamlit app is built from:

- `streamlit_app.py`
- `dashboard/`
- `src/services/`
- `src/utils/`
- `src/llm_integration/thumbnail_generator.py`
- `data/youtube api data/`
- `data/assistant/`
- `outputs/assistant/`
- `outputs/channel_insights/`

## Runtime Flow
```mermaid
flowchart LR
    A["YouTube Data API v3"] --> B["Active Service Layer"]
    C["Google OAuth + YouTube Analytics (Optional)"] --> B
    D["Curated Assistant Knowledge"] --> E["Sidebar Assistant"]
    B --> F["Streamlit Views"]
    E --> F
    F --> G["Creator Insights, Tools, Outlier Research, Channel Diagnostics"]
```

## Runtime Modules
```mermaid
flowchart TB
    subgraph UI
        A1["dashboard/app.py"]
        A2["dashboard/components/sidebar.py"]
        A3["dashboard/views/*"]
    end
    subgraph Services
        B1["src/services/public_channel_service.py"]
        B2["src/services/channel_insights_service.py"]
        B3["src/services/outliers_finder.py"]
        B4["src/services/youtube_tools.py"]
        B5["src/services/assistant_service.py"]
    end
    subgraph Utilities
        C1["src/utils/api_keys.py"]
        C2["src/utils/channel_parser.py"]
        C3["src/utils/file_utils.py"]
        C4["src/utils/text_normalization.py"]
    end

    Services --> UI
    Utilities --> Services
```

## Archived Research Material

Historical research assets that are not part of the deployed app now live under:

- `research_archive/src/`
- `research_archive/data/`
- `research_archive/docs/`
- `research_archive/notebooks/`

These materials are preserved for reference, but they are not part of the runtime deployment contract.

## Branch Guardrail

The `asher` branch is treated as a research-artifact branch, not a deploy branch.

- Do not merge its Git LFS rules or BERTopic model artifacts into `youtube-ip-v4`.
- The deployed app may use externally hosted optional BERTopic bundles, but not the branch's Git LFS artifacts directly.
- Any topic-model integration must stay optional, artifact-light by default, and outside app boot.
