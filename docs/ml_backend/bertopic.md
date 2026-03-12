# BERTopic — Methodology & Implementation

---

## Corpus

**Source:** `data/processed/combined_videos_raw.csv` — 28,037 videos across 6 categories after merging category CSVs and deduplicating cross-category overlaps.

**After preprocessing drops:** 28,000 documents (37 dropped — 1 null views, 36 unparseable durations).

| Category | Videos |
|---|---|
| research_science | 6,779 |
| entertainment | 4,690 |
| tech | 4,294 |
| food | 4,126 |
| gaming | 4,093 |
| fitness | 4,055 |

---

## Text Preprocessing

### Field Construction

Each document is built from three cleaned fields:

```
bertopic_text = title_clean + " " + title_clean + " " + desc_clean + " " + tags_clean
```

Title is doubled to upweight it relative to the longer description. All three fields are cleaned before concatenation.

**Text coverage in final corpus:**

| Coverage | Count | % |
|---|---|---|
| Title + description + tags | 19,740 | 70.5% |
| Title + description only | 5,512 | 19.7% |
| Title + tags only | 166 | 0.6% |
| Title only | 2,426 | 8.7% |
| Sparse (< 5 tokens) | 182 | 0.7% |

### Cleaning Pipeline

Each field passes through sequential regex substitutions: URLs, hashtags, @mentions, emoji, and punctuation are stripped and replaced with whitespace. Runs of whitespace are collapsed.

Descriptions additionally pass through a boilerplate filter before cleaning. The filter operates line-by-line and drops any line matching a compiled regex covering:

- Subscribe/notification CTAs
- Patreon, Ko-fi, merch, membership links
- Discord server links
- Sponsor mentions (including NordVPN, ExpressVPN, Surfshark)
- Affiliate and promo code language
- Social handle lines and "links in description" variants
- Business inquiry and email addresses
- Twitch/livestream CTAs (donate, TTS, channel points, Streamlabs)
- Copyright notices and end-card phrases
- Channel-specific boilerplate (SmarterEveryDay social variants)

Tags are pipe-delimited in the raw data and are cleaned individually before joining.

### Number Stripping

Standalone digit tokens are removed from `bertopic_text` after field concatenation. This prevents subscriber milestone content ("I hit 100K!") from generating numeric bigrams that surface as spurious topic keywords.

---

## Component Models

### Embedding Model

`all-MiniLM-L6-v2` via `sentence-transformers`. Chosen for speed and strong semantic quality on short-to-medium text. Embeddings are computed in batches across the full 28,000-document corpus.

### UMAP

```python
UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
```

`n_components=5` is used for clustering — not 2, which is visualization-only. `metric="cosine"` matches the geometry of sentence embeddings. `min_dist=0.0` allows tighter cluster packing.

### HDBSCAN

```python
HDBSCAN(
    min_cluster_size=30,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
```

`min_cluster_size=30` sets the minimum videos per topic. `prediction_data=True` is required for `transform()` on new documents at inference time. `eom` (excess of mass) is the default cluster selection method and performs better than `leaf` on this corpus size.

### Vectorizer

```python
CountVectorizer(
    ngram_range=(1, 2),
    stop_words=list(ENGLISH_STOP_WORDS) + _CHANNEL_STOPWORDS,
    min_df=10,
    max_df=0.85,
)
```

Bigrams are included to improve keyword quality (e.g. "practical engineering" vs "engineering"). `min_df=10` filters rare noisy terms. `max_df=0.85` filters near-universal terms. The stopword list extends sklearn's `ENGLISH_STOP_WORDS` with channel-name tokens that would otherwise surface as topic keywords:

```python
_CHANNEL_STOPWORDS = [
    "smarter", "smartereveryday", "vsauce", "veritasium", "kurzgesagt",
    "mrbeast", "markiplier", "pewdiepie", "linus", "linustechtips",
    "technicalguruji", "mkbhd", "unboxtherapy", "jerryrigeverything",
    "jacksepticeye", "affiliate",
]
```

### Representation Model

`KeyBERTInspired()` — refines c-TF-IDF topic keywords using embedding similarity, producing more semantically coherent labels than raw c-TF-IDF alone.

---

## Training

```python
BERTopic(
    nr_topics               = 90,
    calculate_probabilities = False,
    verbose                 = True,
)
```

`nr_topics=90` merges near-duplicate topics post-fit via c-TF-IDF similarity down to a fixed target. `calculate_probabilities=False` reduces memory overhead on the full corpus.

**Training times (March 11, 2026):**

| Stage | Duration |
|---|---|
| Embedding | ~2m 44s |
| UMAP | ~14s |
| HDBSCAN | ~2s |
| c-TF-IDF + representation | ~10s |
| Topic reduction (228 → 90) | ~8s |
| **Total** | **~3m 18s** |

---

## Outlier Reduction

HDBSCAN assigns documents it cannot confidently cluster to topic `-1`. The raw outlier rate before reduction was 36.6% (10,237 docs). To recover these documents, `reduce_outliers()` is applied using embedding similarity:

```python
new_topics = topic_model.reduce_outliers(corpus, topics, strategy="embeddings", threshold=0.3)
topic_model.update_topics(corpus, topics=new_topics, vectorizer_model=vectorizer_model, representation_model=representation_model)
```

`strategy="embeddings"` reassigns each outlier to the nearest topic centroid in embedding space. `threshold=0.3` sets the minimum cosine similarity required for reassignment — docs below this threshold remain as outliers. `update_topics()` re-derives c-TF-IDF and KeyBERT labels on the updated assignments.

**Outlier rate after reduction: 3.2%** (down from 36.6%).

---

## Results

**89 topics** found across 28,000 documents.

| Metric | Value |
|---|---|
| Topics found (excl. -1) | 89 |
| Outlier docs after reduction | 3.2% |
| Topics < 50 videos | 7 |
| Median dominant-category share | 0.87 |
| Topics > 80% one category | 58 |
| Cross-category topics (≤ 50% dominant) | 6 |

**Top 15 topics by median views:**

| Topic | Label | Videos | Median Views | Top Category |
|---|---|---|---|---|
| 67 | order, food, black friday, games | 59 | 7,757,749 | entertainment |
| 88 | claire, anne, anna, kelly | 42 | 4,425,666 | entertainment |
| 16 | twitch, streaming, gaming channel, stream | 433 | 2,147,734 | gaming |
| 72 | uncle roger, dish, restaurant | 44 | 2,033,997 | food |
| 18 | engineering, real engineering, engineer, construction | 394 | 1,978,948 | research_science |
| 26 | youtuber, youtubers, chrome, dad | 277 | 1,835,197 | gaming |
| 69 | gym, olympic, olympics, ballerina | 87 | 1,812,677 | fitness |
| 76 | season episode, episode, finale | 45 | 1,698,303 | gaming |
| 32 | scientists, researchers, science future, scientist | 273 | 1,215,842 | research_science |
| 10 | periodic, elements, science chemistry, crystals | 588 | 1,097,906 | research_science |
| 68 | minecraft, modded, messed, gaming | 90 | 1,051,687 | gaming |
| 49 | nuclear fusion, nuclear, fusion, science news | 191 | 928,033 | research_science |
| 31 | minecraft, cash, nico, mod | 188 | 632,640 | entertainment |
| 82 | engine, engines, turbo, motor | 71 | 581,085 | tech |
| 6 | planets, planet, exoplanets, nasa | 960 | 566,108 | research_science |

**Category coherence** — median dominant-category share of 0.87 indicates topics cluster strongly within categories rather than blending across niches. 58 of 89 topics are >80% one category.

---

## Outputs

| File | Location | Description |
|---|---|---|
| `bertopic_model` | `outputs/models/` | Serialized model (pickle + c-TF-IDF) |
| `combined_videos_clean.csv` | `data/processed/` | 28,000 rows × 42 cols, cleaned text fields |
| `bertopic_corpus.txt` | `data/processed/` | 28,000 documents, one per line |
| `bertopic_metadata.csv` | `data/processed/` | 28,000 rows with `topic_id` and `topic_label` |
| `topic_stats.csv` | `data/processed/` | Per-topic performance stats (89 topics) |