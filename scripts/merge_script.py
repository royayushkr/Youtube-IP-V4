import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_FILES = {
    "entertainment":    RAW_DIR / "entertainment_channels_videos.csv",
    "fitness":          RAW_DIR / "fitness_channels_videos.csv",
    "food":             RAW_DIR / "food_channels_videos.csv",
    "gaming":           RAW_DIR / "gaming_channels_videos.csv",
    "research_science": RAW_DIR / "research_science_channels_videos.csv",
    "tech":             RAW_DIR / "tech_channels_videos.csv",
}

# Channels scraped under both research_science and tech; assign to the
# more specific label.
RESEARCH_SCIENCE_PRIORITY_CHANNELS = {
    "Ben Eater",
    "Computerphile",
    "Practical Engineering",
    "Real Engineering",
    "Two Minute Papers",
}


def load_and_tag(category: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["category_name"] = category
    df["snapshot_utc"]  = pd.to_datetime(df["snapshot_utc"], utc=True, errors="coerce")
    log.info(f"  {category:20s}  {len(df):>6,} rows | {df['channel_title'].nunique():>3} channels")
    return df


def resolve_cross_category_dupes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For channels scraped under both research_science and tech,
    reassign category_name to 'research_science'.
    """
    mask = df["channel_title"].isin(RESEARCH_SCIENCE_PRIORITY_CHANNELS)
    n = mask.sum()
    if n:
        df.loc[mask, "category_name"] = "research_science"
        log.info(
            f"  Reassigned {n} rows for "
            f"{df.loc[mask,'channel_title'].nunique()} cross-category channels -> research_science"
        )
    return df


def keep_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    For any video_id with multiple rows (different snapshot dates),
    keep only the row with the most recent snapshot_utc.
    """
    n_before = len(df)
    df = (
        df.sort_values("snapshot_utc", ascending=False)
        .drop_duplicates(subset="video_id", keep="first")
        .reset_index(drop=True)
    )
    n_dropped = n_before - len(df)
    log.info(f"  Dropped {n_dropped:,} older-snapshot rows -> {len(df):,} unique videos remain")
    return df


def main() -> None:
    log.info("Loading category files")
    frames = [load_and_tag(cat, path) for cat, path in CATEGORY_FILES.items()]

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Combined (pre-dedup)   : {len(combined):>6,} rows | {combined['video_id'].nunique():,} unique video_ids")

    log.info("Resolving cross-category duplicates")
    combined = resolve_cross_category_dupes(combined)

    log.info("Deduplicating -- keeping latest snapshot per video_id")
    combined = keep_latest_snapshot(combined)

    log.info(f"Final shape: {combined.shape}")
    log.info(f"Unique channels: {combined['channel_title'].nunique()}")
    log.info("Category distribution:")
    for cat, count in combined["category_name"].value_counts().items():
        log.info(f"    {cat:20s}  {count:>5,} videos")

    out_path = OUT_DIR / "combined_videos_raw.csv"
    combined.to_csv(out_path, index=False)
    log.info("=" * 55)
    log.info(f"OK Saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()