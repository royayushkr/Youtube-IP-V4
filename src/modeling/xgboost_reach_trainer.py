import re
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

RAW_PATH     = Path("data/processed/combined_videos_raw.csv")
TOPIC_PATH   = Path("data/processed/bertopic_metadata.csv")
STATS_PATH   = Path("data/processed/topic_stats.csv")
OUT_DIR      = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_VIEWS    = 1
MIN_AGE_DAYS = 3
N_TRIALS     = 60
N_CV_FOLDS   = 5

FEATURES = [
    "log_subscribers", "log_duration", "log_age_days",
    "publish_hour", "publish_dow", "platform_era",
    "title_wordcount", "title_has_number", "title_has_question",
    "title_has_brackets", "title_has_caps_word", "title_sentiment",
    "has_description", "has_tags", "is_hd", "is_sparse_text",
    "bertopic_token_count", "category_name", "topic_id", "trend_score",
]


def parse_duration(val):
    if pd.isna(val):
        return np.nan
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(val).strip())
    if not m:
        return np.nan
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return float(h * 3600 + mi * 60 + s)


def load_and_merge() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw["video_publishedAt"] = pd.to_datetime(raw["video_publishedAt"], utc=True, errors="coerce")
    raw["snapshot_utc"]      = pd.to_datetime(raw["snapshot_utc"],      utc=True, errors="coerce")

    for col in ["views", "likes", "comments", "channel_subscriberCount"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if "duration_sec" not in raw.columns:
        raw["duration_sec"] = raw["duration"].apply(parse_duration)

    topic = pd.read_csv(
        TOPIC_PATH,
        usecols=["video_id", "topic_id", "bertopic_token_count", "is_sparse_text", "is_short", "duration_sec"],
    )

    trend = pd.read_csv(STATS_PATH, usecols=["topic_id", "trend_score"])
    topic = topic.merge(trend, on="topic_id", how="left")

    raw_cols = [c for c in raw.columns if c not in ("duration_sec", "is_short", "is_sparse_text")]
    df = raw[raw_cols].merge(topic, on="video_id", how="inner")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    age_days             = (df["snapshot_utc"] - df["video_publishedAt"]).dt.total_seconds() / 86_400
    df["video_age_days"] = age_days
    df["views_per_day"]  = df["views"] / age_days.replace(0, np.nan)

    df["publish_hour"] = df["video_publishedAt"].dt.hour
    df["publish_dow"]  = df["video_publishedAt"].dt.dayofweek
    df["publish_year"] = df["video_publishedAt"].dt.year

    df["platform_era"] = pd.cut(
        df["publish_year"],
        bins=[0, 2016, 2020, 2023, 9999],
        labels=["early", "growth", "covid", "recent"],
    ).astype("category")

    df["log_subscribers"] = np.log1p(df["channel_subscriberCount"].fillna(0))
    df["log_duration"]    = np.log1p(df["duration_sec"].fillna(0))
    df["log_age_days"]    = np.log1p(age_days.clip(lower=0))

    title = df["video_title"].fillna("")
    df["title_wordcount"]     = title.str.split().str.len()
    df["title_has_number"]    = title.str.contains(r"\d", regex=True).astype(int)
    df["title_has_question"]  = title.str.contains(r"\?").astype(int)
    df["title_has_brackets"]  = title.str.contains(r"[\[\(]", regex=True).astype(int)
    df["title_has_caps_word"] = title.apply(
        lambda t: int(any(w.isupper() and len(w) > 1 for w in t.split()))
    )
    _pos = r"\b(best|amazing|incredible|winning|success|grow|top|ultimate)\b"
    _neg = r"\b(worst|fail|broke|lost|mistake|wrong|never)\b"
    df["title_sentiment"] = (
        title.str.lower().str.count(_pos) - title.str.lower().str.count(_neg)
    ).clip(-2, 2)

    df["has_description"] = df["video_description"].notna().astype(int)
    df["has_tags"]        = df["video_tags"].notna().astype(int)
    df["is_hd"]           = (df["definition"] == "hd").astype(int)
    df["is_sparse_text"]  = df["is_sparse_text"].astype(int)
    df["trend_score"]     = df["trend_score"].fillna(0.0).astype(float)
    df["topic_id"]        = df["topic_id"].astype(str).astype("category")
    df["category_name"]   = df["category_name"].astype("category")

    return df


def build_feature_matrix(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = np.log1p(df["views_per_day"])
    return X, y


def cv_score(X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict) -> float:
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(**params, enable_categorical=True)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
        oof[va_idx] = m.predict(X.iloc[va_idx])
    return r2_score(y, oof)


def make_objective(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":        "reg:squarederror",
            "tree_method":      "hist",
            "random_state":     42,
            "n_jobs":           -1,
            "n_estimators":     trial.suggest_int("n_estimators", 400, 1200, step=100),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 9),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        return cv_score(X, y, groups, params)
    return objective


def tune(X: pd.DataFrame, y: pd.Series, groups: pd.Series, label: str) -> dict:
    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(X, y, groups), n_trials=N_TRIALS, show_progress_bar=True)
    best = study.best_params
    print(f"  [{label}] best r2={study.best_value:.4f}  params={best}")
    return best


def run_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict, label: str) -> xgb.XGBRegressor:
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    oof = np.zeros(len(y))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBRegressor(**params, enable_categorical=True)
        model.fit(X_tr, y_tr, verbose=False)
        oof[va_idx] = model.predict(X_va)

        rmse = ((y_va - oof[va_idx]) ** 2).mean() ** 0.5
        r2   = r2_score(y_va, oof[va_idx])
        print(f"  [{label}] fold {fold+1}  rmse={rmse:.4f}  r2={r2:.4f}")

    oof_rmse = ((y - oof) ** 2).mean() ** 0.5
    oof_r2   = r2_score(y, oof)
    print(f"  [{label}] oof   rmse={oof_rmse:.4f}  r2={oof_r2:.4f}")

    final = xgb.XGBRegressor(**params, enable_categorical=True)
    final.fit(X, y, verbose=False)

    importances = dict(zip(X.columns, final.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"  [{label}] top features: {top}")

    return final


def save_model(model: xgb.XGBRegressor, name: str, best_params: dict):
    path = OUT_DIR / f"{name}.json"
    model.get_booster().save_model(path)

    meta = {
        "model_name":    name,
        "feature_names": FEATURES,
        "target":        "log1p(views_per_day)",
        "best_params":   best_params,
    }
    with open(OUT_DIR / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  saved -> {path}")


def main():
    df = load_and_merge()
    df = engineer_features(df)

    mask_valid = (
        df["views"].gt(MIN_VIEWS) &
        df["video_age_days"].gt(MIN_AGE_DAYS) &
        df["views_per_day"].notna() &
        df["log_subscribers"].notna()
    )
    df = df[mask_valid].reset_index(drop=True)
    print(f"Rows after filtering: {len(df):,}")

    for label, mask in [("longform", ~df["is_short"]), ("shorts", df["is_short"])]:
        sub = df[mask].reset_index(drop=True)
        grp = sub["channel_title"]
        print(f"\n{label}: {len(sub):,} videos — tuning ({N_TRIALS} trials)...")

        X, y = build_feature_matrix(sub)

        best_params = tune(X, y, grp, label)
        best_params.update({"objective": "reg:squarederror", "tree_method": "hist",
                            "random_state": 42, "n_jobs": -1})

        print(f"\n{label}: final cv with best params...")
        model = run_cv(X, y, grp, best_params, label)
        save_model(model, f"xgboost_{label}", best_params)


if __name__ == "__main__":
    main()