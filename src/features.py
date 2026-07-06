"""
features.py
───────────
Feature engineering pipeline:
  1. NLP sentiment on game descriptions (VADER)
  2. Price tier encoding
  3. Genre / tag one-hot encoding
  4. Platform flags
  5. Temporal features
  6. Metacritic integration
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    TOP_N_TAGS, TOP_N_GENRES, PRICE_BINS,
    MAX_DESCRIPTION_LEN, FEATURES_PATH, DATA_DIR
)

log = logging.getLogger(__name__)


# ── NLP ───────────────────────────────────────────────────────────────────────

def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run VADER sentiment analysis on the short game description.
    Adds: sentiment_pos, sentiment_neg, sentiment_neu, sentiment_compound
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        log.info("Running VADER sentiment analysis...")

        def score(text: str) -> dict:
            text = str(text)[:MAX_DESCRIPTION_LEN]
            return analyzer.polarity_scores(text)

        scores = df["short_description"].fillna("").apply(score)
        df = df.copy()
        df["sentiment_pos"]      = scores.apply(lambda s: s["pos"])
        df["sentiment_neg"]      = scores.apply(lambda s: s["neg"])
        df["sentiment_neu"]      = scores.apply(lambda s: s["neu"])
        df["sentiment_compound"] = scores.apply(lambda s: s["compound"])
        log.info("Sentiment features added.")
    except ImportError:
        log.warning("vaderSentiment not installed — skipping NLP features.")
        df = df.copy()
        for col in ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]:
            df[col] = 0.0
    return df


# ── Price ─────────────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bin continuous price into categorical tiers."""
    df = df.copy()
    labels = [f"price_tier_{i}" for i in range(len(PRICE_BINS) - 1)]
    df["price_tier"] = pd.cut(
        df["price_usd"], bins=PRICE_BINS, labels=labels, right=True
    ).astype(str)
    price_dummies = pd.get_dummies(df["price_tier"], prefix="")
    df = pd.concat([df, price_dummies], axis=1)
    return df


# ── Genres & Tags ─────────────────────────────────────────────────────────────

def _one_hot_pipe_column(df: pd.DataFrame, col: str, top_n: int, prefix: str) -> pd.DataFrame:
    """One-hot encode a pipe-separated string column, keeping only the top_n values."""
    exploded = df[col].fillna("").str.split("|").explode()
    top_vals = exploded[exploded != ""].value_counts().head(top_n).index.tolist()
    for val in top_vals:
        safe = val.lower().replace(" ", "_").replace("&", "and").replace("-", "_")
        df[f"{prefix}_{safe}"] = df[col].fillna("").str.contains(val, regex=False).astype(int)
    return df


def add_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Encoding genres...")
    return _one_hot_pipe_column(df.copy(), "genres", TOP_N_GENRES, "genre")


def add_tag_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Encoding tags...")
    return _one_hot_pipe_column(df.copy(), "user_tags", TOP_N_TAGS, "tag")


# ── Temporal ──────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["release_year_norm"] = (df["release_year"] - 2003) / (2026 - 2003)  # Normalise to [0,1]
    # Cyclical encoding for month (captures seasonality)
    df["month_sin"] = np.sin(2 * np.pi * df["release_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["release_month"] / 12)
    return df


# ── Metacritic ────────────────────────────────────────────────────────────────

def add_metacritic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add metacritic features. Gracefully handles missing column."""
    df = df.copy()
    metacritic = df.get("metacritic_score", pd.Series(-1, index=df.index)).fillna(-1)
    df["has_metacritic"]  = (metacritic > 0).astype(int)
    df["metacritic_norm"] = metacritic.clip(lower=0) / 100.0
    return df


# ── Language count ────────────────────────────────────────────────────────────

def add_language_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count supported languages. Works with 'supported_languages' or 'languages' column."""
    df = df.copy()
    lang_col = "supported_languages" if "supported_languages" in df.columns else "languages"
    if lang_col in df.columns:
        df["num_languages"] = (
            df[lang_col].fillna("").str.split(",").apply(len)
        )
    else:
        df["num_languages"] = 1
    return df


def add_steamspy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric features from SteamSpy owner/playtime data."""
    df = df.copy()
    df["log_owners"]     = np.log1p(df.get("owners_estimate", pd.Series(0, index=df.index)))
    df["log_ccu"]        = np.log1p(df.get("ccu", pd.Series(0, index=df.index)))
    df["log_playtime"]   = np.log1p(df.get("avg_playtime_forever", pd.Series(0, index=df.index)))
    df["playtime_ratio"] = (
        df.get("avg_playtime_2w", pd.Series(0, index=df.index)) /
        (df.get("avg_playtime_forever", pd.Series(1, index=df.index)) + 1)
    ).fillna(0)
    df["has_discount"]   = (df.get("discount", pd.Series(0, index=df.index)) > 0).astype(int)
    return df


# ── Final feature matrix ──────────────────────────────────────────────────────

# Columns that are IDs / raw strings / targets — never used as model features
DROP_COLS = [
    "app_id", "name", "developer", "publisher",
    "release_date", "release_dt", "genres", "user_tags",
    "short_description", "supported_languages", "languages",
    "review_score_desc", "price_tier",
    "release_year", "release_month",
    "total_reviews", "positive_reviews", "negative_reviews",
    "pct_positive", "score_rank",
    "owners_estimate", "ccu",          # raw values replaced by log-transformed versions
    "avg_playtime_forever", "avg_playtime_2w", "median_playtime",
    "metacritic_score",
]


def build_feature_matrix(df: pd.DataFrame, out_path: str = FEATURES_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run all feature engineering steps and return (X, y).

    X : DataFrame of numeric/binary features ready for XGBoost
    y : Series of binary success labels
    """
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    log.info("Building feature matrix...")

    df = add_sentiment_features(df)
    df = add_price_features(df)
    df = add_genre_features(df)
    df = add_tag_features(df)
    df = add_temporal_features(df)
    df = add_metacritic_features(df)
    df = add_language_count(df)
    df = add_steamspy_features(df)

    y = df["success"]
    drop = [c for c in DROP_COLS if c in df.columns] + ["success"]
    X = df.drop(columns=drop)

    # Keep only numeric columns (belt-and-suspenders)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)

    # Save combined for inspection
    df_out = X.copy()
    df_out["success"] = y.values
    df_out.to_csv(out_path, index=False)

    log.info("Feature matrix shape: %s  |  %d features", X.shape, X.shape[1])
    log.info("Saved to %s", out_path)
    return X, y
