"""
# preprocess.py
# =============
Clean and standardise the raw Steam dataset before feature engineering.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    MIN_REVIEWS, MIN_POSITIVE_PCT,
    RAW_DATA_PATH, PROCESSED_PATH, DATA_DIR
)

log = logging.getLogger(__name__)


# Cleaning


def parse_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert free-text release_date to datetime; extract year/month.
    Gracefully handles datasets (e.g. SteamSpy) that lack a release_date column."""
    df = df.copy()
    if "release_date" not in df.columns:
        df["release_year"]  = 0
        df["release_month"] = 0
        return df
    df["release_dt"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]  = df["release_dt"].dt.year.fillna(0).astype(int)
    df["release_month"] = df["release_dt"].dt.month.fillna(0).astype(int)
    return df


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that can't be meaningfully modelled."""
    before = len(df)
    df = df[df["name"].str.strip() != ""]    # Must have a name
    df = df[df["app_id"] > 0]                # Valid app ID
    df = df[df["total_reviews"] >= 0]        # Sanity check
    df = df.drop_duplicates(subset="app_id")
    log.info("Dropped %d unusable rows. %d remain.", before - len(df), len(df))
    return df.reset_index(drop=True)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or fill missing values with sensible defaults."""
    df = df.copy()
    df["price_usd"]           = df["price_usd"].fillna(0.0)
    df["genres"]              = df["genres"].fillna("")
    df["user_tags"]           = df["user_tags"].fillna("")
    df["short_description"]   = df["short_description"].fillna("")
    df["developer"]           = df["developer"].fillna("Unknown")
    df["publisher"]           = df["publisher"].fillna("Unknown")
    df["owners_estimate"]     = df["owners_estimate"].fillna(0).astype(int)
    df["avg_playtime_forever"]= df["avg_playtime_forever"].fillna(0).astype(int)
    df["avg_playtime_2w"]     = df.get("avg_playtime_2w", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["ccu"]                 = df["ccu"].fillna(0).astype(int)
    df["discount"]            = df["discount"].fillna(0).astype(int)
    return df


# Target label


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: 1 = "successful", 0 = "not successful".

    Primary: review-based (MIN_REVIEWS reviews AND >= MIN_POSITIVE_PCT positive).
    Fallback: owner-based (>= 100,000 owners) for games with few/no reviews.
    """
    df = df.copy()
    has_enough_reviews = df["total_reviews"] >= MIN_REVIEWS
    is_well_reviewed   = df["pct_positive"] >= MIN_POSITIVE_PCT
    review_success     = has_enough_reviews & is_well_reviewed

    # Fallback for games with few reviews: use owner estimate
    owner_success      = df.get("owners_estimate", pd.Series(0, index=df.index)) >= 100_000
    df["success"]      = (review_success | (~has_enough_reviews & owner_success)).astype(int)

    rate = df["success"].mean() * 100
    log.info(
        "Target created: %d successful / %d total (%.1f%% base rate).",
        df["success"].sum(), len(df), rate
    )
    return df


# Developer reputation


def add_developer_reputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode developer track record as the mean success rate across
    their previous games (leave-one-out style to avoid leakage).
    """
    df = df.copy()
    dev_success = (
        df.groupby("developer")["success"]
        .transform(lambda x: (x.sum() - x) / max(len(x) - 1, 1))
    )
    df["developer_avg_success"] = dev_success.fillna(df["success"].mean())
    return df


# Pipeline entry point


def preprocess(raw_path: str = RAW_DATA_PATH, out_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a processed DataFrame."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    log.info("Loading raw data from %s...", raw_path)
    df = pd.read_csv(raw_path)
    log.info("Raw shape: %s", df.shape)

    df = drop_bad_rows(df)
    df = fill_missing(df)
    df = parse_release_date(df)
    df = create_target(df)
    df = add_developer_reputation(df)

    df.to_csv(out_path, index=False)
    log.info("Processed data saved → %s  (shape: %s)", out_path, df.shape)
    return df
