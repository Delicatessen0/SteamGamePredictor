"""
# visualize.py
# ============
All chart generation for the Steam Game Success Predictor.
Charts are saved to the output/ directory as high-res PNGs.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

from config import OUTPUT_DIR

log = logging.getLogger(__name__)

matplotlib.use("Agg")   # Non-interactive backend — works without a display

# Global style


DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
ACCENT       = "#c084fc"   # Neon purple (matches portfolio)
ACCENT2      = "#818cf8"   # Indigo
ACCENT3      = "#34d399"   # Teal
TEXT_COLOR   = "#e6edf3"
GRID_COLOR   = "#30363d"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "text.color":        TEXT_COLOR,
    "font.family":       "sans-serif",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "figure.dpi":        150,
})

def _save(fig: plt.Figure, name: str) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Saved chart → %s", path)


# Individual charts


def plot_review_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Review Score Distribution", fontsize=16, fontweight="bold", color=TEXT_COLOR)

    # Left: Histogram of pct_positive
    axes[0].hist(df["pct_positive"], bins=30, color=ACCENT, edgecolor=DARK_BG, alpha=0.9)
    axes[0].axvline(70, color=ACCENT3, linestyle="--", linewidth=1.5, label="Success threshold (70%)")
    axes[0].set_xlabel("% Positive Reviews")
    axes[0].set_ylabel("Number of Games")
    axes[0].set_title("Positive Review % Distribution")
    axes[0].legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR)

    # Right: Total reviews (log scale)
    axes[1].hist(np.log1p(df["total_reviews"]), bins=30, color=ACCENT2, edgecolor=DARK_BG, alpha=0.9)
    axes[1].set_xlabel("log(1 + Total Reviews)")
    axes[1].set_ylabel("Number of Games")
    axes[1].set_title("Review Volume Distribution (log scale)")

    fig.tight_layout()
    _save(fig, "01_review_distribution")


def plot_genre_success_rate(df: pd.DataFrame) -> None:
    # Explode pipe-separated genres
    exploded = df[["genres", "success"]].copy()
    exploded["genres"] = exploded["genres"].fillna("Unknown").str.split("|")
    exploded = exploded.explode("genres")
    exploded = exploded[exploded["genres"].str.strip() != ""]

    genre_stats = (
        exploded.groupby("genres")["success"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "success_rate", "count": "n_games"})
        .query("n_games >= 10")
        .sort_values("success_rate", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(genre_stats.index, genre_stats["success_rate"] * 100,
                   color=ACCENT, edgecolor=DARK_BG, alpha=0.9)
    ax.bar_label(bars, fmt="%.0f%%", padding=4, color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("Success Rate by Genre  (min 10 games)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)
    fig.tight_layout()
    _save(fig, "02_genre_success_rate")


def plot_price_vs_success(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price vs. Success", fontsize=16, fontweight="bold", color=TEXT_COLOR)

    # Box plot: price distributions by success/flop
    data_hit  = df.loc[df["success"] == 1, "price_usd"].clip(0, 60)
    data_flop = df.loc[df["success"] == 0, "price_usd"].clip(0, 60)
    axes[0].boxplot(
        [data_flop, data_hit],
        tick_labels=["Flop", "Hit"],
        patch_artist=True,
        boxprops=dict(facecolor=PANEL_BG, color=ACCENT),
        whiskerprops=dict(color=ACCENT),
        capprops=dict(color=ACCENT),
        medianprops=dict(color=ACCENT3, linewidth=2),
        flierprops=dict(marker="o", markerfacecolor=ACCENT2, alpha=0.4, markersize=3),
    )
    axes[0].set_ylabel("Price (USD, capped $60)")
    axes[0].set_title("Price Distribution by Outcome")

    # Bar chart: mean success rate by price tier
    bins   = [0, 1, 5, 10, 20, 30, 60, 999]
    labels = ["Free", "$0-5", "$5-10", "$10-20", "$20-30", "$30-60", ">$60"]
    df2 = df.copy()
    df2["price_tier"] = pd.cut(df2["price_usd"], bins=bins, labels=labels, right=True)
    tier_stats = df2.groupby("price_tier", observed=True)["success"].mean() * 100
    axes[1].bar(tier_stats.index, tier_stats.values, color=ACCENT2, edgecolor=DARK_BG, alpha=0.9)
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_title("Success Rate by Price Tier")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, "03_price_vs_success")


def plot_sentiment_vs_reviews(df: pd.DataFrame) -> None:
    if "sentiment_compound" not in df.columns:
        log.warning("No sentiment columns — skipping sentiment plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    hits  = df[df["success"] == 1]
    flops = df[df["success"] == 0]

    ax.scatter(flops["sentiment_compound"], flops["pct_positive"],
               alpha=0.4, s=15, color=ACCENT2, label="Flop")
    ax.scatter(hits["sentiment_compound"],  hits["pct_positive"],
               alpha=0.6, s=15, color=ACCENT,  label="Hit")

    ax.set_xlabel("Description Sentiment (VADER compound score)")
    ax.set_ylabel("% Positive Reviews")
    ax.set_title("Sentiment vs. Community Reception", fontsize=14, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    fig.tight_layout()
    _save(fig, "04_sentiment_vs_reviews")


def plot_feature_importance(fi_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [ACCENT if i < 5 else ACCENT2 for i in range(len(fi_df))]
    bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
                   color=colors[::-1], edgecolor=DARK_BG, alpha=0.9)
    ax.set_xlabel("XGBoost Feature Importance (gain)")
    ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_feature_importance")


def plot_confusion_matrix(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["Flop", "Hit"], yticklabels=["Flop", "Hit"],
        ax=ax, linewidths=0.5, linecolor=DARK_BG,
        annot_kws={"size": 14, "color": TEXT_COLOR}
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_confusion_matrix")


def plot_roc_curve(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=GRID_COLOR, linestyle="--", linewidth=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    fig.tight_layout()
    _save(fig, "07_roc_curve")


def plot_release_year_trend(df: pd.DataFrame) -> None:
    year_stats = (
        df[df["release_year"].between(2010, 2026)]
        .groupby("release_year")["success"]
        .agg(["mean", "count"])
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(year_stats["release_year"], year_stats["mean"] * 100,
             color=ACCENT, linewidth=2.5, marker="o", markersize=6, label="Success Rate %")
    ax2.bar(year_stats["release_year"], year_stats["count"],
            color=ACCENT2, alpha=0.35, label="# Games Released")

    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Success Rate (%)", color=ACCENT)
    ax2.set_ylabel("# Games Released", color=ACCENT2)
    ax1.set_title("Steam Game Success Rate Over Time", fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    fig.tight_layout()
    _save(fig, "08_success_rate_over_time")


# Master function


def generate_all_charts(
    df: pd.DataFrame,
    model: XGBClassifier,
    fi_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    log.info("Generating all charts → %s/", OUTPUT_DIR)
    plot_review_distribution(df)
    plot_genre_success_rate(df)
    plot_price_vs_success(df)
    plot_sentiment_vs_reviews(df)
    plot_feature_importance(fi_df)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_release_year_trend(df)
    log.info("All charts saved.")
