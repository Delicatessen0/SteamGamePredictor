"""
model.py
────────
XGBoost classification model: training, evaluation, and persistence.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

from config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, MODEL_PATH

log = logging.getLogger(__name__)


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost classifier with cross-validated hyperparameter search.
    Returns the fitted model.
    """
    log.info("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    log.info(
        "CV ROC-AUC: %.4f ± %.4f  (folds: %s)",
        cv_scores.mean(), cv_scores.std(),
        np.round(cv_scores, 4)
    )

    model.fit(X_train, y_train)
    log.info("Model training complete.")
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Compute and log evaluation metrics on the held-out test set.
    Returns a dict with all metrics.
    """
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    log.info("── Test Set Metrics ─────────────────────────")
    for k, v in metrics.items():
        log.info("  %-12s %.4f", k.upper(), v)
    log.info("─────────────────────────────────────────────")
    log.info("\n%s", classification_report(y_test, y_pred, target_names=["Flop", "Hit"]))

    return metrics


def get_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    top_n: int = 20
) -> pd.DataFrame:
    """Return a DataFrame of the top_n most important features."""
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return fi_df


def save_model(model: XGBClassifier, path: str = MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved → %s", path)


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    with open(path, "rb") as f:
        model = pickle.load(f)
    log.info("Model loaded from %s", path)
    return model
