"""
main.py
───────
Full pipeline orchestrator for the Steam Game Success Predictor.

Usage:
  python main.py              # Run full pipeline (collect + train + visualize)
  python main.py --skip-collect  # Skip API collection (use existing data)
  python main.py --collect-only  # Only collect data, don't train
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ]
)
log = logging.getLogger(__name__)

# Add project root to path so config.py and src/ package are both importable
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_DATA_PATH, N_GAMES
from src.collect    import collect_dataset
from src.preprocess import preprocess
from src.features   import build_feature_matrix
from src.model      import split_data, train_model, evaluate_model, get_feature_importance, save_model
from src.visualize  import generate_all_charts


def parse_args():
    p = argparse.ArgumentParser(description="Steam Game Success Predictor Pipeline")
    p.add_argument("--skip-collect",  action="store_true", help="Use existing raw data")
    p.add_argument("--collect-only",  action="store_true", help="Only run data collection")
    p.add_argument("--n-games",       type=int, default=N_GAMES, help="Number of games to collect")
    return p.parse_args()


def main():
    args = parse_args()

    # -- Step 1: Data Collection -----------------------------
    if not args.skip_collect:
        log.info("=== Step 1/4: Data Collection ===")
        collect_dataset(n=args.n_games)
    else:
        if not Path(RAW_DATA_PATH).exists():
            log.error("No raw data found at %s. Run without --skip-collect first.", RAW_DATA_PATH)
            sys.exit(1)
        log.info("Skipping collection - using existing data at %s", RAW_DATA_PATH)

    if args.collect_only:
        log.info("Collection-only mode. Done.")
        return

    # -- Step 2: Preprocessing --------------------------------
    log.info("=== Step 2/4: Preprocessing ===")
    df = preprocess()

    # -- Step 3: Feature Engineering + Modeling --------------
    log.info("=== Step 3/4: Feature Engineering ===")
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    log.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)

    fi_df = get_feature_importance(model, list(X.columns))
    log.info("\nTop 10 Features:\n%s", fi_df.head(10).to_string(index=False))

    # -- Step 4: Visualizations -------------------------------
    log.info("=== Step 4/4: Generating Charts ===")
    generate_all_charts(df, model, fi_df, X_test, y_test)

    # -- Summary ----------------------------------------------
    log.info("")
    log.info("============================================")
    log.info("     PIPELINE COMPLETE -- RESULTS          ")
    log.info("============================================")
    log.info("  Accuracy:   %.4f", metrics["accuracy"])
    log.info("  Precision:  %.4f", metrics["precision"])
    log.info("  Recall:     %.4f", metrics["recall"])
    log.info("  F1 Score:   %.4f", metrics["f1"])
    log.info("  ROC AUC:    %.4f", metrics["roc_auc"])
    log.info("============================================")
    log.info("  Charts saved -> output/")
    log.info("  Model saved  -> model/xgb_model.pkl")
    log.info("============================================")


if __name__ == "__main__":
    main()
