# ============================================================
#  Steam Game Success Predictor — Configuration
# ============================================================

# Data Collection
N_GAMES         = 1000      # Number of games to collect from Steam
REQUEST_DELAY   = 1.2       # Seconds between API calls (respect rate limits)
MAX_RETRIES     = 3         # Retry failed API requests this many times
RETRY_DELAY     = 5         # Seconds to wait before retrying

# Modeling
RANDOM_STATE    = 42
TEST_SIZE       = 0.2       # 80/20 train/test split
CV_FOLDS        = 5         # Cross-validation folds

# Success Definition
# A game is "successful" if it has at least this many reviews
# AND its positive review percentage is >= MIN_POSITIVE_PCT
MIN_REVIEWS     = 50
MIN_POSITIVE_PCT = 70       # Percentage (e.g. 70 = 70%)

# NLP
MAX_DESCRIPTION_LEN = 2000  # Truncate long descriptions before sentiment analysis

# Feature Engineering
TOP_N_TAGS      = 30        # How many of the most common tags to one-hot encode
TOP_N_GENRES    = 15        # How many genres to one-hot encode
PRICE_BINS      = [0, 1, 5, 10, 20, 30, 60, 999]  # USD price tiers

# Paths
DATA_DIR        = "data"
RAW_DATA_PATH   = "data/raw_steam_data.csv"
PROCESSED_PATH  = "data/processed_data.csv"
FEATURES_PATH   = "data/features.csv"
MODEL_PATH      = "model/xgb_model.pkl"
OUTPUT_DIR      = "output"
