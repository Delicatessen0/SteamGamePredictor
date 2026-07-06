# 🎮 Steam Game Success Predictor

An end-to-end machine learning pipeline that predicts whether a Steam game will be a **hit or a flop** — before the reviews roll in.

Built as a data science portfolio project targeting the gaming industry, this project demonstrates a full ML lifecycle: **live API data collection → NLP → feature engineering → gradient-boosted classification → visualization**.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **85.0%** |
| Precision | **88.4%** |
| Recall | **95.5%** |
| F1 Score | **91.8%** |
| ROC AUC | **76.6%** |

Trained on **1,000 Steam games** sourced from the SteamSpy API.

### Top Predictive Features
| Rank | Feature | Insight |
|------|---------|---------|
| 1 | `is_free` | F2P games have a very different success profile |
| 2 | `price_usd` | Pricing strategy is a strong success signal |
| 3 | `developer_avg_success` | Established developers ship winning games more often |
| 4 | `log_ccu` | Concurrent user count captures real engagement |
| 5 | `log_owners` | Ownership estimate is a proxy for reach |

### Sample Charts

![Feature Importance](output/05_feature_importance.png)
![ROC Curve](output/07_roc_curve.png)
![Genre Success Rate](output/02_genre_success_rate.png)

---

## 🔍 How It Works

### 1. Data Collection (`src/collect.py`)
Pulls live data from Steam's **public APIs** (no API key required):
- **App list** — all 100,000+ games on Steam
- **App details** — genres, tags, price, developer, description, platform support, DLC count, achievement count
- **Review summary** — total reviews, % positive, review score descriptor

Auto-saves a checkpoint every 50 games so long runs can be safely resumed.

### 2. Preprocessing (`src/preprocess.py`)
- Removes DLCs, tools, and unreleased games
- Imputes missing values (price → 0, metacritic → -1, etc.)
- Parses free-text release dates into year + month
- **Target variable**: A game is a "hit" if it has ≥ 50 reviews AND ≥ 70% positive

### 3. Feature Engineering (`src/features.py`)
| Feature Group | Description |
|---|---|
| **NLP Sentiment** | VADER compound/pos/neg/neu scores on the game description |
| **Price Tiers** | One-hot encoded price buckets (Free, $0-5, $5-10, …, >$60) |
| **Genre Encoding** | Top-15 genres as binary flags |
| **Tag Encoding** | Top-30 Steam user tags as binary flags |
| **Temporal** | Release year (normalised), month encoded cyclically with sin/cos |
| **Metacritic** | Normalised score + has_metacritic flag |
| **Developer Rep.** | Leave-one-out mean success rate for the developer (no data leakage) |
| **Platform** | Windows / Mac / Linux support flags |
| **Content** | Achievement count, DLC count, has_demo, language count, age rating |

### 4. Model (`src/model.py`)
- **XGBoost** gradient-boosted classifier
- **Stratified 5-fold cross-validation** during training to assess stability
- **Early stopping** via eval metric on a validation split
- Outputs: accuracy, precision, recall, F1, ROC AUC

### 5. Visualization (`src/visualize.py`)
8 publication-ready charts saved to `output/`:

| Chart | File |
|---|---|
| Review score distribution | `01_review_distribution.png` |
| Success rate by genre | `02_genre_success_rate.png` |
| Price vs. success | `03_price_vs_success.png` |
| Sentiment vs. community reception | `04_sentiment_vs_reviews.png` |
| Top feature importances | `05_feature_importance.png` |
| Confusion matrix | `06_confusion_matrix.png` |
| ROC curve | `07_roc_curve.png` |
| Success rate over time (2010–2026) | `08_success_rate_over_time.png` |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Internet connection (for Steam API calls)

### Installation
```bash
git clone https://github.com/Delicatessen0/SteamGamePredictor.git
cd SteamGamePredictor
pip install -r requirements.txt
```

### Run the full pipeline
```bash
# Collect 1000 games, train model, generate charts
python main.py

# Skip data collection (use existing data/raw_steam_data.csv)
python main.py --skip-collect

# Only collect data, don't train
python main.py --collect-only

# Collect a custom number of games
python main.py --n-games 500
```

> ⏱ Data collection takes ~20-30 min for 1,000 games due to API rate limiting.  
> The pipeline auto-saves progress every 50 games — safe to interrupt and resume.

---

## 📁 Project Structure

```
SteamGamePredictor/
├── src/
│   ├── collect.py       # Steam API data collection
│   ├── preprocess.py    # Cleaning, target creation
│   ├── features.py      # Feature engineering (NLP, encoding, etc.)
│   ├── model.py         # XGBoost training & evaluation
│   └── visualize.py     # Chart generation
├── data/
│   ├── raw_steam_data.csv      # Raw API output
│   ├── processed_data.csv      # After cleaning
│   └── features.csv            # Final feature matrix
├── model/
│   └── xgb_model.pkl           # Trained model
├── output/
│   └── *.png                   # All generated charts
├── config.py            # Pipeline settings (tweak here)
├── main.py              # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Edit `config.py` to customise the pipeline:

```python
N_GAMES          = 1000   # Games to collect
MIN_REVIEWS      = 50     # Min reviews for a game to be labelled
MIN_POSITIVE_PCT = 70     # Min % positive for "hit" label
TOP_N_TAGS       = 30     # Tags to one-hot encode
TOP_N_GENRES     = 15     # Genres to one-hot encode
```

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Requests** — Steam API calls
- **Pandas / NumPy** — data manipulation
- **VaderSentiment** — NLP sentiment analysis
- **Scikit-learn** — preprocessing, model evaluation, cross-validation
- **XGBoost** — gradient-boosted classification
- **Matplotlib / Seaborn** — visualization

---

## 🔮 Future Work

- [ ] Add Steam user-tag embeddings via Word2Vec / FastText
- [ ] Regression variant to predict exact review score
- [ ] Streamlit web app for interactive game success prediction
- [ ] Incorporate SteamSpy data (owner estimates, peak player counts)
- [ ] Time-series analysis of how genres cycle in/out of popularity

---

## 👤 Author

**Zachary Smith** — Data Science graduate, Penn State University  
📧 zachary90266@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/zach-smith-8228a8272) · [GitHub](https://github.com/Delicatessen0) · [Portfolio](https://delicatessen0.github.io/Portfolio)
