# E-Commerce User Churn Predictor

A production-grade machine learning pipeline to predict which **e-commerce users** are at risk of stopping purchases, built with FastAPI and deployed via Docker.

## Table of Contents

1. [Overview](#overview)
2. [Churn Definition](#churn-definition)
3. [Architecture](#architecture)
4. [Project Setup](#project-setup)
5. [Data Collection](#data-collection)
6. [Feature Engineering](#feature-engineering)
7. [Feature Selection](#feature-selection)
8. [Model Training](#model-training)
9. [API Usage](#api-usage)
10. [Docker Deployment](#docker-deployment)
11. [Business Insights](#business-insights)

---

## Overview

**Goal**: Predict which users will stop purchasing, enabling targeted re-engagement campaigns.

**Key Insight**: By analyzing cart behavior and product engagement patterns, we can identify users likely to churn before they fully disengage.

**Key Metrics**:
- 13 engineered features across 3 domains (Recency, Frequency, Magnitude)
- 4 feature selection methods (Filter, RFE, Random Forest, Decision Tree)
- Random Forest classifier (100 trees, class-weighted)
- REST API with `/health`, `/predict` (UI + JSON), and `/model-info` endpoints
- Dockerized for reproducibility

**Tech Stack**:
- Python 3.11, FastAPI, scikit-learn, pandas, joblib
- DummyJSON API (no auth required)
- Docker + docker-compose

---

## Churn Definition

### Definition

**A user is considered CHURNED if:**
- No cart activity in the last **30 days**

**A user is considered ACTIVE if:**
- At least one cart activity in the last **30 days**

### Justification

1. **Why 30 days?**
   - Standard e-commerce re-engagement window
   - Short enough to detect churn early
   - Long enough to avoid noise from temporary inactivity

2. **Why cart-based churn?**
   - Cart activity is the strongest purchase intent signal
   - More actionable than passive browsing
   - Directly tied to revenue impact

---

## Architecture

```
churn-predictor/
├── app/
│   ├── main.py                  # FastAPI app with UI and endpoints
│   ├── model.py                 # Model loading and inference
│   ├── features.py              # Feature engineering (13 features)
│   ├── train.py                 # Model training pipeline
│   ├── scraper.py               # DummyJSON API data collection
│   ├── generate_sample_data.py  # Synthetic data generator
│   └── entrypoint.sh            # Docker startup script
├── notebooks/
│   └── eda_and_selection.ipynb   # EDA + 4 feature selection methods
├── data/
│   └── raw/                     # Raw JSON data
├── Dockerfile                   # Container image
├── docker-compose.yml           # Service orchestration
├── requirements.txt             # Dependencies
├── .env                         # Environment variables
├── .env.example                 # Environment template
├── .gitignore                   # Exclusions
└── README.md                    # This file
```

---

## Project Setup

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for containerization)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local development)
```

---

## Data Collection

### Option A: DummyJSON API (Recommended)

The `app/scraper.py` module fetches real data from DummyJSON:

```bash
python app/scraper.py
```

**Output**:
- `data/raw/products.json` — Product catalog (100+ items)
- `data/raw/carts.json` — Shopping carts with timestamps
- `data/raw/users.json` — User profiles
- `data/raw/categories.json` — Product categories

### Option B: Synthetic Data (Offline/Testing)

```bash
python app/generate_sample_data.py data/raw
```

Generates realistic synthetic data:
- 50 users
- 500 carts
- 100 products

---

## Feature Engineering

### 13 Features Across 3 Domains (4 types)

#### Recency Domain (3 features)
Capture how recently the user has engaged.

| Feature | Type | Business Meaning |
|---------|------|---|
| `days_since_last_cart` | Time-based | PRIMARY CHURN SIGNAL: days since last cart activity |
| `is_active_last_7d` | Binary | 1 if active in last 7 days |
| `is_active_last_30d` | Binary | 1 if active in last 30 days |

#### Frequency Domain (5 features)
Capture how consistently the user purchases.

| Feature | Type | Business Meaning |
|---------|------|---|
| `total_orders` | Aggregation | Lifetime number of orders |
| `total_items_purchased` | Aggregation | Total items bought |
| `unique_products_bought` | Aggregation | Number of unique products |
| `avg_cart_size` | Ratio | Average items per cart |
| `unique_categories_bought` | Aggregation | Category diversity |

#### Magnitude Domain (5 features)
Capture intensity and quality of engagement.

| Feature | Type | Business Meaning |
|---------|------|---|
| `total_spent` | Aggregation | Lifetime spend |
| `avg_order_value` | Ratio | Average order value |
| `avg_product_rating` | Aggregation | Quality preference |
| `avg_discount_pct` | Aggregation | Price sensitivity |
| `avg_price_per_item` | Ratio | Spending per item |

---

## Feature Selection

### 4 Complementary Methods

#### Method 1: Filter Methods
- Remove high correlation (> 0.9)
- Remove near-zero variance (< 0.01)
- Rank by target correlation

#### Method 2: Recursive Feature Elimination (RFE)
- Logistic regression base estimator
- Iteratively eliminates features
- Selects top 5 features

#### Method 3: Random Forest Importance
- 100-tree Random Forest
- Extract Gini importance
- Cross-validation F1 score

#### Method 4: Decision Tree Importance
- Single tree (max_depth=5)
- Interpretable decision rules
- Cross-validation F1 score

### Consensus Analysis

| Method | Features | CV F1 |
|--------|----------|-------|
| Full Features (13 → PCA) | 10 PCA components | ~0.86 |
| Filter | 12-13 | N/A |
| RFE | 5 | N/A |
| Random Forest | 13 | ~0.86 |
| Decision Tree | 13 | ~0.86 |

**Decision**: Use all 13 features with PCA reduction to ~10 components (95% variance retained)

---

## Model Training

### Training Process

```bash
python -m app.train
```

Or run the notebook: `notebooks/eda_and_selection.ipynb`

### Model Specification

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

### Output Files

- `app/model.pkl` — Trained model
- `app/features.json` — Feature metadata

---

## API Usage

### Endpoints

#### 1. Web UI (Interactive Form)

```
GET /
```

Opens an interactive HTML form in your browser.

#### 2. Health Check

```
GET /health
```

**Response**:
```json
{
  "status": "ok",
  "timestamp": "2026-06-09T17:27:44.767848",
  "model_loaded": true,
  "prediction_type": "e-commerce_user_churn"
}
```

#### 3. Churn Prediction

```
POST /predict
```

**Request Body**:
```json
{
  "days_since_last_cart": 45,
  "is_active_last_7d": 0,
  "is_active_last_30d": 0,
  "total_orders": 5,
  "total_items_purchased": 15,
  "unique_products_bought": 10,
  "avg_cart_size": 3.0,
  "unique_categories_bought": 4,
  "total_spent": 500.0,
  "avg_order_value": 100.0,
  "avg_product_rating": 4.0,
  "avg_discount_pct": 10.0,
  "avg_price_per_item": 33.0
}
```

**Response**:
```json
{
  "user_id": "user_input",
  "prediction": "CHURNED",
  "churn_probability": 0.89,
  "confidence": 0.89,
  "top_features": [
    {"name": "days_since_last_cart", "importance": 0.3120, "value": 45.0},
    {"name": "is_active_last_30d", "importance": 0.2540, "value": 0.0},
    {"name": "is_active_last_7d", "importance": 0.1980, "value": 0.0}
  ],
  "timestamp": "2026-06-09T17:27:51.703529",
  "interpretation": "This user has a HIGH churn risk (89.0%)."
}
```

#### 4. Model Info

```
GET /model-info
```

---

## Docker Deployment

### Building & Running

```bash
# Build image
docker-compose build

# Start service
docker-compose up

# API available at http://localhost:8000
```

### What Happens on Startup

1. Checks for existing data in `/data/raw`
2. If no data: Fetches from DummyJSON API (or generates sample data)
3. If no model: Trains Random Forest on collected data
4. Starts FastAPI server

### Docker Cleanup

```bash
docker-compose down
docker system prune
```

---

## Business Insights

### 1. Key Churn Predictors

Top features by importance (PCA-weighted loadings):
1. **days_since_last_cart** (31%) — PRIMARY: Time since last activity
2. **is_active_last_30d** (25%) — 30-day activity flag
3. **is_active_last_7d** (20%) — 7-day activity flag

### 2. User Risk Profiles

**High-Risk User** (Likely to churn):
- No cart activity for 45+ days
- Not active in last 7 or 30 days
- Low order frequency
- Low category diversity

**Low-Risk User** (Likely to stay):
- Recent cart activity (< 7 days)
- Active in both 7d and 30d windows
- High order frequency
- Diverse product categories

### 3. Actionable Interventions

| Risk Level | Trigger | Recommended Action |
|-----------|---------|---|
| **CRITICAL** | 30+ days inactive | Win-back campaign with discount |
| **HIGH** | 14-30 days inactive | Personalized product recommendations |
| **MEDIUM** | 7-14 days inactive | Cart abandonment reminder |
| **LOW** | Active (< 7 days) | Loyalty rewards, new arrivals |

### 4. Model Limitations

- **30-day threshold**: May miss short-term disengagement
- **New users**: Features unreliable for users with < 7 days history
- **Seasonal effects**: Holiday patterns may affect predictions
- **External factors**: Stock availability looks like user churn

---

## Files & Execution Order

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data (or use DummyJSON API)
python app/generate_sample_data.py data/raw

# 3. Train model
python -m app.train

# 4. Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Visit http://localhost:8000
```

### Docker Deployment

```bash
# 1. Build and start
docker-compose up --build

# 2. Visit http://localhost:8000
```

---

## Troubleshooting

### API Won't Start

```bash
# Check if model exists
ls -la app/model.pkl app/features.json

# Check Docker logs
docker-compose logs churn-api
```

### Docker Build Fails

```bash
# Clear cache
docker-compose down
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

### Model Not Loading

- Ensure `model.pkl` and `features.json` exist in `app/` directory
- Check scikit-learn version compatibility
- Run `python -m app.train` to regenerate

---

## License

This project is provided as-is for educational purposes.

---

**Last Updated**: June 2026
**Version**: 2.0.0
**Status**: Production Ready
