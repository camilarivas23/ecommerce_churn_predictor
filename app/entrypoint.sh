#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data/raw}"
MODEL_PATH="${MODEL_PATH:-/app/model.pkl}"
FEATURES_PATH="${FEATURES_PATH:-/app/features.json}"
MODEL_OUTPUT="${MODEL_OUTPUT:-/app}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"

echo "=== E-Commerce Churn Predictor ==="
echo "DATA_DIR=$DATA_DIR MODEL_PATH=$MODEL_PATH MODEL_OUTPUT=$MODEL_OUTPUT"

# Step 1: Fetch data from DummyJSON API
echo "Fetching data from DummyJSON API..."
DATA_DIR="$DATA_DIR" python -u scraper.py

# Step 2: Train model
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$FEATURES_PATH" ]; then
    echo "No model found. Training..."
    DATA_DIR="$DATA_DIR" MODEL_OUTPUT="$MODEL_OUTPUT" python -u train.py
else
    echo "Model already exists at $MODEL_PATH"
fi

# Step 3: Start API
echo "Starting API on $API_HOST:$API_PORT..."
exec uvicorn main:app --host "$API_HOST" --port "$API_PORT"