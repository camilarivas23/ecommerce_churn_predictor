"""
Model Training Pipeline
Trains Random Forest classifier on e-commerce features.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

try:
    from app.features import EcommerceChurnFeatureEngineer
except ImportError:
    from features import EcommerceChurnFeatureEngineer

FEATURE_NAMES = [
    'total_orders',
    'total_items_purchased',
    'unique_products_bought',
    'avg_cart_size',
    'unique_categories_bought',
    'total_spent',
    'avg_order_value',
    'avg_product_rating',
    'avg_discount_pct',
    'avg_price_per_item',
]


def load_data(data_dir: str):
    """Load raw data from JSON files."""
    products_file = os.path.join(data_dir, "products.json")
    carts_file = os.path.join(data_dir, "carts.json")
    users_file = os.path.join(data_dir, "users.json")

    with open(products_file) as f:
        products_data = json.load(f)
    with open(carts_file) as f:
        carts_data = json.load(f)
    with open(users_file) as f:
        users_data = json.load(f)

    products_df = pd.DataFrame(products_data.get("products", []))
    carts_df = pd.DataFrame(carts_data.get("carts", []))
    users_df = pd.DataFrame(users_data.get("users", []))

    return products_df, carts_df, users_df


def train(data_dir: str = "data/raw", output_dir: str = "."):
    """Train churn prediction model.

    Args:
        data_dir: Directory containing raw JSON data
        output_dir: Directory to save model and features
    """
    print("Loading data...")
    products_df, carts_df, users_df = load_data(data_dir)
    print(f"  Products: {len(products_df)}, Carts: {len(carts_df)}, Users: {len(users_df)}")

    if carts_df.empty:
        print("ERROR: No cart data found. Cannot train model.")
        return

    print("\nGenerating features...")
    engineer = EcommerceChurnFeatureEngineer(carts_df, products_df, users_df)
    features_df = engineer.generate_all_features()

    if features_df.empty:
        print("ERROR: No features generated. Cannot train model.")
        return

    print(f"  Generated {len(features_df.columns)} features for {len(features_df)} users")

    # Generate churn labels
    churn_df = engineer.generate_churn_labels(features_df, threshold_days=30)
    data_df = features_df.merge(churn_df, on='user_id')

    print(f"\nChurn distribution:")
    print(data_df['churn'].value_counts().to_dict())

    # Prepare X and y
    X = data_df[FEATURE_NAMES].copy()
    y = data_df['churn'].copy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train Random Forest
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Cross-validation
    n_folds = min(5, len(X))
    if n_folds >= 2:
        cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring='f1_weighted')
        print(f"Cross-Val F1 ({n_folds}-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Feature importances
    importances = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string(index=False))

    # Save model
    model_path = os.path.join(output_dir, "model.pkl")
    features_path = os.path.join(output_dir, "features.json")

    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    features_meta = {
        "feature_names": FEATURE_NAMES,
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "churn_threshold_days": 30,
        "n_features": len(FEATURE_NAMES),
        "training_samples": len(X_train),
    }
    with open(features_path, "w") as f:
        json.dump(features_meta, f, indent=2)
    print(f"Features metadata saved to {features_path}")


if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIR", "data/raw")
    output_dir = os.environ.get("MODEL_OUTPUT", ".")
    train(data_dir, output_dir)
