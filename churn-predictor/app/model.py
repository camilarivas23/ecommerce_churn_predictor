"""
Model Loading & Inference Module
Loads trained scikit-learn model and performs predictions.
"""
import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np


class ChurnPredictor:
    """Load trained model and make predictions."""

    def __init__(self, model_path: str, features_path: str):
        """Load model and features metadata.

        Args:
            model_path: Path to saved model.pkl
            features_path: Path to saved features.json
        """
        self.model = joblib.load(model_path)

        with open(features_path) as f:
            self.features_meta = json.load(f)

        self.feature_names = self.features_meta.get("feature_names", [])

    def predict(self, features_dict: Dict[str, float]) -> Dict:
        """Predict churn for a single user.

        Args:
            features_dict: Dict mapping feature names to values

        Returns:
            Dict with prediction, probability, confidence, top_features
        """
        # Build feature vector in correct order
        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])

        prediction = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]

        churn_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = float(max(proba))

        # Feature importances
        importances = self.model.feature_importances_
        top_features = self._get_top_features(importances, features_dict, top_k=3)

        return {
            "prediction": "CHURNED" if prediction == 1 else "ACTIVE",
            "churn_probability": round(churn_prob, 4),
            "confidence": round(confidence, 4),
            "top_features": top_features,
        }

    def _get_top_features(self, importances, features_dict, top_k=3):
        """Get top K most important features with their values."""
        pairs = list(zip(self.feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [
            {"name": name, "importance": round(float(imp), 4), "value": features_dict.get(name, 0)}
            for name, imp in pairs[:top_k]
        ]

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    def get_model_info(self) -> Dict:
        return {
            "model_type": self.features_meta.get("model_type", "unknown"),
            "n_features": self.features_meta.get("n_features", 0),
            "n_estimators": self.features_meta.get("n_estimators", 0),
            "max_depth": self.features_meta.get("max_depth", 0),
            "churn_threshold_days": self.features_meta.get("churn_threshold_days", 30),
            "training_samples": self.features_meta.get("training_samples", 0),
        }


def load_model(model_path: str, features_path: str) -> ChurnPredictor:
    """Factory function to load model."""
    return ChurnPredictor(model_path, features_path)
