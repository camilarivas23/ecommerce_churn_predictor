"""
Model Loading & Inference Module
Loads trained scikit-learn model, scaler, PCA and performs predictions.
"""
import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np


class ChurnPredictor:
    """Load trained model, scaler, PCA and make predictions."""

    def __init__(self, model_path: str, features_path: str, scaler_path: str = None, pca_path: str = None):
        """Load model, scaler, PCA and features metadata.

        Args:
            model_path: Path to saved model.pkl
            features_path: Path to saved features.json
            scaler_path: Path to saved scaler.pkl
            pca_path: Path to saved pca.pkl
        """
        self.model = joblib.load(model_path)

        base_dir = os.path.dirname(model_path)
        self.scaler = joblib.load(scaler_path or os.path.join(base_dir, "scaler.pkl"))
        self.pca = joblib.load(pca_path or os.path.join(base_dir, "pca.pkl"))

        with open(features_path) as f:
            self.features_meta = json.load(f)

        self.feature_names = self.features_meta.get("feature_names", [])
        # PCA component names for importance mapping
        self.pca_components = self.pca.n_components_

    def predict(self, features_dict: Dict[str, float]) -> Dict:
        """Predict churn for a single user.

        Args:
            features_dict: Dict mapping feature names to values

        Returns:
            Dict with prediction, probability, confidence, top_features
        """
        # Build feature vector in correct order
        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])

        # Apply preprocessing (scaler + PCA)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        prediction = int(self.model.predict(X_pca)[0])
        proba = self.model.predict_proba(X_pca)[0]

        churn_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = float(max(proba))

        # Map PCA feature importances back to original features
        top_features = self._get_top_features_pca(features_dict, top_k=3)

        return {
            "prediction": "CHURNED" if prediction == 1 else "ACTIVE",
            "churn_probability": round(churn_prob, 4),
            "confidence": round(confidence, 4),
            "top_features": top_features,
        }

    def _get_top_features_pca(self, features_dict, top_k=3):
        """Map PCA component importances back to top contributing original features."""
        # Use PCA components to find which original features contribute most
        # Take the absolute sum of component loadings across all PCs, weighted by explained variance
        weights = self.pca.explained_variance_ratio_
        loadings = np.abs(self.pca.components_)  # (n_components, n_features)
        weighted_importances = np.dot(weights, loadings)  # (n_features,)
        weighted_importances = weighted_importances / weighted_importances.sum()

        pairs = list(zip(self.feature_names, weighted_importances))
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
            "pca_components": self.features_meta.get("pca_components", 0),
            "pca_explained_variance": self.features_meta.get("pca_explained_variance", 0),
        }


def load_model(model_path: str, features_path: str) -> ChurnPredictor:
    """Factory function to load model."""
    return ChurnPredictor(model_path, features_path)
