"""
Model definition and wrapper for Entity Resolution pair classification.
Uses XGBoost with CPU multi-threading and gradient boosting.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import xgboost as xgb
from src import config
from src.features import FEATURE_NAMES

class EntityMatchingModel:
    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        n_jobs: int = config.N_JOBS,
        random_state: int = config.RANDOM_SEED
    ):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on feature matrix X and labels y."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities (returns P(match=1))."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        # Class 1 probability
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> List[tuple]:
        """Return list of (feature_name, importance) sorted descending."""
        if not self.is_fitted:
            return []
        importances = self.model.feature_importances_
        named = list(zip(FEATURE_NAMES, importances))
        return sorted(named, key=lambda x: x[1], reverse=True)

    def save(self, filepath: Path = config.MODEL_PATH):
        """Save trained model to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(filepath))

    def load(self, filepath: Path = config.MODEL_PATH):
        """Load trained model from file."""
        self.model.load_model(str(filepath))
        self.is_fitted = True
