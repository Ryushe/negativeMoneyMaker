"""
Random Forest model.

Changes from v1:
  - FEATURE_COLS now imported from features.py (15 features, not 5)
  - Removed the redundant sigmoid: CalibratedClassifierCV with isotonic
    regression already outputs well-calibrated probabilities. Applying
    logit → sigmoid on top of that is a mathematical no-op.
  - Added feature_importance() helper for diagnostics.
"""

import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS  # single source of truth

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.joblib")


def build_model() -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=200,       # more trees = more stable votes
        max_features="sqrt",    # √n_features per tree (Phase 1)
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=5)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    calibrated),
    ])


def train(X, y) -> Pipeline:
    model = build_model()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"[model] Trained on {len(X)} samples → {MODEL_PATH}")
    return model


def load() -> Pipeline | None:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_proba(model: Pipeline, feature_array: np.ndarray) -> float:
    """
    Return calibrated YES probability in [0, 1].
    CalibratedClassifierCV output is already in probability space —
    no further transformation needed.
    """
    return float(model.predict_proba(feature_array.reshape(1, -1))[0][1])


def feature_importance(model: Pipeline) -> dict[str, float]:
    """
    Return per-feature importance scores (useful for debugging which
    features the model is actually using).
    """
    try:
        # Unwrap: Pipeline → CalibratedClassifierCV → base RF
        clf = model.named_steps["clf"]
        rf  = clf.estimators_[0].base_estimator  # first CV fold's RF
        imp = rf.feature_importances_
        return dict(zip(FEATURE_COLS, imp.tolist()))
    except Exception:
        return {}
