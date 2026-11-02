"""
Stage 3 — Unsupervised anomaly detection (Isolation Forest).

Trained on raw + physics-engineered features with NO risk labels. Its
decision_function output becomes one more engineered feature ("Anomaly_Score")
that flows into Stage 4's supervised classifiers.

The `contamination` parameter is exposed as a tunable, not hardcoded silently,
because it is an analyst prior with no empirically-derived value in the
source data -- the dashboard's CDA page lets you see how sensitive the
downstream model is to this choice, rather than hiding that assumption.
"""
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from . import config

ANOMALY_FEATURES = config.RAW_FEATURES + config.ENGINEERED_FEATURES
MODEL_PATH = os.path.join(config.MODEL_DIR, "isolation_forest.joblib")


def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.1) -> IsolationForest:
    X = df[ANOMALY_FEATURES]
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X)
    joblib.dump(model, MODEL_PATH)
    return model


def load_isolation_forest() -> IsolationForest:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Isolation Forest not trained yet. Call /api/train first.")
    return joblib.load(MODEL_PATH)


def score(df: pd.DataFrame, model: IsolationForest = None) -> pd.Series:
    """
    Higher decision_function output = more normal.
    We flip sign so that higher Anomaly_Score = more anomalous,
    which is the more intuitive convention for the dashboard and for
    feeding into the supervised classifier as a "risk-like" feature.
    """
    model = model or load_isolation_forest()
    X = df[ANOMALY_FEATURES]
    raw_score = model.decision_function(X)
    return pd.Series(-raw_score, index=df.index, name="Anomaly_Score")


def score_single(row: dict, model: IsolationForest = None) -> float:
    model = model or load_isolation_forest()
    X = pd.DataFrame([{k: row[k] for k in ANOMALY_FEATURES}])
    raw_score = model.decision_function(X)[0]
    return float(-raw_score)
