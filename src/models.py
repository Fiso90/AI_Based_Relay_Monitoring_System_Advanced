"""
Stage 4 — Supervised classification.

Trains and compares 5 classifiers spanning different inductive biases:
linear (Logistic Regression), kernel (SVM-RBF), shallow NN (MLP),
and 2 tree ensembles (Random Forest, XGBoost).

Exposes TWO split strategies deliberately:
  - "random": standard stratified random split (what the original slide
    deck implicitly used)
  - "chronological": train on the earliest N%, test on the most recent
    (1-N)% of days

The chronological split exists because a random split can leak
autocorrelated day-to-day operating conditions from test into train.
Comparing the two split strategies side-by-side on the dashboard is the
single most important "confirmatory" check in this whole project --
a big drop in F1 under chronological split is evidence the random-split
number is optimistic.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import xgboost as xgb

from . import config
from . import feature_engineering as fe
from . import anomaly

FEATURE_COLS = config.MODEL_FEATURES


def _prepare_features(df: pd.DataFrame, iso_model=None) -> pd.DataFrame:
    """Ensure physics features + anomaly score are present and consistent."""
    out = df.copy()
    if "Anomaly_Score" not in out.columns:
        out["Anomaly_Score"] = anomaly.score(out, model=iso_model)
    return out


def _split(df: pd.DataFrame, strategy: str, test_size: float = 0.25):
    if strategy == "chronological":
        df_sorted = df.sort_values("Timestamp")
        n = len(df_sorted)
        cut = int(n * (1 - test_size))
        train_df = df_sorted.iloc[:cut]
        test_df = df_sorted.iloc[cut:]
        return train_df, test_df
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=config.RANDOM_STATE,
            stratify=df[config.TARGET],
        )
        return train_df, test_df


def train_all_models(df: pd.DataFrame, split_strategy: str = "random",
                      test_size: float = 0.25, contamination: float = 0.1) -> dict:
    """
    Full Stage 3 + Stage 4 run: fits the Isolation Forest, derives the
    anomaly score, splits the data per `split_strategy`, trains all 5
    classifiers, and returns metrics + artifacts needed by the dashboard.
    """
    iso_model = anomaly.fit_isolation_forest(df, contamination=contamination)
    df = _prepare_features(df, iso_model=iso_model)

    train_df, test_df = _split(df, split_strategy, test_size)

    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]
    y_train, y_test = train_df[config.TARGET], test_df[config.TARGET]

    le = LabelEncoder()
    le.fit(df[config.TARGET])
    y_train_enc, y_test_enc = le.transform(y_train), le.transform(y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    trained_models = {}

    # --- Tree ensembles: raw features, scale-invariant ---
    rf = RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = _metrics(y_test, rf_pred)
    trained_models["random_forest"] = rf

    xgbc = xgb.XGBClassifier(n_estimators=300, random_state=config.RANDOM_STATE, eval_metric="mlogloss")
    xgbc.fit(X_train, y_train_enc)
    xgb_pred = le.inverse_transform(xgbc.predict(X_test))
    results["XGBoost"] = _metrics(y_test, xgb_pred)
    trained_models["xgboost"] = xgbc

    # --- Scale-sensitive models: standardized features ---
    svm = SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE)
    svm.fit(X_train_s, y_train)
    results["SVM (RBF)"] = _metrics(y_test, svm.predict(X_test_s))
    trained_models["svm"] = svm

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=config.RANDOM_STATE)
    mlp.fit(X_train_s, y_train)
    results["Neural Net (MLP)"] = _metrics(y_test, mlp.predict(X_test_s))
    trained_models["mlp"] = mlp

    logr = LogisticRegression(max_iter=2000)
    logr.fit(X_train_s, y_train)
    results["Logistic Regression"] = _metrics(y_test, logr.predict(X_test_s))
    trained_models["logreg"] = logr

    # Feature importance from the Random Forest (best axis-aligned model)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    best_model_name = max(results, key=lambda k: results[k]["f1_weighted"])

    artifact = {
        "results": results,
        "feature_importance": importances.to_dict(),
        "best_model": best_model_name,
        "split_strategy": split_strategy,
        "test_size": test_size,
        "contamination": contamination,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "class_labels": list(le.classes_),
    }

    # Persist everything needed for live prediction
    joblib.dump({
        "models": trained_models,
        "scaler": scaler,
        "label_encoder": le,
        "best_model": best_model_name,
        "feature_cols": FEATURE_COLS,
    }, os.path.join(config.MODEL_DIR, "supervised_bundle.joblib"))

    return artifact


def _metrics(y_true, y_pred) -> dict:
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def load_supervised_bundle():
    path = os.path.join(config.MODEL_DIR, "supervised_bundle.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Models not trained yet. Call /api/train first.")
    return joblib.load(path)


def predict_single(mw, current_a, power_factor, voltage_kv, model_key: str = None) -> dict:
    """
    Full live inference path used by the interactive /predict page:
    raw inputs -> physics engine -> anomaly score -> chosen classifier.
    """
    bundle = load_supervised_bundle()
    model_key = model_key or bundle["best_model"].lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_key_map = {
        "random_forest": "random_forest", "randomforest": "random_forest",
        "xgboost": "xgboost",
        "svm_rbf": "svm", "svm": "svm",
        "neural_net_mlp": "mlp", "mlp": "mlp",
        "logistic_regression": "logreg", "logreg": "logreg",
    }
    key = model_key_map.get(model_key, "random_forest")

    physics = fe.engineer_for_row(mw, current_a, power_factor, voltage_kv)
    row = {"MW": mw, "Current_A": current_a, "Power_Factor": power_factor, "Voltage_kV": voltage_kv, **physics}
    a_score = anomaly.score_single(row)
    row["Anomaly_Score"] = a_score

    X = pd.DataFrame([{c: row[c] for c in bundle["feature_cols"]}])
    model = bundle["models"][key]

    if key in ("svm", "mlp", "logreg"):
        X_input = bundle["scaler"].transform(X)
    else:
        X_input = X

    pred = model.predict(X_input)[0]
    if key == "xgboost":
        pred_label = bundle["label_encoder"].inverse_transform([int(pred)])[0]
    else:
        pred_label = pred

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_input)[0]
            classes = bundle["label_encoder"].classes_ if key == "xgboost" else model.classes_
            proba = {str(c): float(p) for c, p in zip(classes, probs)}
        except Exception:
            proba = None

    return {
        "prediction": str(pred_label),
        "probabilities": proba,
        "anomaly_score": round(a_score, 4),
        "engineered_features": physics,
        "model_used": key,
    }
