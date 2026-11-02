"""
Stage 4 — Supervised classification.

Trains and compares 5 classifiers spanning different inductive biases:
linear (Logistic Regression), kernel (SVM-RBF), shallow NN (MLP),
and 2 tree ensembles (Random Forest, XGBoost).

Exposes TWO split strategies deliberately:
  - "random": standard stratified random split
  - "chronological": train on the earliest N%, test on the most recent
    (1-N)% of days -- guards against random-split leakage of
    autocorrelated day-to-day operating conditions.

Every model's evaluation now reports the FULL metric set (not just
weighted F1): accuracy, precision/recall/F1 at both macro and weighted
average, per-class precision/recall/F1/support, a confusion matrix, and
macro one-vs-rest ROC-AUC where probability estimates are available.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score,
)
import xgboost as xgb

from . import config
from . import feature_engineering as fe
from . import anomaly

FEATURE_COLS = config.MODEL_FEATURES

# --- Hyperparameter search spaces for the SAME 5 models -- optimization is
# a search over each existing model's own hyperparameters, not a new model
# family. Kept intentionally small so RandomizedSearchCV finishes in a few
# seconds on a dataset this size, which matters since this runs on-demand
# from the Train page rather than as an offline batch job. ---
PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
    },
    "svm": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
    },
    "mlp": {
        "hidden_layer_sizes": [(32,), (64,), (64, 32), (100,), (64, 64)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    },
    "logreg": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
    },
}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM (RBF)",
    "mlp": "Neural Net (MLP)",
    "logreg": "Logistic Regression",
}
MODEL_KEY_MAP = {
    "random_forest": "random_forest", "randomforest": "random_forest", "random forest": "random_forest",
    "xgboost": "xgboost",
    "svm_rbf": "svm", "svm": "svm", "svm (rbf)": "svm",
    "neural_net_mlp": "mlp", "mlp": "mlp", "neural net (mlp)": "mlp",
    "logistic_regression": "logreg", "logreg": "logreg", "logistic regression": "logreg",
}


def _prepare_features(df: pd.DataFrame, iso_model=None) -> pd.DataFrame:
    out = df.copy()
    if "Anomaly_Score" not in out.columns:
        out["Anomaly_Score"] = anomaly.score(out, model=iso_model)
    return out


def _split(df: pd.DataFrame, strategy: str, test_size: float = 0.25):
    if strategy == "chronological":
        df_sorted = df.sort_values("Timestamp")
        n = len(df_sorted)
        cut = int(n * (1 - test_size))
        return df_sorted.iloc[:cut], df_sorted.iloc[cut:]
    else:
        return train_test_split(
            df, test_size=test_size, random_state=config.RANDOM_STATE,
            stratify=df[config.TARGET],
        )


def _fit_with_optional_search(estimator, param_key, X, y, optimize, n_iter, cv):
    """
    Fits `estimator` directly, or wraps it in RandomizedSearchCV over its
    own predefined search space when optimize=True. Returns
    (fitted_best_estimator, best_params_or_None). cv is clamped to the
    smallest class count in y so stratified CV never errors out on a rare
    class -- important since HIGH-risk rows are a small minority by design.
    """
    if not optimize:
        estimator.fit(X, y)
        return estimator, None

    min_class_count = pd.Series(y).value_counts().min()
    safe_cv = max(2, min(cv, int(min_class_count)))
    if min_class_count < 2:
        # Too few examples of some class to cross-validate at all -- fall
        # back to a plain fit rather than erroring the whole training run.
        estimator.fit(X, y)
        return estimator, None

    search = RandomizedSearchCV(
        estimator, PARAM_DISTRIBUTIONS[param_key], n_iter=n_iter, cv=safe_cv,
        scoring="f1_weighted", random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_all_models(df: pd.DataFrame, split_strategy: str = "random",
                      test_size: float = 0.25, contamination: float = 0.1,
                      optimize: bool = False, search_iter: int = 8, search_cv: int = 3) -> dict:
    """
    Full Stage 3 + Stage 4 run. Trains all 5 classifiers and returns the
    complete metric set for each, plus everything needed for live
    multi-model inference afterward.

    optimize=True runs RandomizedSearchCV over each model's OWN
    hyperparameter space (see PARAM_DISTRIBUTIONS) before final evaluation
    -- this is the Hyperparameter Optimization stage of the ML lifecycle,
    applied to the same 5 model families rather than adding new ones.
    """
    iso_model = anomaly.fit_isolation_forest(df, contamination=contamination)
    df = _prepare_features(df, iso_model=iso_model)

    train_df, test_df = _split(df, split_strategy, test_size)

    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]
    y_train, y_test = train_df[config.TARGET], test_df[config.TARGET]

    le = LabelEncoder()
    le.fit(df[config.TARGET])
    all_classes = list(le.classes_)
    y_train_enc, y_test_enc = le.transform(y_train), le.transform(y_test)
    y_test_bin = label_binarize(y_test, classes=all_classes)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    trained_models = {}
    best_params_used = {}

    def evaluate(key, label, y_pred, y_proba):
        results[label] = _metrics(y_test, y_pred, all_classes, y_test_bin, y_proba)

    # --- Tree ensembles: raw features, scale-invariant ---
    rf_base = RandomForestClassifier(random_state=config.RANDOM_STATE)
    rf, rf_params = _fit_with_optional_search(rf_base, "random_forest", X_train, y_train, optimize, search_iter, search_cv)
    evaluate("random_forest", MODEL_LABELS["random_forest"], rf.predict(X_test), rf.predict_proba(X_test))
    trained_models["random_forest"] = rf
    best_params_used["random_forest"] = rf_params

    xgb_base = xgb.XGBClassifier(random_state=config.RANDOM_STATE, eval_metric="mlogloss")
    xgbc, xgb_params = _fit_with_optional_search(xgb_base, "xgboost", X_train, y_train_enc, optimize, search_iter, search_cv)
    xgb_proba = xgbc.predict_proba(X_test)
    xgb_pred = le.inverse_transform(xgbc.predict(X_test))
    evaluate("xgboost", MODEL_LABELS["xgboost"], xgb_pred, xgb_proba)
    trained_models["xgboost"] = xgbc
    best_params_used["xgboost"] = xgb_params

    # --- Scale-sensitive models: standardized features ---
    svm_base = SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE)
    svm, svm_params = _fit_with_optional_search(svm_base, "svm", X_train_s, y_train, optimize, search_iter, search_cv)
    evaluate("svm", MODEL_LABELS["svm"], svm.predict(X_test_s), svm.predict_proba(X_test_s))
    trained_models["svm"] = svm
    best_params_used["svm"] = svm_params

    mlp_base = MLPClassifier(max_iter=2000, random_state=config.RANDOM_STATE)
    mlp, mlp_params = _fit_with_optional_search(mlp_base, "mlp", X_train_s, y_train, optimize, search_iter, search_cv)
    evaluate("mlp", MODEL_LABELS["mlp"], mlp.predict(X_test_s), mlp.predict_proba(X_test_s))
    trained_models["mlp"] = mlp
    best_params_used["mlp"] = mlp_params

    logr_base = LogisticRegression(max_iter=2000)
    logr, logr_params = _fit_with_optional_search(logr_base, "logreg", X_train_s, y_train, optimize, search_iter, search_cv)
    evaluate("logreg", MODEL_LABELS["logreg"], logr.predict(X_test_s), logr.predict_proba(X_test_s))
    trained_models["logreg"] = logr
    best_params_used["logreg"] = logr_params

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
        "class_labels": all_classes,
        "optimized": optimize,
        "best_params": {MODEL_LABELS[k]: v for k, v in best_params_used.items()},
    }

    joblib.dump({
        "models": trained_models,
        "scaler": scaler,
        "label_encoder": le,
        "best_model": best_model_name,
        "feature_cols": FEATURE_COLS,
    }, os.path.join(config.MODEL_DIR, "supervised_bundle.joblib"))

    return artifact


def _metrics(y_true, y_pred, all_classes, y_true_bin=None, y_proba=None) -> dict:
    """
    Complete evaluation: accuracy, macro/weighted precision-recall-F1,
    per-class breakdown (via classification_report), confusion matrix, and
    macro one-vs-rest ROC-AUC when probability estimates are available.
    """
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    roc_auc = None
    if y_proba is not None and y_true_bin is not None and y_true_bin.shape[1] > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))
        except ValueError:
            roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_macro": roc_auc,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def load_supervised_bundle():
    path = os.path.join(config.MODEL_DIR, "supervised_bundle.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Models not trained yet. Go to Train models and run training first.")
    return joblib.load(path)


def _build_input_row(mw, current_a, power_factor, voltage_kv):
    physics = fe.engineer_for_row(mw, current_a, power_factor, voltage_kv)
    row = {"MW": mw, "Current_A": current_a, "Power_Factor": power_factor, "Voltage_kV": voltage_kv, **physics}
    a_score = anomaly.score_single(row)
    row["Anomaly_Score"] = a_score
    return row, physics, a_score


def _predict_with_model(bundle, key, row):
    X = pd.DataFrame([{c: row[c] for c in bundle["feature_cols"]}])
    model = bundle["models"][key]
    X_input = bundle["scaler"].transform(X) if key in ("svm", "mlp", "logreg") else X

    pred = model.predict(X_input)[0]
    pred_label = bundle["label_encoder"].inverse_transform([int(pred)])[0] if key == "xgboost" else pred

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_input)[0]
            classes = bundle["label_encoder"].classes_ if key == "xgboost" else model.classes_
            proba = {str(c): float(p) for c, p in zip(classes, probs)}
        except Exception:
            proba = None
    return str(pred_label), proba


def predict_single(mw, current_a, power_factor, voltage_kv, model_key: str = None) -> dict:
    """Single-model live inference (used for a quick one-off prediction)."""
    bundle = load_supervised_bundle()
    model_key = model_key or bundle["best_model"]
    key = MODEL_KEY_MAP.get(str(model_key).lower(), "random_forest")

    row, physics, a_score = _build_input_row(mw, current_a, power_factor, voltage_kv)
    pred_label, proba = _predict_with_model(bundle, key, row)

    return {
        "prediction": pred_label,
        "probabilities": proba,
        "anomaly_score": round(a_score, 4),
        "engineered_features": physics,
        "model_used": key,
    }


def predict_all_models(mw, current_a, power_factor, voltage_kv) -> dict:
    """
    Predictive Analytics page: run the SAME input through all 5 trained
    models simultaneously so their predictions can be compared side by
    side, including whether they agree.
    """
    bundle = load_supervised_bundle()
    row, physics, a_score = _build_input_row(mw, current_a, power_factor, voltage_kv)

    per_model = {}
    for key, label in MODEL_LABELS.items():
        pred_label, proba = _predict_with_model(bundle, key, row)
        per_model[key] = {"label": label, "prediction": pred_label, "probabilities": proba}

    predictions = [v["prediction"] for v in per_model.values()]
    unanimous = len(set(predictions)) == 1

    return {
        "per_model": per_model,
        "unanimous": unanimous,
        "engineered_features": physics,
        "anomaly_score": round(a_score, 4),
        "best_model": bundle["best_model"],
    }
