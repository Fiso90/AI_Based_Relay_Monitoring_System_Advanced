"""
3 MW Alternator — Machine Learning Lifecycle Dashboard

Flask application tying together the full pipeline:
  Data ingestion -> EDA -> Physics engine -> CDA (confirmatory stats)
  -> Isolation Forest -> Supervised classifiers -> Live prediction

Run locally:
    pip install -r requirements.txt
    python app.py
    -> http://localhost:5000

Run in Docker:
    docker build -t alternator-dashboard .
    docker run -p 5000:5000 alternator-dashboard
"""
import os
from flask import Flask, render_template, request, jsonify

from src import config
from src import data_loader
from src import feature_engineering as fe
from src import anomaly
from src import models as ml_models
from src import eda as eda_mod
from src import cda as cda_mod

app = Flask(__name__)

# ---- In-memory cache so we don't re-read the Excel file on every request ----
_CACHE = {"raw_df": None, "full_df": None, "train_artifact": None}


def get_raw_df():
    if _CACHE["raw_df"] is None:
        _CACHE["raw_df"] = data_loader.load_raw_data()
    return _CACHE["raw_df"]


def get_full_df():
    """Raw + physics features recomputed by our own engine (Stage 2)."""
    if _CACHE["full_df"] is None:
        raw = get_raw_df()
        engineered = fe.compute_physics_features(raw)
        # Use our recomputed engineered features as the canonical ones
        full = raw.copy()
        full["Calculated_MVA"] = engineered["Calculated_MVA_engine"]
        full["Reactive_Power_MVAr"] = engineered["Reactive_Power_MVAr_engine"]
        full["Load_Percent"] = engineered["Load_Percent_engine"]
        full["Voltage_Deviation_Percent"] = engineered["Voltage_Deviation_Percent_engine"]
        _CACHE["full_df"] = full
    return _CACHE["full_df"]


# ---------------------------------------------------------------- PAGES ----

@app.route("/")
def index():
    df = get_raw_df()
    health = data_loader.dataset_health_report(df)
    return render_template("index.html", health=health, active="overview")


@app.route("/eda")
def eda_page():
    return render_template(
        "eda.html", active="eda",
        numeric_features=eda_mod.NUMERIC_COLS,
        target=config.TARGET,
    )


@app.route("/cda")
def cda_page():
    return render_template("cda.html", active="cda")


@app.route("/pipeline")
def pipeline_page():
    return render_template("pipeline.html", active="pipeline", setpoints=config.SETPOINTS,
                            rated_mw=config.RATED_MW, rated_kv=config.RATED_KV)


@app.route("/train")
def train_page():
    return render_template("train.html", active="train")


@app.route("/predict")
def predict_page():
    return render_template("predict.html", active="predict", setpoints=config.SETPOINTS)


# ------------------------------------------------------------- JSON API ----

@app.route("/api/health")
def api_health():
    df = get_raw_df()
    return jsonify(data_loader.dataset_health_report(df))


@app.route("/api/eda/summary")
def api_eda_summary():
    df = get_full_df()
    return jsonify(eda_mod.summary_statistics(df))


@app.route("/api/eda/correlation")
def api_eda_correlation():
    df = get_full_df()
    return jsonify(eda_mod.correlation_matrix(df))


@app.route("/api/eda/histogram")
def api_eda_histogram():
    feature = request.args.get("feature", "MW")
    bins = int(request.args.get("bins", 30))
    df = get_full_df()
    return jsonify(eda_mod.histogram_data(df, feature, bins))


@app.route("/api/eda/boxplot")
def api_eda_boxplot():
    feature = request.args.get("feature", "MW")
    df = get_full_df()
    return jsonify(eda_mod.boxplot_by_class(df, feature))


@app.route("/api/eda/class_balance")
def api_eda_class_balance():
    df = get_full_df()
    return jsonify(eda_mod.class_balance(df))


@app.route("/api/eda/scatter")
def api_eda_scatter():
    x = request.args.get("x", "MW")
    y = request.args.get("y", "Current_A")
    df = get_full_df()
    return jsonify(eda_mod.scatter_data(df, x, y))


@app.route("/api/eda/timeseries")
def api_eda_timeseries():
    feature = request.args.get("feature", "MW")
    df = get_full_df()
    return jsonify(eda_mod.timeseries_data(df, feature))


@app.route("/api/cda/anova")
def api_cda_anova():
    df = get_full_df()
    return jsonify(cda_mod.anova_by_risk_class(df))


@app.route("/api/cda/correlation_test")
def api_cda_correlation_test():
    x = request.args.get("x", "MW")
    y = request.args.get("y", "Current_A")
    df = get_full_df()
    return jsonify(cda_mod.correlation_significance(df, x, y))


@app.route("/api/cda/vif")
def api_cda_vif():
    df = get_full_df()
    return jsonify(cda_mod.variance_inflation_factors(df))


@app.route("/api/cda/rule_baseline")
def api_cda_rule_baseline():
    df = get_full_df()
    return jsonify(cda_mod.rule_based_baseline(df))


@app.route("/api/cda/contamination_sensitivity")
def api_cda_contamination_sensitivity():
    df = get_full_df()
    return jsonify(cda_mod.contamination_sensitivity(df))


@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(force=True, silent=True) or {}
    split_strategy = payload.get("split_strategy", "random")
    test_size = float(payload.get("test_size", 0.25))
    contamination = float(payload.get("contamination", 0.1))

    df = get_full_df()
    artifact = ml_models.train_all_models(
        df, split_strategy=split_strategy, test_size=test_size, contamination=contamination
    )
    _CACHE["train_artifact"] = artifact
    return jsonify(artifact)


@app.route("/api/train/last")
def api_train_last():
    if _CACHE["train_artifact"] is None:
        return jsonify({"trained": False})
    return jsonify({"trained": True, **_CACHE["train_artifact"]})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = ml_models.predict_single(
            mw=float(payload["mw"]),
            current_a=float(payload["current_a"]),
            power_factor=float(payload["power_factor"]),
            voltage_kv=float(payload["voltage_kv"]),
            model_key=payload.get("model_key"),
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400


@app.route("/api/predict/models_available")
def api_predict_models_available():
    try:
        bundle = ml_models.load_supervised_bundle()
        return jsonify({"available": True, "models": list(bundle["models"].keys()),
                         "best_model": bundle["best_model"]})
    except FileNotFoundError:
        return jsonify({"available": False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
