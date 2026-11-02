"""
AI-Based Relay Monitor System

Flask application tying together the full pipeline, spanning both the
ML lifecycle AND the operational relay-monitoring workflow:

  ML lifecycle:      Upload/ingest -> Validation -> Physics feature
                      engineering -> EDA -> CDA -> Isolation Forest ->
                      Supervised classifiers (+ optional hyperparameter
                      search) -> Model evaluation/selection -> Deployment
  Operational layer:  Telemetry (live or simulated) -> Engineering
                      protection rules -> AI anomaly detection -> AI risk
                      prediction -> AI Decision Engine (fuses all three)
                      -> Alarm/Warning/Critical event -> Operator Dashboard
                      -> persisted event log -> monitoring

Run locally:
    pip install -r requirements.txt
    python app.py
    -> http://localhost:5000

Run in Docker:
    docker build -t relay-monitor-system .
    docker run -p 5000:5000 relay-monitor-system
"""
import os
import time
import json
from functools import wraps
from flask import (Flask, render_template, request, jsonify, session, redirect,
                   url_for, g, abort, make_response)

from src import config
from src import state
from src import data_loader
from src.data_loader import DatasetValidationError
from src import feature_engineering as fe
from src import anomaly
from src import models as ml_models
from src import eda as eda_mod
from src import cda as cda_mod
from src import decision_engine
from src import event_log
from src import auth

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

app.config["SECRET_KEY"] = auth.get_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE", "0") == "1"  # set 1 behind HTTPS
app.config["PERMANENT_SESSION_LIFETIME"] = 8 * 3600  # one shift

state.ensure_active_dataset()
event_log.init_db()
auth.init_db()
auth.bootstrap_admin()

LAST_TRAIN_PATH = os.path.join(config.MODEL_DIR, "last_train.json")

_CACHE = {"raw_df": None, "full_df": None, "train_artifact": None, "sim_index": 0}


# ------------------------------------------------------------ AUTH GUARD ----
# Minimum role per endpoint. Anything not listed needs only a signed-in user
# (viewer). Public endpoints are listed explicitly.
PUBLIC_ENDPOINTS = {"login", "register", "static"}
ENDPOINT_MIN_ROLE = {
    "upload_page": "admin", "train_page": "admin", "api_upload": "admin",
    "api_upload_reset": "admin", "api_train": "admin", "api_events_clear": "admin",
    "users_page": "admin", "api_users_role": "admin", "api_users_active": "admin",
    "api_decision_evaluate": "operator", "api_telemetry_simulate_next": "operator",
    "api_events_acknowledge": "operator",
}


def _wants_json():
    return request.path.startswith("/api/")


def _csrf_token():
    if "csrf" not in session:
        import secrets
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


@app.before_request
def guard():
    ep = request.endpoint
    if ep is None or ep in PUBLIC_ENDPOINTS:
        if request.method == "POST" and ep in ("login", "register"):
            if request.form.get("csrf_token") != session.get("csrf"):
                abort(400, "Invalid or missing CSRF token. Reload the page and try again.")
        return None
    g.user = None
    uid = session.get("uid")
    if uid:
        g.user = auth.get_user(uid)
        if g.user is None:  # deactivated or deleted while signed in
            session.clear()
    if g.user is None:
        if _wants_json():
            return jsonify({"error": "Authentication required.", "login": url_for("login")}), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
        if not sent or sent != session.get("csrf"):
            if _wants_json():
                return jsonify({"error": "Invalid or missing CSRF token. Reload the page."}), 400
            abort(400, "Invalid or missing CSRF token.")
    need = ENDPOINT_MIN_ROLE.get(ep)
    if need and not auth.has_role(g.user, need):
        if _wants_json():
            return jsonify({"error": "Your role (%s) cannot do this; %s required." % (g.user["role"], need)}), 403
        return render_template("forbidden.html", need=need, active=""), 403


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if request.endpoint not in ("static",):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


def _safe_next(target):
    return target if target and target.startswith("/") and not target.startswith("//") else url_for("index")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("uid") and auth.get_user(session["uid"]):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        try:
            user = auth.authenticate(request.form.get("identifier"), request.form.get("password"))
            session.clear()
            session["uid"] = user["id"]
            session.permanent = True
            return redirect(_safe_next(request.args.get("next")))
        except auth.AuthError as e:
            error = str(e)
    return render_template("login.html", error=error, csrf=_csrf_token(),
                           first_run=auth.count_users() == 0, next=request.args.get("next", ""))


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        pw = request.form.get("password", "")
        try:
            if pw != request.form.get("confirm", ""):
                raise auth.AuthError("Passwords do not match.")
            uid, role = auth.create_user(request.form.get("username"), request.form.get("email"),
                                         pw, request.form.get("full_name"))
            session.clear()
            session["uid"] = uid
            session.permanent = True
            return redirect(url_for("index"))
        except auth.AuthError as e:
            error = str(e)
    return render_template("register.html", error=error, csrf=_csrf_token(),
                           first_run=auth.count_users() == 0, form=request.form)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/users")
def users_page():
    return render_template("users.html", users=auth.list_users(), roles=auth.ROLES, active="users")


@app.route("/api/users/<int:uid>/role", methods=["POST"])
def api_users_role(uid):
    try:
        auth.set_role(uid, (request.get_json(silent=True) or {}).get("role"))
    except auth.AuthError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"success": True})


@app.route("/api/users/<int:uid>/active", methods=["POST"])
def api_users_active(uid):
    if uid == g.user["id"]:
        return jsonify({"error": "You cannot deactivate your own account."}), 400
    try:
        auth.set_active(uid, bool((request.get_json(silent=True) or {}).get("active")))
    except auth.AuthError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"success": True})


@app.context_processor
def inject_globals():
    return {
        "csrf_token": _csrf_token(),
        "current_user": getattr(g, "user", None),
        "can": lambda role: auth.has_role(getattr(g, "user", None), role),
        "g_rated_mw": state.get_rated_mw(),
        "g_rated_kv": state.get_rated_kv(),
        "g_source_label": state.get_source_label(),
    }


def load_last_artifact():
    """Latest training run: in-memory if this worker trained it, else from disk."""
    if _CACHE["train_artifact"] is not None:
        return _CACHE["train_artifact"]
    try:
        with open(LAST_TRAIN_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def invalidate_cache():
    _CACHE["raw_df"] = None
    _CACHE["full_df"] = None
    _CACHE["train_artifact"] = None
    try:
        os.remove(LAST_TRAIN_PATH)
    except OSError:
        pass


def get_raw_df():
    if _CACHE["raw_df"] is None:
        _CACHE["raw_df"] = data_loader.load_raw_data()
    return _CACHE["raw_df"]


def get_full_df():
    """Raw + physics features recomputed by our own engine (Stage 2)."""
    if _CACHE["full_df"] is None:
        raw = get_raw_df()
        engineered = fe.compute_physics_features(raw)
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
    return render_template("index.html", health=health, active="overview",
                            rated_mw=state.get_rated_mw(), rated_kv=state.get_rated_kv())


@app.route("/upload")
def upload_page():
    return render_template("upload.html", active="upload",
                            rated_mw=state.get_rated_mw(), rated_kv=state.get_rated_kv(),
                            source_label=state.get_source_label())


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
                            rated_mw=state.get_rated_mw(), rated_kv=state.get_rated_kv())


@app.route("/train")
def train_page():
    return render_template("train.html", active="train")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html", active="analytics")


@app.route("/operator")
def operator_page():
    return render_template("operator.html", active="operator")


# ------------------------------------------------------------- JSON API ----

@app.route("/api/health")
def api_health():
    df = get_raw_df()
    return jsonify(data_loader.dataset_health_report(df))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file selected."}), 400

    file_storage = request.files["file"]
    rated_mw = request.form.get("rated_mw", type=float)
    rated_kv = request.form.get("rated_kv", type=float)

    try:
        df, warnings, ordered_classes = data_loader.validate_and_prepare(file_storage)
    except DatasetValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error reading file: {e}"}), 400

    data_loader.save_active_dataset(df)
    state.set_active_dataset(
        rated_mw=rated_mw, rated_kv=rated_kv, classes=ordered_classes,
        source_label=f"Uploaded: {file_storage.filename} ({len(df)} rows)",
    )
    invalidate_cache()
    # Clear any previously trained models -- they were fit on the old data/schema.
    for fname in ("isolation_forest.joblib", "supervised_bundle.joblib"):
        fpath = os.path.join(config.MODEL_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    health = data_loader.dataset_health_report(get_raw_df())
    return jsonify({
        "success": True,
        "warnings": warnings,
        "classes_detected": ordered_classes,
        "health": health,
    })


@app.route("/api/upload/reset", methods=["POST"])
def api_upload_reset():
    state.reset_to_sample()
    invalidate_cache()
    for fname in ("isolation_forest.joblib", "supervised_bundle.joblib"):
        fpath = os.path.join(config.MODEL_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    health = data_loader.dataset_health_report(get_raw_df())
    return jsonify({"success": True, "health": health})


@app.route("/api/eda/summary")
def api_eda_summary():
    return jsonify(eda_mod.summary_statistics(get_full_df()))


@app.route("/api/eda/correlation")
def api_eda_correlation():
    return jsonify(eda_mod.correlation_matrix(get_full_df()))


@app.route("/api/eda/histogram")
def api_eda_histogram():
    feature = request.args.get("feature", "MW")
    bins = int(request.args.get("bins", 30))
    return jsonify(eda_mod.histogram_data(get_full_df(), feature, bins))


@app.route("/api/eda/boxplot")
def api_eda_boxplot():
    feature = request.args.get("feature", "MW")
    return jsonify(eda_mod.boxplot_by_class(get_full_df(), feature))


@app.route("/api/eda/class_balance")
def api_eda_class_balance():
    return jsonify(eda_mod.class_balance(get_full_df()))


@app.route("/api/eda/scatter")
def api_eda_scatter():
    x = request.args.get("x", "MW")
    y = request.args.get("y", "Current_A")
    return jsonify(eda_mod.scatter_data(get_full_df(), x, y))


@app.route("/api/eda/timeseries")
def api_eda_timeseries():
    feature = request.args.get("feature", "MW")
    return jsonify(eda_mod.timeseries_data(get_full_df(), feature))


@app.route("/api/eda/meta")
def api_eda_meta():
    """Class list + color map so the frontend doesn't hardcode label names."""
    return jsonify({"classes": state.get_classes(), "colors": state.class_color_map()})


@app.route("/api/cda/anova")
def api_cda_anova():
    return jsonify(cda_mod.anova_by_risk_class(get_full_df()))


@app.route("/api/cda/normality")
def api_cda_normality():
    return jsonify(cda_mod.normality_tests(get_full_df()))


@app.route("/api/cda/outliers")
def api_cda_outliers():
    return jsonify(cda_mod.outlier_summary(get_full_df()))


@app.route("/api/cda/correlation_test")
def api_cda_correlation_test():
    x = request.args.get("x", "MW")
    y = request.args.get("y", "Current_A")
    return jsonify(cda_mod.correlation_significance(get_full_df(), x, y))


@app.route("/api/cda/vif")
def api_cda_vif():
    return jsonify(cda_mod.variance_inflation_factors(get_full_df()))


@app.route("/api/cda/rule_baseline")
def api_cda_rule_baseline():
    return jsonify(cda_mod.rule_based_baseline(get_full_df()))


@app.route("/api/cda/contamination_sensitivity")
def api_cda_contamination_sensitivity():
    return jsonify(cda_mod.contamination_sensitivity(get_full_df()))


@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(force=True, silent=True) or {}
    split_strategy = payload.get("split_strategy", "random")
    test_size = float(payload.get("test_size", 0.25))
    contamination = float(payload.get("contamination", 0.1))
    optimize = bool(payload.get("optimize", False))
    search_iter = int(payload.get("search_iter", 8))
    search_cv = int(payload.get("search_cv", 3))

    artifact = ml_models.train_all_models(
        get_full_df(), split_strategy=split_strategy, test_size=test_size, contamination=contamination,
        optimize=optimize, search_iter=search_iter, search_cv=search_cv,
    )
    _CACHE["train_artifact"] = artifact
    try:  # persist so every worker (and the report page) sees the latest run
        artifact["trained_at"] = time.time()
        artifact["trained_by"] = g.user["username"]
        with open(LAST_TRAIN_PATH, "w") as f:
            json.dump(artifact, f)
    except Exception:
        pass
    return jsonify(artifact)


@app.route("/api/train/last")
def api_train_last():
    art = load_last_artifact()
    if art is None:
        return jsonify({"trained": False})
    return jsonify({"trained": True, **art})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = ml_models.predict_single(
            mw=float(payload["mw"]), current_a=float(payload["current_a"]),
            power_factor=float(payload["power_factor"]), voltage_kv=float(payload["voltage_kv"]),
            model_key=payload.get("model_key"),
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400


@app.route("/api/predict/all", methods=["POST"])
def api_predict_all():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = ml_models.predict_all_models(
            mw=float(payload["mw"]), current_a=float(payload["current_a"]),
            power_factor=float(payload["power_factor"]), voltage_kv=float(payload["voltage_kv"]),
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


# ------------------------------------------------- DECISION ENGINE / OPERATOR ----

@app.route("/api/decision/evaluate", methods=["POST"])
def api_decision_evaluate():
    """
    The AI Decision Engine endpoint: fuses engineering protection rules +
    anomaly detection + the trained ML ensemble into one verdict, and logs
    it to the persisted event history. This is what the Operator Dashboard
    calls for both manual input and the simulated telemetry stepper.
    """
    payload = request.get_json(force=True, silent=True) or {}
    source = payload.get("source", "manual")
    try:
        verdict = decision_engine.evaluate(
            mw=float(payload["mw"]), current_a=float(payload["current_a"]),
            power_factor=float(payload["power_factor"]), voltage_kv=float(payload["voltage_kv"]),
        )
    except Exception as e:
        return jsonify({"error": f"Decision engine failed: {e}"}), 400

    event_id = event_log.log_event(verdict, source=source)
    verdict["event_id"] = event_id
    return jsonify(verdict)


@app.route("/api/telemetry/simulate_next", methods=["POST"])
def api_telemetry_simulate_next():
    """
    Simulated live telemetry: steps through the active dataset's real rows
    one at a time (looping back to the start once exhausted), so the
    Operator Dashboard has something resembling a live feed without needing
    an actual streaming source. Each step runs the full decision engine and
    logs it, exactly like a manual entry would.
    """
    try:
        df = get_raw_df()
        idx = _CACHE["sim_index"] % len(df)
        row = df.iloc[idx]
        _CACHE["sim_index"] = idx + 1

        verdict = decision_engine.evaluate(
            mw=float(row["MW"]), current_a=float(row["Current_A"]),
            power_factor=float(row["Power_Factor"]), voltage_kv=float(row["Voltage_kV"]),
        )
        verdict["source_row_index"] = int(idx)
        verdict["actual_label"] = str(row[config.TARGET])
        event_id = event_log.log_event(verdict, source="simulated")
        verdict["event_id"] = event_id
        return jsonify(verdict)
    except Exception as e:
        return jsonify({"error": f"Simulation step failed: {e}"}), 400


@app.route("/api/events")
def api_events():
    limit = int(request.args.get("limit", 50))
    severity = request.args.get("severity")
    return jsonify(event_log.list_events(limit=limit, severity=severity))


@app.route("/api/events/<int:event_id>/acknowledge", methods=["POST"])
def api_events_acknowledge(event_id):
    ok = event_log.acknowledge_event(event_id)
    return jsonify({"success": ok})


@app.route("/api/events/clear", methods=["POST"])
def api_events_clear():
    event_log.clear_events()
    return jsonify({"success": True})


@app.route("/api/monitoring/summary")
def api_monitoring_summary():
    """
    Lightweight model-monitoring view: compares the severity mix of recent
    logged events against the training dataset's own class distribution.
    A large divergence is a plain-language proxy for drift -- the live
    population of readings no longer resembles what the models were
    evaluated on -- without requiring ground-truth labels on live data
    (which, in a real deployment, arrive late or never).

    Both sides are normalized onto the SAME 3-tier severity vocabulary
    (NORMAL/WARNING/CRITICAL) before comparing -- the raw dataset label
    (e.g. "HIGH TRIP RISK") and the decision engine's fused output label
    (e.g. "CRITICAL") describe the same thing but aren't spelled the same,
    and comparing them un-normalized would show fake "drift" that's really
    just a vocabulary mismatch.
    """
    df = get_raw_df()
    raw_counts = df[config.TARGET].value_counts().to_dict()
    baseline_counts = {}
    for label, count in raw_counts.items():
        norm = decision_engine.normalize_severity(label)
        baseline_counts[norm] = baseline_counts.get(norm, 0) + count
    baseline_total = sum(baseline_counts.values())
    baseline_pct = {k: v / baseline_total for k, v in baseline_counts.items()}

    recent_counts = event_log.severity_counts(since_seconds=None)
    recent_total = sum(recent_counts.values()) or 1
    recent_pct = {k: v / recent_total for k, v in recent_counts.items()}

    all_labels = set(baseline_pct) | set(recent_pct)
    drift_terms = {}
    for label in all_labels:
        b = baseline_pct.get(label, 0.0001)
        r = recent_pct.get(label, 0.0001)
        drift_terms[label] = round((r - b) * 100, 2)  # percentage-point shift

    bundle_available = os.path.exists(os.path.join(config.MODEL_DIR, "supervised_bundle.joblib"))

    return jsonify({
        "baseline_distribution_pct": {k: round(v * 100, 2) for k, v in baseline_pct.items()},
        "recent_distribution_pct": {k: round(v * 100, 2) for k, v in recent_pct.items()},
        "recent_event_count": sum(recent_counts.values()),
        "drift_pp_shift": drift_terms,
        "models_deployed": bundle_available,
    })


# --------------------------------------------------------------- REPORTS ----

def _build_report(event_limit=50):
    df = get_raw_df()
    events = event_log.list_events(limit=event_limit)
    counts = {}
    for k, v in event_log.severity_counts(since_seconds=None).items():
        counts[decision_engine.normalize_severity(k)] = counts.get(decision_engine.normalize_severity(k), 0) + v
    unack = sum(1 for e in events if not e["acknowledged"] and e["severity"] != "NORMAL")
    art = load_last_artifact()
    ranked = []
    if art:
        ranked = sorted(art["results"].items(), key=lambda kv: -kv[1]["f1_weighted"])
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "health": data_loader.dataset_health_report(df),
        "stats": eda_mod.summary_statistics(get_full_df()),
        "setpoints": config.SETPOINTS,
        "rated_mw": state.get_rated_mw(), "rated_kv": state.get_rated_kv(),
        "artifact": art, "ranked": ranked,
        "trained_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(art["trained_at"])) if art and art.get("trained_at") else None,
        "events": events, "event_counts": counts, "event_total": sum(counts.values()),
        "unacknowledged": unack,
        "models_deployed": os.path.exists(os.path.join(config.MODEL_DIR, "supervised_bundle.joblib")),
    }


@app.route("/report")
def report_page():
    try:
        limit = max(10, min(500, int(request.args.get("n", 50))))
    except ValueError:
        limit = 50
    return render_template("report.html", r=_build_report(limit), active="report", event_limit=limit,
                           sections={k: (request.args.get(k, "1") == "1") if not request.args else (k in request.args)
                                     for k in ("data", "rules", "models", "events")})


@app.route("/report/events.csv")
def report_events_csv():
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "time", "source", "severity", "agreement", "ml_prediction", "confidence",
                "anomaly_score", "MW", "Current_A", "Power_Factor", "Voltage_kV",
                "recommended_action", "acknowledged"])
    for e in event_log.list_events(limit=5000):
        i = e["inputs"]
        w.writerow([e["id"], e["time_str"], e["source"], e["severity"], e["agreement"],
                    e["predicted_label"], e["confidence"], e["anomaly_score"],
                    i.get("mw"), i.get("current_a"), i.get("power_factor"), i.get("voltage_kv"),
                    e["recommended_action"], e["acknowledged"]])
    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=relay_events_%s.csv" % time.strftime("%Y%m%d_%H%M")
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
