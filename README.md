# AI-Based Relay Monitor System

A Flask system that demonstrates **both** the ML/data-science lifecycle and an
operational relay-monitoring workflow on top of it, for a 3-phase alternator's
protection telemetry.

```
ML lifecycle:      Upload/ingest → Validation → Physics feature engineering →
                    EDA → CDA → Isolation Forest (anomaly) → 5 supervised
                    classifiers → Hyperparameter optimization → Evaluation →
                    Model selection → Deployment (joblib) → Live inference

Operational layer:  Telemetry (live/simulated) → Data quality → Engineering
                    protection rules → AI anomaly detection → AI risk
                    prediction → AI Decision Engine (fuses all three) →
                    Alarm/Warning/Critical event → Operator Dashboard →
                    persisted event log → drift monitoring
```

## Project structure

```
AI_Based_Relay_Monitor_System/
├── app.py                     # Flask app: 9 pages + JSON API for both lifecycles
├── requirements.txt / Dockerfile / docker-compose.yml
├── data/
│   ├── alternator_data.xlsx   # bundled sample (1,000 observations) — never overwritten
│   ├── active_dataset.xlsx    # dataset actually in use (sample copy, or your upload)
│   └── events.db              # SQLite event/alarm log (created at runtime)
├── models/                    # trained model artifacts (.joblib)
├── src/
│   ├── config.py              # fixed defaults: protection thresholds, feature lists
│   ├── state.py               # runtime state: active dataset, rated MW/kV, classes
│   ├── data_loader.py         # ingestion + upload validation/schema checking
│   ├── feature_engineering.py # physics engine (MVA, load %, voltage deviation)
│   ├── anomaly.py             # Isolation Forest (unsupervised anomaly score)
│   ├── models.py              # RF/XGBoost/SVM/MLP/LogReg + hyperparameter search
│   │                           # + full metrics + multi-model predict
│   ├── decision_engine.py     # NEW — fuses rules + anomaly + ML into one verdict
│   ├── event_log.py           # NEW — persisted alarm/event history (SQLite)
│   ├── eda.py                 # descriptive statistics
│   └── cda.py                 # inferential statistics
├── templates/                 # 9 pages (see below)
└── static/                    # CSS + JS (Chart.js self-hosted, no CDN dependency)
```

## Run locally in VS Code

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                  # or press F5 — debug config in .vscode/launch.json
```
Open **http://localhost:5000**.

## Run with Docker

```bash
docker compose up --build
```
`data/` and `models/` are mounted as volumes so the active dataset, trained
models, and event log all persist across container restarts.

## Pages

| Page | Lifecycle stage(s) it covers | What it does |
|---|---|---|
| **Overview** | Data Acquisition | Dataset health check, active-dataset banner, class balance. |
| **Operator Dashboard** ⚡ | Telemetry → Decision Engine → Alarm → Historical analysis | The operational face of the system. **Live decision** tab: manual input or a simulated telemetry feed (steps through real dataset rows) run through the fused AI Decision Engine, with a color-coded severity banner and full explainability breakdown (rule/anomaly/ML each shown separately). **Event history** tab: every decision is persisted to SQLite — filter by severity, acknowledge alarms. **Monitoring** tab: compares the live severity mix against the training baseline as a plain-language drift signal. |
| **Upload data** | Data Acquisition, Validation | Upload your own `.xlsx`/`.csv`. Validates schema, detects risk-label classes dynamically, clears stale models automatically. |
| **EDA** | EDA | Interactive histograms, per-class distributions, scatter, time series, correlation matrix. |
| **Physics & Protection Rules** | Preprocessing, Feature Engineering | Live calculator for engineered features, plus the engineering setpoint threshold table the rule-based verdict uses. |
| **CDA** | CDA | ANOVA + eta-squared, Shapiro-Wilk normality, IQR outliers, Pearson CI, VIF, rule-based zero-ML baseline, Isolation Forest contamination sensitivity. |
| **Train models** | Supervised Learning, Hyperparameter Optimization, Evaluation, Selection | Trains all 5 classifiers; optional **RandomizedSearchCV hyperparameter search** over each model's own parameter space (no new model families); full metrics table (accuracy, precision/recall/F1 macro+weighted, ROC-AUC), per-class breakdown, confusion matrices. |
| **Predictive analytics** | Deployment, Live Inference | Multi-model live prediction with agreement indicator; deep model comparison (F1/ROC-AUC/per-class charts). |

## The AI Decision Engine (`src/decision_engine.py`)

This is the piece that turns three independent signals into one operator-facing
verdict, rather than just handing over a raw model prediction:

1. **Engineering protection rules** — the same deterministic setpoint logic a
   physical relay uses (`config.SETPOINTS`), zero ML involved.
2. **AI anomaly detection** — the Isolation Forest score, mapped onto the same
   3-tier severity scale.
3. **AI risk prediction** — the trained ensemble's class + confidence.

The overall severity is always the **most severe** of the three signals (never
averaged down), and the response includes an `agreement` flag plus the full
per-signal breakdown, so a CRITICAL verdict can be audited: was it a hard
threshold breach, a statistical outlier, or the ML ensemble alone? Every
evaluation — manual or simulated — is persisted to `data/events.db` via
`src/event_log.py`, which is what makes the Event History and Monitoring tabs
real rather than live-only demos.

## Authentication, roles and reports

Every page and API call requires sign-in (`/login`, `/register`). Passwords are salted+hashed (scrypt), sessions are signed HttpOnly cookies (8 h), all state-changing requests need a CSRF token, and 5 wrong passwords lock an account for 5 minutes. Users are stored in `data/users.db`; the signing key is `SECRET_KEY` or an auto-generated `data/.secret_key` (shared by all gunicorn workers).

| Role | Can do |
|---|---|
| viewer | Read-only: all dashboards, predictions, reports |
| operator | + run decision evaluations / simulated telemetry, acknowledge alarms |
| admin | + upload data, train models, clear the event log, manage users (`/users`) |

The **first account registered becomes admin**; everyone after starts as viewer and is promoted by an admin. Set `COOKIE_SECURE=1` when serving over HTTPS.

**Reports** (`/report`, also a button on the Operator Dashboard): dataset & statistics, protection setpoints, model evaluation (ranking, confusion matrix, per-class metrics, feature importance) and the alarm log. Choose sections and event count, then **Print / Save as PDF** (print stylesheet gives clean A4 black-on-white output). The event log also downloads as CSV (`/report/events.csv`). The latest training run is persisted to `models/last_train.json` so reports work across workers and restarts.

## Deploying online

The app runs unchanged in any Docker host. Set these environment variables:

| Variable | Purpose |
|---|---|
| `PERSIST_DIR` | Folder on a **persistent volume** (e.g. `/var/data`). Users, events, uploaded data, secret key and trained models are stored under it, so they survive redeploys. |
| `SECRET_KEY` | Long random string used to sign sessions (required in production). |
| `COOKIE_SECURE=1` | Send the session cookie over HTTPS only. |
| `PORT` | Injected by most hosts; the container honours it (default 5000). |
| `BOOTSTRAP_ADMIN_USER` / `_EMAIL` / `_PASSWORD` | Optional. Creates the first admin automatically when no users exist (handy on free hosts whose disk resets). |
| `WEB_CONCURRENCY` | Number of gunicorn workers (default 2; use 1 on small/free servers). |

`render.yaml` (Render) and `fly.toml` (Fly.io) are provided. After the first deploy, **register your admin account immediately** -- the first registered user becomes administrator.

## Uploading your own dataset

Required columns: `MW`, `Current_A`, `Power_Factor`, `Voltage_kV`, `Trip_Risk`.
`Timestamp`/`Date` are optional (synthesized if missing). Any category names
work for `Trip_Risk` — the decision engine normalizes arbitrary labels (e.g.
`SAFE`/`DANGER`) onto the NORMAL/WARNING/CRITICAL severity scale by keyword
match, so the Operator Dashboard and monitoring drift comparison still work
sensibly even on a dataset that doesn't use this project's original label
names. The CDA page's rule-based baseline check is the one feature that only
applies to the canonical NORMAL/WARNING/HIGH TRIP RISK taxonomy and is skipped
gracefully otherwise.

## Notes on methodology

- **Hyperparameter optimization** searches each of the 5 existing models' own
  parameter space via `RandomizedSearchCV` — this is a search over
  configurations of models already in the project, not a new model family.
  Cross-validation folds are automatically clamped to the smallest class
  count so a rare class (e.g. a handful of CRITICAL readings) never crashes
  the search.
- **The rule-based baseline** (CDA page) matters more than it looks: because
  `Trip_Risk` in the bundled sample was generated from the same deterministic
  thresholds the rule engine uses, a classifier's high scores should be read
  relative to how well the raw thresholds already reconstruct the label.
- **VIF** is reported because `Calculated_MVA` and `Load_Percent` are exact
  algebraic functions of `MW`/`Current_A`/`Voltage_kV` — expected high VIF,
  and a caveat on reading Random Forest Gini importance as causal.
- **Drift monitoring** normalizes both the training-data baseline and live
  event severities onto the same 3-tier vocabulary before comparing —
  comparing un-normalized label spellings would show fake "drift" that's
  really just a naming mismatch.
