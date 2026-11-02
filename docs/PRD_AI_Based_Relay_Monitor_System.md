---
title: "Product Requirements Document"
subtitle: "AI-Based Relay Monitor System"
author: "Final-Year Project Documentation"
date: "2026"
---

# Product Requirements Document
## AI-Based Relay Monitor System

**Document version:** 1.0
**Status:** Final — Deployed
**Repository:** `AI_Based_Relay_Monitor_System`

---

## 1. Executive Summary

The **AI-Based Relay Monitor System** is a decision-support application that
combines classical protection-relay engineering rules with machine learning
to monitor a 3-phase alternator's electrical telemetry and produce a single,
explainable risk verdict — **NORMAL**, **WARNING**, or **CRITICAL** — for a
human operator.

Unlike a conventional ML demo that stops at "the model predicts X", this
system is built around an explicit **AI Decision Engine** that fuses three
independent signals — deterministic engineering thresholds, unsupervised
anomaly detection, and a supervised classifier ensemble — into one verdict,
with a full audit trail explaining *why* each signal fired. Every verdict is
persisted to an event log, giving the system a genuine operational history
rather than only live, disposable predictions.

The system demonstrates the complete machine learning lifecycle (data
acquisition through monitoring and feedback) and a complete operational
relay-monitoring workflow (telemetry through historical event analysis) in
one deployable Flask application, containerized with Docker and runnable
locally in VS Code.

---

## 2. Problem Statement

Protection relays on rotating electrical machinery (alternators, generators)
traditionally trip on **fixed deterministic thresholds** — e.g., "trip if
current exceeds 120 A." This approach is reliable but blunt:

- It cannot detect a reading that is *statistically abnormal* without
  crossing a hard threshold (e.g., a combination of values never seen before,
  even if each individual value is within its own limit).
- It offers no graduated risk signal — a relay is either tripped or it isn't;
  there is no "this is trending toward a problem" state.
- It cannot learn from historical operating data to refine what "normal"
  actually looks like for a specific unit.

Pure ML approaches have the opposite problem: a black-box classifier can
flag risk more subtly, but operators are reasonably reluctant to trust an
opaque model's verdict for safety-critical equipment without knowing *why*
it fired, and a model trained once can silently drift out of relevance as
operating conditions change.

**This project's core hypothesis:** neither approach alone is sufficient for
an operator-facing decision-support tool. The two should be fused, with the
deterministic rule always available as a transparent floor, ML used to add
statistical nuance on top, and every decision logged so the fused system's
behavior can be audited and monitored over time.

---

## 3. Goals and Objectives

| # | Goal | Measurable outcome |
|---|---|---|
| G1 | Preserve and demonstrate a complete ML lifecycle | All 15 lifecycle stages (see companion ML Lifecycle document) implemented and traceable to source files |
| G2 | Fuse rule-based and AI signals into one auditable verdict | `src/decision_engine.py` returns severity + per-signal breakdown for 100% of evaluations |
| G3 | Let a non-technical operator use the system without touching code | Operator Dashboard requires no Python/CLI knowledge; all interaction is via forms and buttons |
| G4 | Support datasets beyond the bundled sample | Upload feature validates and accepts arbitrary `.xlsx`/`.csv` telemetry with a compatible schema |
| G5 | Make model comparison and selection evidence-based | Full metrics suite (accuracy, precision/recall/F1 macro & weighted, ROC-AUC, per-class report, confusion matrix) computed for all 5 models on every training run |
| G6 | Improve on hand-picked hyperparameters | Optional `RandomizedSearchCV` over each model's own search space, measurably changing model ranking (verified: XGBoost overtakes Random Forest post-tuning on the bundled sample) |
| G7 | Give the system operational memory | Every decision persisted to a local event log (SQLite), survives restarts |
| G8 | Detect when live conditions diverge from training data | Monitoring view compares live event severity mix against the training baseline on a normalized severity scale |
| G9 | Deploy anywhere with minimal setup | Runs via `python app.py` (VS Code, F5 debug config included) or `docker compose up --build` |

---

## 4. Target Users / Stakeholders

| Persona | Needs |
|---|---|
| **Plant operator / relay technician** | A single, fast-to-read verdict with a clear reason and recommended action; a history of past alarms; no need to understand ML internals. |
| **ML/data science reviewer (academic evaluator)** | Evidence of a complete, methodologically sound ML lifecycle: EDA, CDA, proper train/test protocol, honest evaluation, documented limitations. |
| **Power-systems/electrical engineer** | Confidence that the AI layer does not override or hide the underlying engineering protection thresholds; ability to inspect the physics-based feature engineering. |
| **Maintainer / future developer** | A codebase organized so that data ingestion, feature engineering, statistics, modeling, decision fusion, and presentation are cleanly separated (`src/` modules), documented, and testable. |

---

## 5. Scope

### 5.1 In scope

- Ingestion of tabular alternator telemetry (bundled sample or user upload)
- Schema validation, timestamp handling, and dynamic risk-class detection
- Physics-based feature engineering (apparent power, reactive power, load %,
  voltage deviation) from raw electrical channels
- Exploratory Data Analysis (EDA): distributions, correlations, class balance,
  time series
- Confirmatory Data Analysis (CDA): ANOVA + effect size, normality testing,
  outlier detection, correlation significance, multicollinearity (VIF), a
  zero-ML rule-based baseline, and anomaly-detector sensitivity analysis
- Unsupervised anomaly detection (Isolation Forest)
- Supervised classification across 5 model families (Random Forest, XGBoost,
  SVM, Neural Net/MLP, Logistic Regression)
- Optional hyperparameter optimization (RandomizedSearchCV) over each model's
  own parameter space
- Full evaluation metric suite and model selection by weighted F1
- A fused AI Decision Engine combining rule-based, anomaly, and ML signals
- A persisted event/alarm log with acknowledgment workflow
- An Operator Dashboard: live decision evaluation, simulated telemetry
  stepping, event history, and drift-style monitoring
- Multi-model live prediction and deep model comparison (Predictive Analytics
  page)
- Docker and VS Code deployment paths

### 5.2 Out of scope (explicitly not built)

- Real-time streaming ingestion from live SCADA/PLC hardware (the "simulated
  telemetry feed" steps through historical rows as a stand-in)
- Multi-user authentication, roles, or permissions
- Automated retraining on a schedule (retraining is manual: upload data →
  train)
- Alerting via external channels (SMS/email/PagerDuty) — alarms are
  surfaced only within the dashboard
- Support for protection schemes beyond the 4 monitored channels (MW,
  current, power factor, voltage) — e.g., frequency or differential
  protection are not modeled

---

## 6. Functional Requirements

### FR-1 Data Ingestion & Validation
- FR-1.1 The system shall load a bundled sample dataset (1,000 observations)
  on first run.
- FR-1.2 The system shall accept user-uploaded `.xlsx` or `.csv` files.
- FR-1.3 The system shall reject uploads missing any of `MW`, `Current_A`,
  `Power_Factor`, `Voltage_kV`, or `Trip_Risk`, with a specific error message
  naming the missing column(s).
- FR-1.4 The system shall synthesize a sequential timestamp when neither
  `Timestamp` nor `Date` is present, and warn the user this was done.
- FR-1.5 The system shall detect the set of risk-label classes present in
  the target column dynamically, rather than assuming a fixed 3-class set.
- FR-1.6 The system shall allow the user to override rated MW and nominal kV
  at upload time.

### FR-2 Feature Engineering
- FR-2.1 The system shall compute Apparent Power (MVA), Reactive Power
  (MVAr), Load Percent, and Voltage Deviation Percent from raw channels
  using standard three-phase AC power equations.
- FR-2.2 Engineered features shall be recomputed from the active dataset on
  every load, not trusted from any pre-baked source column.

### FR-3 Exploratory & Confirmatory Analysis
- FR-3.1 The system shall provide interactive histogram, per-class boxplot,
  scatter, and time-series views for any numeric feature.
- FR-3.2 The system shall compute and display a full correlation matrix and
  summary statistics table.
- FR-3.3 The system shall run one-way ANOVA with eta-squared effect size for
  every feature against the risk class.
- FR-3.4 The system shall run a Shapiro-Wilk normality test and report skew
  for every feature.
- FR-3.5 The system shall report IQR-fence outlier counts for every feature.
- FR-3.6 The system shall compute Variance Inflation Factors to flag
  multicollinearity.
- FR-3.7 The system shall compute a zero-ML rule-based baseline accuracy for
  comparison against trained classifiers, and shall skip this check with an
  explanation if the active dataset's classes don't match the canonical
  taxonomy.

### FR-4 Modeling
- FR-4.1 The system shall fit an Isolation Forest with a user-selectable
  contamination parameter.
- FR-4.2 The system shall train Random Forest, XGBoost, SVM (RBF), MLP, and
  Logistic Regression classifiers on request.
- FR-4.3 The system shall support both random-stratified and chronological
  train/test splitting.
- FR-4.4 The system shall optionally run RandomizedSearchCV hyperparameter
  optimization for each of the 5 models before final evaluation.
- FR-4.5 The system shall report accuracy, macro and weighted precision,
  recall, and F1, ROC-AUC (macro, one-vs-rest), a full per-class report, and
  a confusion matrix for every model.
- FR-4.6 The system shall automatically select the best model by weighted
  F1 and mark it as such in the UI.
- FR-4.7 Trained models shall be persisted to disk and reloaded without
  retraining until the next training run or dataset change.

### FR-5 Decision Engine & Operator Dashboard
- FR-5.1 The system shall compute an engineering protection-rule verdict
  from fixed setpoint thresholds independent of any trained model.
- FR-5.2 The system shall compute an anomaly verdict from the Isolation
  Forest score, mapped to the same severity scale.
- FR-5.3 The system shall compute an ML verdict from the trained ensemble's
  top prediction and confidence.
- FR-5.4 The system shall combine the three verdicts into one overall
  severity equal to the single most severe signal.
- FR-5.5 The system shall report whether the three signals agree, and a
  recommended action for the operator.
- FR-5.6 The system shall degrade gracefully — the rule-based verdict must
  always be available even if no model has been trained yet.
- FR-5.7 The system shall provide a simulated telemetry stepper that walks
  through real dataset rows as if arriving live.
- FR-5.8 Every decision evaluation shall be persisted with a timestamp,
  source (manual/simulated), inputs, and full verdict.
- FR-5.9 The system shall allow filtering the event log by severity and
  acknowledging individual events.
- FR-5.10 The system shall compare the live event severity distribution
  against the training data's baseline distribution, normalized onto a
  common severity vocabulary.

### FR-6 Predictive Analytics
- FR-6.1 The system shall run one input through all 5 trained models
  simultaneously and display each model's prediction and probabilities.
- FR-6.2 The system shall flag whether all 5 models unanimously agree.
- FR-6.3 The system shall provide a model-comparison view with F1, ROC-AUC,
  and per-class F1 across all 5 models from the most recent training run.

---

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Explainability** | Every AI-driven verdict must be traceable to its contributing signal(s); no verdict may be presented without its breakdown available on request. |
| **Deployability** | The system must run identically via a local Python virtual environment or Docker, with no code changes between the two. |
| **Data portability** | The system must not hard-code assumptions that break on a differently-labeled but schema-compatible dataset. |
| **Statistical honesty** | Reported model performance must be accompanied by the rule-based baseline and the chronological-split comparison so a reviewer can judge how much of the apparent accuracy is attributable to label circularity or split leakage rather than genuine predictive signal. |
| **No new dependencies for operational features** | The event log and hyperparameter search use only libraries already in `requirements.txt` (`sqlite3` is standard library; `RandomizedSearchCV` is already part of `scikit-learn`). |
| **Resilience** | A missing trained model or anomaly detector must not crash the Decision Engine; each signal degrades independently. |
| **Performance** | A full 5-model training run (no search) completes in a few seconds on the bundled 1,000-row dataset; a hyperparameter search (5 iterations, 3-fold CV) completes in under 15 seconds. |

---

## 8. System Architecture (Summary)

```
Telemetry (upload / bundled sample / simulated stepper)
        │
        ▼
 data_loader.py  ──── validation, schema check, class detection
        │
        ▼
 feature_engineering.py ── physics engine (MVA, load %, V-deviation)
        │
   ┌────┴─────────────────────┐
   ▼                          ▼
 eda.py / cda.py        anomaly.py (Isolation Forest)
 (descriptive +                │
  inferential stats)           ▼
                          models.py (5 classifiers,
                           optional hyperparameter search,
                           full metrics)
                                │
                                ▼
                     decision_engine.py ── fuses rule + anomaly + ML
                                │
                                ▼
                       event_log.py (SQLite persistence)
                                │
                                ▼
                    Operator Dashboard / Predictive Analytics
                          (Flask templates + Chart.js)
```

Full technical detail on each stage is provided in the companion document,
*Machine Learning Lifecycle — AI-Based Relay Monitor System*.

---

## 9. Data Requirements

| Field | Type | Required | Notes |
|---|---|---|---|
| `MW` | float | Yes | Real power output |
| `Current_A` | float | Yes | Stator current, amps |
| `Power_Factor` | float | Yes | 0–1 |
| `Voltage_kV` | float | Yes | Terminal voltage |
| `Trip_Risk` | string | Yes | Target label; any category set accepted |
| `Timestamp` / `Date` | datetime | No | Synthesized sequentially if absent |

Bundled sample: 1,000 observations, 84 calendar days, class balance
NORMAL 55.5% / WARNING 38.2% / HIGH TRIP RISK 6.3%.

---

## 10. Success Metrics

| Metric | Target | Status at submission |
|---|---|---|
| All 5 models trainable and evaluated with full metric suite | 100% | ✅ Achieved |
| Best model weighted F1 (bundled sample, random split) | > 90% | ✅ 98.8% (Random Forest / XGBoost) |
| Rule-based zero-ML baseline reported alongside ML metrics | Present | ✅ 96.1% accuracy — read as a caveat on ML "uplift" |
| Decision Engine returns a verdict even with no trained model | Required | ✅ Verified — rule-only fallback confirmed in testing |
| Event log persists across app restart | Required | ✅ SQLite file on disk |
| Hyperparameter search changes at least one model's rank | Informative | ✅ Verified — XGBoost overtook Random Forest post-search |
| Upload accepts a non-canonical label set end-to-end | Required | ✅ Verified with a `SAFE`/`DANGER` test dataset |

---

## 11. Assumptions & Constraints

- The bundled sample's `Trip_Risk` label was itself generated from
  deterministic setpoint thresholds on the same features the models consume.
  This is disclosed explicitly (see the CDA rule-based baseline) because it
  means high classifier accuracy partly reflects successful approximation of
  a known rule, not solely discovery of a hidden pattern — an important
  caveat for anyone citing the model metrics as evidence of real-world
  predictive power.
- The system assumes single-machine, single-user operation; there is no
  concurrent-write protection on the SQLite event log or the active dataset
  file beyond what SQLite/the OS provide by default.
- "Live" telemetry is simulated by replaying historical rows; no hardware
  integration exists or is claimed.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Label circularity inflating perceived model quality | Rule-based baseline surfaced prominently on the CDA page and referenced in the Predictive Analytics comparison |
| Random-split evaluation overstating real-world performance | Chronological split option provided and documented |
| Isolation Forest contamination is an uncalibrated prior | Sensitivity analysis tab shows flagged-rate and correlation-with-risk across a contamination sweep |
| Uploaded dataset with unfamiliar labels breaking downstream logic | Dynamic class detection, generic severity normalization by keyword, and graceful skip of dataset-specific checks (rule baseline) |
| Chart rendering failing silently in constrained environments | Server-driven tables are rendered independently of Chart.js so failure of one does not block the other (defensive `try/catch` around chart calls) |

## Addendum: Authentication and Reporting (v1.1)

- **FR-AUTH-1** Users register and sign in with username or email plus password (scrypt-hashed, minimum 8 characters).
- **FR-AUTH-2** Roles: viewer (read-only), operator (adds running decisions and acknowledging alarms), admin (adds data upload, model training, clearing the log, user management). The first registered user is admin.
- **FR-AUTH-3** CSRF protection on all state-changing requests; 8-hour sessions; 5-minute lockout after 5 failed attempts; the API returns JSON 401/403.
- **FR-REP-1** A printable A4 report (dataset, setpoints, model evaluation, alarm log) with selectable sections, browser Print / Save as PDF, and CSV export of the event log.

---

*End of Product Requirements Document.*
