---
title: "Machine Learning Lifecycle"
subtitle: "AI-Based Relay Monitor System — End-to-End Documentation"
author: "Final-Year Project Documentation"
date: "2026"
---

# Machine Learning Lifecycle
## AI-Based Relay Monitor System

**Document version:** 1.0
**Companion to:** Product Requirements Document — AI-Based Relay Monitor System

This document walks through all fifteen stages of the machine learning
lifecycle as implemented in this project, from raw data acquisition through
monitoring and feedback, with the exact source file, method, and — where
applicable — measured result for each stage. It also documents the parallel
operational relay-monitoring workflow the system implements on top of the ML
pipeline.

```
Data Acquisition → Data Validation/Quality → Preprocessing → Feature
Engineering → EDA → CDA → Anomaly Detection → Supervised Learning →
Hyperparameter Optimization → Model Evaluation → Model Selection →
Model Deployment → Live/Simulated Inference → Model Monitoring →
Feedback/Retraining
```

---

## Stage 1 — Data Acquisition

**Implementation:** `src/data_loader.py`, `src/state.py`

The system supports two acquisition paths:

1. **Bundled sample** — `data/alternator_data.xlsx`, 1,000 observations
   across 84 calendar days (2026-01-01 to 2026-03-25), with columns for raw
   telemetry (`MW`, `Current_A`, `Power_Factor`, `Voltage_kV`), pre-computed
   engineering fields, and the target `Trip_Risk`.
2. **User upload** — `.xlsx` or `.csv` via the Upload Data page, handled by
   `data_loader.validate_and_prepare()`.

The active dataset is tracked at runtime (not hard-coded) via `src/state.py`,
which stores the current file path, user-overridable nameplate constants
(rated MW, nominal kV), and the dynamically detected set of risk classes.
This allows the system to acquire genuinely new data without a code change
or restart.

**Design decision:** acquisition is deliberately decoupled from every
downstream stage — nothing past `data_loader.py` assumes the bundled
sample's specific column values, only its schema.

---

## Stage 2 — Data Validation / Quality

**Implementation:** `data_loader.validate_and_prepare()`, `DatasetValidationError`

Every uploaded file passes through:

- **Schema check** — `MW`, `Current_A`, `Power_Factor`, `Voltage_kV`, and
  `Trip_Risk` must all be present; missing columns are named explicitly in
  the rejection message rather than a generic failure.
- **Type coercion** — all four numeric columns are coerced with
  `pd.to_numeric(..., errors="coerce")`; rows that fail to coerce are
  dropped, with a warning reporting exactly how many rows were dropped out
  of how many.
- **Minimum row count** — fewer than 20 usable rows after cleaning is
  rejected outright (too few to train meaningfully).
- **Timestamp handling** — `Timestamp` is derived from `Date` if only one is
  present, or synthesized as a sequential hourly series if neither exists,
  with a warning either way so the gap is never silent.
- **Class cardinality check** — fewer than 2 distinct target classes is
  rejected (nothing to classify); more than 8 is accepted but flagged as
  likely to crowd the charts.

**Verified behavior:** a file missing `Power_Factor` is rejected with
`"Missing required column(s): Power_Factor..."`; a file with no timestamp
column at all is accepted with the warning `"No 'Timestamp' or 'Date' column
found -- generated a synthetic sequential timestamp."` — both confirmed by
direct testing during development.

---

## Stage 3 — Preprocessing

**Implementation:** `data_loader.py`, `app.py::get_full_df()`

Beyond validation, preprocessing includes:

- Sorting all rows chronologically by `Timestamp` (required for the
  chronological train/test split used later).
- Assigning a synthetic `Observation_ID` if the source data lacks one.
- Trimming whitespace and normalizing the target column to string type,
  so `"NORMAL "` and `"NORMAL"` are not treated as different classes.
- Caching the preprocessed dataframe in memory (`app._CACHE`) so repeated
  page loads don't re-read and re-clean the file on every request.

---

## Stage 4 — Feature Engineering

**Implementation:** `src/feature_engineering.py`

Four features are derived from the four raw channels using standard
three-phase AC power relationships, recomputed live rather than trusted from
any pre-existing column in the source file:

| Feature | Formula |
|---|---|
| Apparent Power (MVA) | `√3 × Voltage_kV × Current_A / 1000` |
| Reactive Power (MVAr) | `MVA × sin(arccos(Power_Factor))` |
| Load Percent | `MW / rated_MW × 100` |
| Voltage Deviation Percent | `(Voltage_kV − rated_kV) / rated_kV × 100` |

`rated_MW` and `rated_kV` are read from runtime state (`src/state.py`), not
hard-coded, so an uploaded dataset from a differently-rated unit computes
correct engineered features once the user supplies its nameplate values on
the Upload page.

**Documented caveat:** MVA and Load Percent are *exact deterministic
functions* of features already in the model's input set (`MW`, `Current_A`,
`Voltage_kV`). This is intentional physics-informed feature engineering, but
it introduces multicollinearity that must be accounted for when interpreting
model feature importances later (see Stage 6, Variance Inflation Factor).

---

## Stage 5 — Exploratory Data Analysis (EDA)

**Implementation:** `src/eda.py`, `templates/eda.html`

Purely descriptive, non-inferential views, all interactive and driven by the
live dataset:

- **Summary statistics** — mean, std, min, max, skew, and kurtosis for every
  numeric feature.
- **Correlation matrix** — full pairwise Pearson correlation across raw and
  engineered features, rendered as a heat-mapped table.
- **Histogram** — distribution of any selected feature.
- **Boxplot by risk class** — spread of a feature within each risk category.
- **Scatter** — any two features plotted against each other, colored by
  risk class.
- **Time series** — any feature plotted over the dataset's time span.
- **Class balance** — count and share per risk class.

**Measured result (bundled sample):** MW and Current_A correlate at r ≈
0.86; class balance is NORMAL 555 / WARNING 382 / HIGH TRIP RISK 63 — a
real-world-plausible imbalance the modeling stage must account for (hence
weighted, not only macro, metrics later).

---

## Stage 6 — Confirmatory Data Analysis (CDA)

**Implementation:** `src/cda.py`, `templates/cda.html`

Where EDA describes, CDA tests. Seven checks, each addressing a specific
methodological question:

1. **One-way ANOVA + eta-squared** — does each feature's mean genuinely
   differ across risk classes, and how large is the effect (not just
   whether p < 0.05, which large sample sizes make easy to satisfy trivially)?
2. **Shapiro-Wilk normality test** — many classical statistics (ANOVA,
   Pearson correlation) assume approximate normality; this quantifies how
   often that assumption actually holds for this data (in practice, most
   electrical telemetry features here fail normality, favoring tree-based
   models over linear ones downstream).
3. **IQR-fence outlier detection** — a distribution-agnostic outlier count
   per feature, independent of the normality result.
4. **Pearson correlation significance** — r, p-value, and a 95% confidence
   interval (Fisher z-transform) for any feature pair, so a specific
   correlation claim can be checked rather than eyeballed from the matrix.
5. **Variance Inflation Factor (VIF)** — quantifies the multicollinearity
   flagged in Stage 4; `Calculated_MVA` and `Load_Percent` score high VIF as
   expected, which is the documented caveat on trusting raw feature
   importance as a causal signal in Stage 10.
6. **Rule-based zero-ML baseline** — reconstructs the risk label using
   *only* the deterministic setpoint thresholds (no model at all) and
   reports the accuracy. **Measured result: 96.1%** on the bundled sample.
   This is the single most important number in the whole CDA page: because
   the sample's label was itself generated from a similar rule, any trained
   classifier's accuracy must be read relative to this baseline, not in
   isolation. Skipped gracefully (with an explanation) for uploaded datasets
   whose classes don't match the canonical taxonomy.
7. **Isolation Forest contamination sensitivity** — re-fits the anomaly
   detector across several contamination values and reports how the flagged
   rate and its correlation with actual risk shift, since contamination has
   no empirically calibrated value in the source data.

---

## Stage 7 — Anomaly Detection

**Implementation:** `src/anomaly.py`

An Isolation Forest is fit on all raw + engineered features with **no
labels used** — it learns what "normal" looks like purely from the shape of
the data. Its output (`decision_function`, sign-flipped so higher = more
anomalous) becomes one additional engineered feature, `Anomaly_Score`, fed
into the supervised stage.

**Design rationale for Isolation Forest specifically:** it isolates
anomalies via random recursive partitioning rather than a distance or
density metric, giving robust behavior on mixed-scale features without
requiring a scaling step, and O(n log n) training cost appropriate for
on-demand retraining from the UI.

**Operational role:** beyond feeding the supervised stage, the anomaly score
is one of the three signals the Decision Engine (Stage 12 onward) fuses
directly — a reading can be flagged CRITICAL for being a statistical
outlier even if it breaches no engineering threshold.

---

## Stage 8 — Supervised Learning

**Implementation:** `src/models.py::train_all_models()`

Five classifiers, deliberately spanning different inductive biases so the
*shape* of the decision boundary can be diagnosed from which family wins:

| Model | Family | Feature scaling |
|---|---|---|
| Random Forest | Tree ensemble, bagging | None (scale-invariant) |
| XGBoost | Tree ensemble, boosting | None (scale-invariant) |
| SVM (RBF kernel) | Kernel method | Standardized |
| Neural Net (MLP) | Shallow feed-forward | Standardized |
| Logistic Regression | Linear | Standardized |

Two split strategies are supported:

- **Random (stratified)** — standard practice, but can leak autocorrelated
  day-to-day operating conditions from test into train.
- **Chronological** — train on the earliest N% of days, test on the most
  recent — the more defensible protocol for anything resembling a real
  deployment decision, since it prevents that leakage.

**Measured result (bundled sample, random split, no optimization):** tree
ensembles clearly outperform the linear model (Random Forest and XGBoost
both ≈98.8% weighted F1 vs. Logistic Regression ≈92%) — consistent with a
label generated by sharp, axis-aligned threshold rules, which trees fit
natively and linear/kernel boundaries can only approximate.

---

## Stage 9 — Hyperparameter Optimization

**Implementation:** `src/models.py::_fit_with_optional_search()`,
`PARAM_DISTRIBUTIONS`

An optional `RandomizedSearchCV` pass searches each of the 5 models' **own**
hyperparameter space — no new model family is introduced by this stage:

| Model | Parameters searched |
|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| SVM | `C`, `gamma` |
| MLP | `hidden_layer_sizes`, `alpha`, `learning_rate_init` |
| Logistic Regression | `C`, `solver` |

Cross-validation folds are automatically clamped to the smallest class
count present in the training split (`max(2, min(cv, min_class_count))`),
so a rare class (the bundled sample's HIGH TRIP RISK class is only ~6% of
rows) never causes a stratified-fold error; if a class has fewer than 2
members the search is skipped for that model with a plain fit, rather than
crashing the whole run.

**Measured result:** a 5-model search (5 iterations, 3-fold CV) completed
in **13 seconds** end-to-end on the bundled sample and **changed the model
ranking** — XGBoost (tuned to `max_depth=5, learning_rate=0.01,
n_estimators=100, subsample=0.85`) overtook the previously-best Random
Forest. This is reported to the UI verbatim (best hyperparameters per model)
so the improvement is auditable, not just asserted.

---

## Stage 10 — Model Evaluation

**Implementation:** `src/models.py::_metrics()`

Every model — not only the eventual "best" one — receives the full
evaluation suite on the held-out test set:

- Accuracy
- Precision, Recall, F1 — both **macro** average (unweighted across
  classes, penalizing poor performance on the rare HIGH TRIP RISK class) and
  **weighted** average (reflecting overall row-level performance)
- ROC-AUC, macro one-vs-rest (only computed when probability estimates are
  available and more than one class is present)
- A full per-class precision/recall/F1/support breakdown
  (`classification_report`)
- A confusion matrix

**Why both macro and weighted:** with a 55/38/6% class split, a model could
achieve a deceptively high *weighted* score by nailing the two common
classes while doing poorly on the rare CRITICAL class — exactly the class
where an operational monitoring system cannot afford to be wrong. Reporting
macro alongside weighted makes that failure mode visible rather than hidden
by the class imbalance.

**Feature importance caveat:** Random Forest Gini importance is reported,
but cross-referenced against the Stage 6 VIF results — `Current_A` and `MW`
dominate importance (~56% and ~15% respectively in one measured run), but
because `Calculated_MVA` and `Load_Percent` are algebraic functions of those
same inputs, the importance split partly reflects arbitrary credit
allocation among collinear features, not a clean causal ranking.

---

## Stage 11 — Model Selection

**Implementation:** `src/models.py` (`best_model_name = max(results, key=...)`)

The model with the highest **weighted F1** on the held-out test set is
automatically selected and flagged (★) throughout the UI — in the Train
page's metrics table, and as the default model for Predictive Analytics.
Selection criteria and the runner-up scores remain visible (not hidden once
a winner is picked), so a reviewer can judge whether the margin between the
best and second-best model is meaningful or noise.

---

## Stage 12 — Model Deployment

**Implementation:** `src/models.py::load_supervised_bundle()`,
`joblib.dump()` / `joblib.load()`

Trained models (all 5, not just the winner — needed for multi-model
comparison), the fitted `StandardScaler`, the `LabelEncoder`, and the
feature column order are serialized to `models/supervised_bundle.joblib`;
the Isolation Forest is serialized separately to
`models/isolation_forest.joblib`. Both are Docker-volume-mounted so they
persist across container restarts without retraining.

**Deployment trigger discipline:** uploading a new dataset automatically
deletes both `.joblib` files (`app.py::api_upload`), since a model trained
on the old schema/distribution should never silently keep serving
predictions against new data — the user is explicitly told to retrain.

---

## Stage 13 — Live / Simulated Inference

**Implementation:** `src/models.py::predict_single()` /
`predict_all_models()`, `src/decision_engine.py::evaluate()`,
`app.py::/api/telemetry/simulate_next`

Three inference surfaces:

1. **Single-model prediction** (`predict_single`) — used internally by the
   Decision Engine.
2. **Multi-model prediction** (`predict_all_models`) — the Predictive
   Analytics page runs one input through all 5 models simultaneously and
   reports an agreement/disagreement flag.
3. **Simulated live telemetry** — the Operator Dashboard's stepper walks
   through real historical rows from the active dataset one at a time (with
   optional auto-play), running the full Decision Engine on each as if it
   had just arrived — a stand-in for a real streaming source, explicitly
   documented as such rather than presented as genuine real-time hardware
   integration.

Every inference on the Operator Dashboard is passed through the fused
Decision Engine (see the companion PRD, Section "The AI Decision Engine")
rather than exposed as a raw model output, and is persisted to the event log
(Stage 14).

---

## Stage 14 — Model Monitoring

**Implementation:** `src/event_log.py`, `app.py::/api/monitoring/summary`

Two complementary monitoring mechanisms:

1. **Event log as ground truth for "what actually happened"** — every
   decision (manual or simulated) is written to `data/events.db` (SQLite)
   with its full verdict, inputs, and timestamp, independent of the
   dataframe cache, and survives app restarts.
2. **Drift-style comparison** — `/api/monitoring/summary` compares the
   severity mix of recently logged events against the training dataset's
   own class distribution. Both sides are normalized onto the same 3-tier
   severity vocabulary (`decision_engine.normalize_severity()`) before
   comparing — an earlier version of this comparison mixed the raw label
   spelling (`"HIGH TRIP RISK"`) against the fused output spelling
   (`"CRITICAL"`) and showed drift that was really just a naming mismatch;
   this was caught during testing and fixed by routing both sides through
   the same normalization function.

This gives a plain-language drift signal — "18% of recent events were
CRITICAL vs. 6% in training" — without requiring ground-truth labels on live
data, which in a real deployment typically arrive late or never.

---

## Stage 15 — Feedback / Retraining

**Implementation:** Upload page (`templates/upload.html`) →
`api_upload` → Train page

The retraining loop in this system is **manual and explicit** rather than
automatic and scheduled — a deliberate choice for a decision-support tool
where silently replacing a deployed model carries real operational risk:

1. An operator or engineer uploads a new telemetry export (Stage 1–3 rerun).
2. The system automatically invalidates cached data and deletes the
   previously trained model artifacts (Stage 12's deployment discipline).
3. A visible prompt directs the user to the Train page to retrain
   explicitly (Stage 8–11 rerun) before any prediction/decision endpoint
   will work again.
4. The event log (Stage 14) is **not** cleared on upload, preserving
   historical operational context across a model refresh, though it can be
   cleared manually from the Operator Dashboard if a clean slate is wanted.

**Explicitly out of scope:** automated scheduled retraining, drift-triggered
retraining, and online/incremental learning are not implemented; the
Monitoring stage (14) surfaces the *signal* that retraining may be needed,
but the *action* remains a human decision, consistent with this being a
decision-support system rather than an autonomous control system.

---

## Cross-Cutting: The Operational Relay-Monitoring Workflow

The fifteen ML-lifecycle stages above are wrapped by a parallel operational
workflow, implemented primarily in `src/decision_engine.py` and the Operator
Dashboard:

```
Telemetry → Data Quality → Engineering Protection Rules → AI Anomaly
Detection → AI Risk Prediction → AI Decision Engine → Alarm/Warning/
Critical Event → Operator Dashboard → Historical Event Analysis
```

The key engineering decision tying the two workflows together is that
**the Decision Engine's overall severity is always the single most severe
of its three inputs, never an average** — a design chosen so that a calm
majority (e.g., rules and ML both say NORMAL) can never mask one alarming
signal (e.g., anomaly detection flagging a statistically unprecedented
reading). This is the concrete mechanism by which "AI Risk Prediction" in
the operational workflow and "Model Evaluation/Selection" in the ML
lifecycle are connected to a real, auditable, safety-conscious decision
rather than a bare model output.

---

## Summary Table: Lifecycle Stage → Source File → Verified Result

| Stage | Source file(s) | Verified result |
|---|---|---|
| 1. Data Acquisition | `data_loader.py`, `state.py` | Bundled sample + upload both load correctly |
| 2. Validation/Quality | `data_loader.py` | Missing-column and no-timestamp cases both handled correctly |
| 3. Preprocessing | `data_loader.py`, `app.py` | Chronological sort, ID synthesis confirmed |
| 4. Feature Engineering | `feature_engineering.py` | 4 engineered features computed from runtime nameplate state |
| 5. EDA | `eda.py` | MW–Current r≈0.86; class split 55.5/38.2/6.3% |
| 6. CDA | `cda.py` | Rule baseline 96.1% accuracy; VIF flags MVA/Load% collinearity |
| 7. Anomaly Detection | `anomaly.py` | Isolation Forest score feeds both ML and Decision Engine |
| 8. Supervised Learning | `models.py` | RF/XGBoost ≈98.8% F1 vs. LogReg ≈92% (random split) |
| 9. Hyperparameter Optimization | `models.py` | 13s search, XGBoost overtakes RF post-tuning |
| 10. Model Evaluation | `models.py` | Full macro+weighted+ROC-AUC+per-class suite confirmed |
| 11. Model Selection | `models.py` | Best-F1 auto-selection confirmed, ★ flagged in UI |
| 12. Model Deployment | `models.py`, joblib | Persists across restart; deleted on new upload |
| 13. Live/Simulated Inference | `decision_engine.py`, `app.py` | Manual + simulated stepper both confirmed working |
| 14. Model Monitoring | `event_log.py`, `app.py` | Drift comparison bug found and fixed during testing |
| 15. Feedback/Retraining | `upload.html`, `app.py` | Model invalidation on upload confirmed |

---

*End of Machine Learning Lifecycle document.*
