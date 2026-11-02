# 3 MW Alternator — ML Lifecycle Dashboard

A Flask dashboard implementing the full machine learning lifecycle for
alternator trip-risk prediction: data ingestion → EDA → physics-informed
feature engineering → confirmatory data analysis (CDA) → unsupervised
anomaly detection (Isolation Forest) → supervised classification (5 models)
→ live interactive inference.

## Project structure

```
alternator_dashboard/
├── app.py                     # Flask app: pages + JSON API
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   └── alternator_data.xlsx   # source telemetry (swap this file to use new data)
├── models/                    # trained model artifacts land here (.joblib)
├── src/
│   ├── config.py              # thresholds, feature lists, nameplate constants
│   ├── data_loader.py         # Stage 0/1: ingestion + data-health report
│   ├── feature_engineering.py # Stage 2: physics engine (MVA, load%, etc.)
│   ├── anomaly.py             # Stage 3: Isolation Forest
│   ├── models.py              # Stage 4: RF / XGBoost / SVM / MLP / LogReg
│   ├── eda.py                 # descriptive stats, histograms, correlations
│   └── cda.py                 # ANOVA, correlation significance, VIF,
│                               # rule-based baseline, contamination sensitivity
├── templates/                 # Jinja2 pages (overview, EDA, pipeline, CDA, train, predict)
└── static/                    # CSS + JS (Chart.js-driven interactivity)
```

## Run locally in VS Code

1. Open this folder in VS Code.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
   Or press **F5** in VS Code — a debug configuration is already provided
   in `.vscode/launch.json`.
4. Open **http://localhost:5000**.

## Run with Docker

```bash
docker build -t alternator-dashboard .
docker run -p 5000:5000 -v $(pwd)/models:/app/models alternator-dashboard
```

Or with Docker Compose (also mounts `data/` so you can swap datasets without rebuilding):

```bash
docker compose up --build
```

Then open **http://localhost:5000**.

## Using the dashboard

| Page | What it does |
|---|---|
| **Overview** | Dataset health check: row count, date range, missing values, class balance. |
| **EDA** | Pick any feature to update histogram, per-class distribution, scatter, and time series together. Correlation matrix and summary stats update independently. |
| **Physics engine** | Live calculator for the Stage 2 formulas (MVA, reactive power, load %, voltage deviation) plus the setpoint threshold table. |
| **CDA** | Tabbed hypothesis tests: ANOVA across risk classes, Pearson correlation significance + CI, Variance Inflation Factor (collinearity), a **rule-based baseline** that reconstructs the label with zero ML (the key check on label circularity), and Isolation Forest contamination sensitivity. |
| **Train** | Choose random vs. chronological train/test split, test size, and Isolation Forest contamination, then train all 5 classifiers live. Results include F1 comparison, RF feature importance, and a per-model confusion matrix. Models persist to `models/*.joblib`. |
| **Predict** | Live inference: enter raw telemetry (or click a preset), pick a trained model, and see the predicted risk class, class probabilities, engineered features, and anomaly score. |

## Swapping in new data

Replace `data/alternator_data.xlsx`, keeping the same sheet name
(`Alternator_Data`) and column names (see `src/config.py` for the expected
raw/engineered feature lists). Restart the app — no code changes required
as long as the schema matches.

## Notes on methodology baked into this dashboard

- The CDA page's **rule-based baseline** is the single most important
  check here: because `Trip_Risk` was generated from deterministic
  setpoint thresholds on the same features the model sees, a classifier's
  high F1 score should always be read relative to how well the raw
  thresholds already reconstruct the label — not treated as evidence of a
  hidden, ML-discovered pattern.
- The Train page's **chronological split** option exists because a random
  split can leak autocorrelated day-to-day operating conditions between
  train and test. If chronological F1 is materially lower than random-split
  F1, trust the chronological number for anything resembling a deployment
  decision.
- **VIF** is reported because `Calculated_MVA` and `Load_Percent` are exact
  algebraic functions of `MW` / `Current_A` / `Voltage_kV`. High VIF on
  those features is expected and is a caveat on interpreting Random Forest
  Gini importance as a causal ranking.
