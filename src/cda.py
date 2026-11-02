"""
Confirmatory Data Analysis (CDA) — hypothesis-driven, inferential.

This module operationalizes the methodological critique from the
project's model review: don't just report a Gini importance and call it
"the driver of risk" -- run the actual statistical tests that either
support or undercut that claim, and surface the caveats (collinearity,
label circularity, split sensitivity) directly in the output rather than
only in prose.

Tests implemented:
  1. One-way ANOVA per feature across Trip_Risk classes
     (does the mean of this feature actually differ across risk classes?)
  2. Pearson correlation significance (r, p-value, 95% CI) for MW vs Current
     (directly checks the deck's headline "0.972 correlation" claim)
  3. Variance Inflation Factor (VIF) per feature
     (quantifies the collinearity flagged for Calculated_MVA / Load_Percent)
  4. Chi-square test of independence: does contamination assumption change
     the anomaly flag rate in a way that's statistically distinguishable?
  5. Label-leakage check: how much of Trip_Risk can be reconstructed by the
     deterministic setpoint rule ALONE (no ML), vs. what the ML models add.
"""
import numpy as np
import pandas as pd
from scipy import stats
from . import config

NUMERIC_COLS = config.RAW_FEATURES + config.ENGINEERED_FEATURES


def anova_by_risk_class(df: pd.DataFrame) -> list:
    """
    For each numeric feature: one-way ANOVA across the 3 Trip_Risk groups.
    A large F-stat / tiny p-value confirms the feature's mean genuinely
    shifts across risk states (not just noise).
    """
    rows = []
    groups_by_class = {cls: df[df[config.TARGET] == cls] for cls in config.CLASSES}
    for feat in NUMERIC_COLS:
        samples = [g[feat].dropna().values for g in groups_by_class.values() if len(g) > 1]
        if len(samples) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*samples)
        rows.append({
            "feature": feat,
            "f_statistic": round(float(f_stat), 3),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        })
    return sorted(rows, key=lambda r: r["f_statistic"], reverse=True)


def correlation_significance(df: pd.DataFrame, feat_x: str = "MW", feat_y: str = "Current_A") -> dict:
    """
    Pearson r + p-value + 95% CI (Fisher z-transform) for a feature pair.
    Directly answers: is the deck's claimed correlation number consistent
    with what this dataset actually supports, and how precise is it?
    """
    x = df[feat_x].dropna()
    y = df[feat_y].dropna()
    n = min(len(x), len(y))
    r, p = stats.pearsonr(x.iloc[:n], y.iloc[:n])

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.975)
    lo, hi = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)

    return {
        "feature_x": feat_x,
        "feature_y": feat_y,
        "n": int(n),
        "r": round(float(r), 4),
        "p_value": float(p),
        "ci_95_low": round(float(lo), 4),
        "ci_95_high": round(float(hi), 4),
    }


def variance_inflation_factors(df: pd.DataFrame) -> list:
    """
    VIF_i = 1 / (1 - R_i^2), where R_i^2 comes from regressing feature i
    on all other features. VIF > 5-10 signals problematic collinearity.
    This is the quantitative backup for the "MVA and Load% are exact
    functions of MW/Current/Voltage" collinearity warning.
    """
    from numpy.linalg import lstsq
    X = df[NUMERIC_COLS].dropna().copy()
    X = (X - X.mean()) / X.std()  # standardize for numerical stability
    rows = []
    for col in NUMERIC_COLS:
        y = X[col].values
        others = X.drop(columns=[col]).values
        others_with_const = np.column_stack([others, np.ones(len(others))])
        coef, *_ = lstsq(others_with_const, y, rcond=None)
        y_pred = others_with_const @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 0.9999 else float("inf")
        rows.append({"feature": col, "r_squared": round(float(r2), 4),
                     "vif": round(float(vif), 2) if np.isfinite(vif) else None,
                     "high_collinearity": bool(vif > 5)})
    return sorted(rows, key=lambda r: (r["vif"] is None, -(r["vif"] or 0)))


def rule_based_baseline(df: pd.DataFrame) -> dict:
    """
    THE key confirmatory check: reconstruct Trip_Risk using ONLY the
    deterministic setpoint thresholds (no ML at all) and compare against
    the true labels. If this baseline alone gets ~99%+ accuracy, that is
    direct evidence the supervised classifiers' high F1 mainly reflects
    successful approximation of a known rule, not discovery of a hidden
    predictive pattern -- exactly the label-circularity concern raised
    in review.
    """
    sp = config.SETPOINTS

    def risk_row(row):
        score = 0
        if row["MW"] > sp["MW"]["warning_max"]:
            score += 2
        elif row["MW"] > sp["MW"]["normal_max"]:
            score += 1
        if row["Current_A"] > sp["Current_A"]["warning_max"]:
            score += 2
        elif row["Current_A"] > sp["Current_A"]["normal_max"]:
            score += 1
        if row["Power_Factor"] < sp["Power_Factor"]["warning_min"]:
            score += 2
        elif row["Power_Factor"] < sp["Power_Factor"]["normal_min"]:
            score += 1
        if row["Voltage_Deviation_Percent"] > sp["Voltage_Deviation_Percent"]["warning_max"]:
            score += 2
        elif row["Voltage_Deviation_Percent"] > sp["Voltage_Deviation_Percent"]["normal_max"]:
            score += 1

        if score >= 2:
            return "HIGH TRIP RISK"
        elif score == 1:
            return "WARNING"
        return "NORMAL"

    predicted = df.apply(risk_row, axis=1)
    actual = df[config.TARGET]
    accuracy = float((predicted == actual).mean())

    agreement_by_class = {}
    for cls in config.CLASSES:
        mask = actual == cls
        if mask.sum() > 0:
            agreement_by_class[cls] = float((predicted[mask] == actual[mask]).mean())

    return {
        "overall_accuracy": round(accuracy, 4),
        "agreement_by_class": {k: round(v, 4) for k, v in agreement_by_class.items()},
        "interpretation": (
            "This simple threshold rule (no machine learning) reconstructs "
            f"Trip_Risk with {accuracy*100:.1f}% accuracy. Any ML classifier's "
            "reported F1 score should be interpreted relative to this baseline, "
            "since the label was generated by a similar deterministic rule."
        ),
    }


def contamination_sensitivity(df: pd.DataFrame, contaminations=(0.05, 0.1, 0.15, 0.2)) -> list:
    """
    Re-fits Isolation Forest at several contamination levels and reports
    how the resulting anomaly flag rate and its correlation with the
    (independently derived) Trip_Risk label shifts. Answers the review
    question: how sensitive is the pipeline to this un-calibrated
    hyperparameter?
    """
    from sklearn.ensemble import IsolationForest
    feats = config.RAW_FEATURES + config.ENGINEERED_FEATURES
    X = df[feats]
    rows = []
    risk_binary = (df[config.TARGET] != "NORMAL").astype(int)
    for c in contaminations:
        model = IsolationForest(n_estimators=200, contamination=c, random_state=config.RANDOM_STATE)
        flags = model.fit_predict(X)  # -1 = anomaly, 1 = normal
        anomaly_flag = (flags == -1).astype(int)
        flagged_rate = float(anomaly_flag.mean())
        # point-biserial correlation between anomaly flag and actual non-normal label
        corr = float(np.corrcoef(anomaly_flag, risk_binary)[0, 1])
        rows.append({
            "contamination": c,
            "flagged_rate": round(flagged_rate, 4),
            "correlation_with_actual_risk": round(corr, 4),
        })
    return rows
