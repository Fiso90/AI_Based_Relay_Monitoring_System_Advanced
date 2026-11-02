"""
Confirmatory Data Analysis (CDA) — hypothesis-driven, inferential.

This module operationalizes a simple discipline: don't just report a
descriptive number (a Gini importance, a correlation, a mean difference)
and call it meaningful -- run the actual statistical test that either
supports or undercuts the claim, quantify uncertainty and effect size, and
surface caveats (collinearity, label circularity, non-normality) directly
in the output.

Tests implemented:
  1. One-way ANOVA + eta-squared effect size per feature across risk classes
  2. Shapiro-Wilk normality test per feature (per class, since ANOVA and
     many downstream stats assume approximate normality within groups)
  3. IQR-based outlier count per feature
  4. Pearson correlation significance (r, p-value, 95% CI) for any pair
  5. Variance Inflation Factor (VIF) per feature (collinearity)
  6. Rule-based baseline: how much of the target can be reconstructed with
     zero ML, using only the deterministic setpoint thresholds
  7. Isolation Forest contamination sensitivity
"""
import numpy as np
import pandas as pd
from scipy import stats
from . import config
from . import state

NUMERIC_COLS = config.RAW_FEATURES + config.ENGINEERED_FEATURES


def anova_by_risk_class(df: pd.DataFrame) -> list:
    """
    One-way ANOVA across risk-class groups, plus eta-squared (the share of
    total variance in the feature explained by class membership) so a
    "statistically significant" result can also be judged for how large the
    effect actually is -- large n makes tiny effects significant too.
    """
    rows = []
    classes = state.get_classes()
    groups_by_class = {cls: df[df[config.TARGET] == cls] for cls in classes}
    for feat in NUMERIC_COLS:
        samples = [g[feat].dropna().values for g in groups_by_class.values() if len(g) > 1]
        if len(samples) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*samples)

        all_vals = np.concatenate(samples)
        grand_mean = all_vals.mean()
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        rows.append({
            "feature": feat,
            "f_statistic": round(float(f_stat), 3),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
            "eta_squared": round(eta_sq, 4),
            "effect_size": _eta_label(eta_sq),
        })
    return sorted(rows, key=lambda r: r["f_statistic"], reverse=True)


def _eta_label(eta_sq: float) -> str:
    if eta_sq < 0.01:
        return "negligible"
    if eta_sq < 0.06:
        return "small"
    if eta_sq < 0.14:
        return "medium"
    return "large"


def normality_tests(df: pd.DataFrame) -> list:
    """
    Shapiro-Wilk test per feature (on the whole column, subsampled if
    large since Shapiro-Wilk is sensitive/slow past a few thousand points).
    A significant result (p < 0.05) means the feature is NOT well-described
    by a normal distribution -- relevant because ANOVA and Pearson
    correlation both lean on approximate normality assumptions.
    """
    rows = []
    rng = np.random.default_rng(config.RANDOM_STATE)
    for feat in NUMERIC_COLS:
        vals = df[feat].dropna().values
        if len(vals) > 4999:
            vals = rng.choice(vals, size=4999, replace=False)
        if len(vals) < 8:
            continue
        stat_w, p_val = stats.shapiro(vals)
        skew = float(stats.skew(vals))
        rows.append({
            "feature": feat,
            "shapiro_w": round(float(stat_w), 4),
            "p_value": float(p_val),
            "likely_normal": bool(p_val >= 0.05),
            "skew": round(skew, 3),
        })
    return sorted(rows, key=lambda r: r["p_value"])


def outlier_summary(df: pd.DataFrame) -> list:
    """
    IQR-fence outlier count per feature: values outside
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. A classic, distribution-agnostic screen
    that complements the skew/kurtosis already shown in EDA.
    """
    rows = []
    n = len(df)
    for feat in NUMERIC_COLS:
        vals = df[feat].dropna()
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((vals < lo) | (vals > hi)).sum())
        rows.append({
            "feature": feat,
            "lower_fence": round(float(lo), 3),
            "upper_fence": round(float(hi), 3),
            "n_outliers": n_out,
            "pct_outliers": round(100 * n_out / n, 2) if n else 0.0,
        })
    return sorted(rows, key=lambda r: r["n_outliers"], reverse=True)


def correlation_significance(df: pd.DataFrame, feat_x: str = "MW", feat_y: str = "Current_A") -> dict:
    """Pearson r + p-value + 95% CI (Fisher z-transform) for a feature pair."""
    x = df[feat_x].dropna()
    y = df[feat_y].dropna()
    n = min(len(x), len(y))
    r, p = stats.pearsonr(x.iloc[:n], y.iloc[:n])

    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(max(n - 3, 1))
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
    """VIF_i = 1 / (1 - R_i^2). VIF > 5 signals problematic collinearity."""
    from numpy.linalg import lstsq
    X = df[NUMERIC_COLS].dropna().copy()
    X = (X - X.mean()) / X.std()
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
    Reconstruct the risk label using ONLY the deterministic setpoint
    thresholds (no ML) and compare against the true labels. Only meaningful
    for the canonical NORMAL / WARNING / HIGH TRIP RISK taxonomy this
    project was designed around; if an uploaded dataset uses different
    label names, this check is skipped with an explanation rather than
    producing a misleading number.
    """
    classes = state.get_classes()
    if not {"NORMAL", "WARNING", "HIGH TRIP RISK"}.issubset(set(classes)):
        return {
            "applicable": False,
            "interpretation": (
                "This dataset's risk labels don't match the NORMAL / WARNING / "
                "HIGH TRIP RISK taxonomy the setpoint rule was written for, so "
                "the zero-ML baseline check is skipped for this dataset."
            ),
        }

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
    for cls in classes:
        mask = actual == cls
        if mask.sum() > 0:
            agreement_by_class[cls] = float((predicted[mask] == actual[mask]).mean())

    return {
        "applicable": True,
        "overall_accuracy": round(accuracy, 4),
        "agreement_by_class": {k: round(v, 4) for k, v in agreement_by_class.items()},
        "interpretation": (
            "This simple threshold rule (no machine learning) reconstructs "
            f"the risk label with {accuracy*100:.1f}% accuracy. Any ML classifier's "
            "reported metrics should be interpreted relative to this baseline, "
            "since the label was generated by a similar deterministic rule."
        ),
    }


def contamination_sensitivity(df: pd.DataFrame, contaminations=(0.05, 0.1, 0.15, 0.2)) -> list:
    """
    Re-fits Isolation Forest at several contamination levels and reports
    how the anomaly flag rate and its correlation with the actual risk
    label shifts, since this hyperparameter has no empirically calibrated
    value in the source data.
    """
    from sklearn.ensemble import IsolationForest
    feats = config.RAW_FEATURES + config.ENGINEERED_FEATURES
    X = df[feats]
    rows = []
    normal_label = "NORMAL" if "NORMAL" in state.get_classes() else state.get_classes()[0]
    risk_binary = (df[config.TARGET] != normal_label).astype(int)
    for c in contaminations:
        model = IsolationForest(n_estimators=200, contamination=c, random_state=config.RANDOM_STATE)
        flags = model.fit_predict(X)
        anomaly_flag = (flags == -1).astype(int)
        flagged_rate = float(anomaly_flag.mean())
        corr = float(np.corrcoef(anomaly_flag, risk_binary)[0, 1])
        rows.append({
            "contamination": c,
            "flagged_rate": round(flagged_rate, 4),
            "correlation_with_actual_risk": round(corr, 4),
        })
    return rows
