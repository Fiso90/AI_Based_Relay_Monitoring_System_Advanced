"""
Exploratory Data Analysis (EDA) — descriptive, non-inferential.

Everything here answers "what does the data look like": distributions,
correlations, summary stats, class balance. No hypothesis tests live
here — those belong in cda.py. Keeping the split explicit mirrors the
classic EDA/CDA distinction (Tukey's exploratory vs. confirmatory analysis).
"""
import numpy as np
import pandas as pd
from . import config

NUMERIC_COLS = config.RAW_FEATURES + config.ENGINEERED_FEATURES


def summary_statistics(df: pd.DataFrame) -> dict:
    desc = df[NUMERIC_COLS].describe().T
    desc["skew"] = df[NUMERIC_COLS].skew()
    desc["kurtosis"] = df[NUMERIC_COLS].kurtosis()
    return desc.round(4).reset_index().rename(columns={"index": "feature"}).to_dict("records")


def correlation_matrix(df: pd.DataFrame) -> dict:
    corr = df[NUMERIC_COLS].corr().round(3)
    return {
        "features": NUMERIC_COLS,
        "matrix": corr.values.tolist(),
    }


def histogram_data(df: pd.DataFrame, feature: str, bins: int = 30) -> dict:
    if feature not in NUMERIC_COLS:
        raise ValueError(f"Unknown feature: {feature}")
    counts, edges = np.histogram(df[feature].dropna(), bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).round(4).tolist()
    return {"feature": feature, "counts": counts.tolist(), "bin_centers": centers}


def boxplot_by_class(df: pd.DataFrame, feature: str) -> dict:
    """Distribution of `feature` split by Trip_Risk class -- the visual
    precursor to the ANOVA test run in cda.py on the same grouping."""
    if feature not in NUMERIC_COLS:
        raise ValueError(f"Unknown feature: {feature}")
    out = {}
    for cls in config.CLASSES:
        vals = df.loc[df[config.TARGET] == cls, feature].dropna()
        out[cls] = vals.tolist()
    return out


def class_balance(df: pd.DataFrame) -> dict:
    counts = df[config.TARGET].value_counts()
    counts = counts.reindex(config.CLASSES).fillna(0).astype(int)
    return {"labels": counts.index.tolist(), "counts": counts.values.tolist()}


def scatter_data(df: pd.DataFrame, x: str, y: str, sample: int = 1000) -> dict:
    sub = df[[x, y, config.TARGET]].dropna()
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=config.RANDOM_STATE)
    return {
        "x": sub[x].tolist(),
        "y": sub[y].tolist(),
        "class": sub[config.TARGET].tolist(),
    }


def timeseries_data(df: pd.DataFrame, feature: str) -> dict:
    sub = df[["Timestamp", feature, config.TARGET]].sort_values("Timestamp")
    return {
        "timestamps": sub["Timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "values": sub[feature].tolist(),
        "class": sub[config.TARGET].tolist(),
    }
