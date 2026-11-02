"""
Stage 0 — Data ingestion.

Loads the raw telemetry workbook and does basic type/shape validation.
This is intentionally separate from feature_engineering.py: ingestion
answers "is the data what we think it is", engineering answers
"what do we derive from it".
"""
import pandas as pd
from . import config


def load_raw_data(path: str = None) -> pd.DataFrame:
    """Load the raw alternator telemetry sheet."""
    path = path or config.DATA_PATH
    df = pd.read_excel(path, sheet_name=config.DATA_SHEET)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def dataset_health_report(df: pd.DataFrame) -> dict:
    """
    A lightweight data-quality summary shown on the dashboard's Overview
    page — row count, date span, missing values, and class balance.
    Surfacing this up front is what let us catch the 94-vs-1000
    observation mismatch between the dataset and the original slide deck.
    """
    report = {
        "n_observations": int(len(df)),
        "date_min": df["Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Date"].max().strftime("%Y-%m-%d"),
        "n_days": int((df["Date"].max() - df["Date"].min()).days + 1),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df[config.TARGET].value_counts().to_dict(),
    }
    return report
