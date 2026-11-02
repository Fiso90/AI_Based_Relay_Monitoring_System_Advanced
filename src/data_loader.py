"""
Stage 0 — Data ingestion.

Loads the active telemetry dataset (bundled sample, or whatever the user
most recently uploaded) and does schema/type validation. Ingestion answers
"is the data what we think it is"; feature_engineering.py answers "what do
we derive from it".
"""
import os
import pandas as pd
import numpy as np
from . import config
from . import state

# Columns a valid upload MUST contain. Timestamp/Date are handled specially
# (see validate_and_prepare) since either can be derived from the other, or
# synthesized if neither is present.
REQUIRED_NUMERIC_COLUMNS = ["MW", "Current_A", "Power_Factor", "Voltage_kV"]
REQUIRED_TARGET_COLUMN = "Trip_Risk"


class DatasetValidationError(ValueError):
    """Raised when an uploaded file doesn't match the expected schema."""
    pass


def load_raw_data(path: str = None) -> pd.DataFrame:
    """Load the active telemetry sheet (or a specific path if given)."""
    path = path or state.ACTIVE_DATA_PATH
    if not os.path.exists(path):
        state.ensure_active_dataset()
    df = pd.read_excel(path, sheet_name=config.DATA_SHEET)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def dataset_health_report(df: pd.DataFrame) -> dict:
    """Lightweight data-quality summary shown on the Overview page."""
    report = {
        "n_observations": int(len(df)),
        "date_min": df["Timestamp"].min().strftime("%Y-%m-%d"),
        "date_max": df["Timestamp"].max().strftime("%Y-%m-%d"),
        "n_days": int((df["Timestamp"].max() - df["Timestamp"].min()).days + 1),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df[config.TARGET].value_counts().to_dict(),
        "source_label": state.get_source_label(),
    }
    return report


def _read_any(file_storage) -> pd.DataFrame:
    """Read an uploaded Werkzeug FileStorage as either .xlsx or .csv."""
    filename = (file_storage.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file_storage)
    elif filename.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(file_storage, sheet_name=config.DATA_SHEET)
        except ValueError:
            file_storage.seek(0)
            return pd.read_excel(file_storage, sheet_name=0)
    else:
        raise DatasetValidationError(
            f"Unsupported file type '{filename}'. Please upload a .xlsx or .csv file."
        )


def validate_and_prepare(file_storage, rated_mw: float = None, rated_kv: float = None) -> tuple:
    """
    Validate an uploaded file against the schema this pipeline needs, fill
    in gaps where reasonable (missing Timestamp/Date), and return
    (cleaned_dataframe, warnings_list, ordered_classes). Raises
    DatasetValidationError with a human-readable message if unusable.
    """
    warnings = []
    try:
        df = _read_any(file_storage)
    except Exception as e:
        raise DatasetValidationError(f"Could not read the file: {e}")

    if len(df) == 0:
        raise DatasetValidationError("The uploaded file has no rows.")

    df.columns = [str(c).strip() for c in df.columns]

    missing_numeric = [c for c in REQUIRED_NUMERIC_COLUMNS if c not in df.columns]
    if missing_numeric:
        raise DatasetValidationError(
            "Missing required column(s): " + ", ".join(missing_numeric) +
            ". Expected columns: " + ", ".join(REQUIRED_NUMERIC_COLUMNS + [REQUIRED_TARGET_COLUMN])
        )
    if REQUIRED_TARGET_COLUMN not in df.columns:
        raise DatasetValidationError(
            f"Missing required target column '{REQUIRED_TARGET_COLUMN}' "
            "(the risk/label column this pipeline predicts)."
        )

    for c in REQUIRED_NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before = len(df)
    bad_rows = df[REQUIRED_NUMERIC_COLUMNS].isna().any(axis=1) | df[REQUIRED_TARGET_COLUMN].isna()
    if bad_rows.any():
        df = df[~bad_rows].copy()
        warnings.append(f"Dropped {int(bad_rows.sum())} of {n_before} rows with missing/non-numeric values.")

    if len(df) < 20:
        raise DatasetValidationError(
            f"Only {len(df)} usable rows after cleaning -- need at least 20 to train meaningfully."
        )

    if "Timestamp" not in df.columns and "Date" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"], errors="coerce")
        warnings.append("No 'Timestamp' column found -- derived it from 'Date'.")
    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        df["Timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="h")
        warnings.append("No 'Timestamp' or 'Date' column found -- generated a synthetic sequential timestamp.")

    if df["Timestamp"].isna().any():
        n_bad_ts = int(df["Timestamp"].isna().sum())
        df = df[df["Timestamp"].notna()].copy()
        warnings.append(f"Dropped {n_bad_ts} row(s) with an unparseable timestamp.")

    if "Date" not in df.columns:
        df["Date"] = df["Timestamp"].dt.date.astype(str)

    if "Observation_ID" not in df.columns:
        df = df.reset_index(drop=True)
        df["Observation_ID"] = df.index + 1

    df[REQUIRED_TARGET_COLUMN] = df[REQUIRED_TARGET_COLUMN].astype(str).str.strip()
    n_classes = df[REQUIRED_TARGET_COLUMN].nunique()
    if n_classes < 2:
        raise DatasetValidationError(
            f"The target column '{REQUIRED_TARGET_COLUMN}' only has {n_classes} distinct value(s) -- "
            "need at least 2 classes to train a classifier."
        )
    if n_classes > 8:
        warnings.append(f"Target column has {n_classes} distinct classes -- charts may get crowded.")

    df = df.sort_values("Timestamp").reset_index(drop=True)

    uniques = list(df[REQUIRED_TARGET_COLUMN].unique())
    canonical = [c for c in config.CLASSES if c in uniques]
    extra = [c for c in uniques if c not in canonical]
    ordered_classes = canonical + extra

    return df, warnings, ordered_classes


def save_active_dataset(df: pd.DataFrame):
    """Persist the validated dataframe as the new active dataset on disk."""
    with pd.ExcelWriter(state.ACTIVE_DATA_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=config.DATA_SHEET, index=False)
