"""
Central configuration for the 3 MW Alternator ML Lifecycle Dashboard.

Keeping every constant here means the EDA, CDA, feature engineering, and
modeling modules never redefine or accidentally drift on shared numbers
(rated capacity, thresholds, feature lists, etc).
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "alternator_data.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_SHEET = "Alternator_Data"

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Alternator nameplate ---
RATED_MW = 3.0
RATED_KV = 11.0  # nominal line voltage used for % deviation

# --- Prototype setpoints (mirrors the source workbook's Prototype_Setpoints sheet) ---
# These thresholds are what DETERMINISTICALLY generated the Trip_Risk label,
# which is why the CDA page treats them as ground truth to confirm/deny,
# not as something to be "discovered" by the classifier.
SETPOINTS = {
    "MW":            {"normal_max": 2.40, "warning_max": 2.70},
    "Current_A":     {"normal_max": 110.0, "warning_max": 120.0},
    "Power_Factor":  {"normal_min": 0.85, "warning_min": 0.80},
    "Voltage_Deviation_Percent": {"normal_max": 3.0, "warning_max": 5.0},
}

RAW_FEATURES = ["MW", "Current_A", "Power_Factor", "Voltage_kV"]

ENGINEERED_FEATURES = [
    "Calculated_MVA",
    "Reactive_Power_MVAr",
    "Load_Percent",
    "Voltage_Deviation_Percent",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES + ["Anomaly_Score"]

TARGET = "Trip_Risk"
# NOTE: the source workbook's actual label values are "NORMAL", "WARNING",
# and "HIGH TRIP RISK" (not "HIGH") -- verified against the raw data rather
# than assumed, since this is exactly the kind of stale-value mismatch this
# project has been flagging.
CLASSES = ["NORMAL", "WARNING", "HIGH TRIP RISK"]

RANDOM_STATE = 42
