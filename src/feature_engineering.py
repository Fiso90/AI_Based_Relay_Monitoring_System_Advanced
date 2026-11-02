"""
Stage 2 — Physics engine.

Derives MVA, reactive power, load %, and voltage deviation from the raw
electrical channels using standard 3-phase AC power relationships.

NOTE ON COLLINEARITY (surfaced deliberately, not swept under the rug):
Calculated_MVA and Load_Percent are exact deterministic functions of
MW / Current_A / Voltage_kV, which are already model inputs. That means
the "8 features" the classifier sees are not 8 independent dimensions —
several are mathematically entangled. This matters when interpreting
Gini/feature importances later (see CDA page) — high importance on
Current_A does not, by itself, prove Current is the unique causal driver.
"""
import numpy as np
import pandas as pd
from . import config


def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute MVA / reactive power / load% / voltage deviation from the
    raw channels. If the workbook already has these columns we recompute
    them anyway so the dashboard is self-consistent even if fed a
    different raw dataset with only MW/Current/PF/Voltage.
    """
    out = df.copy()

    # Apparent power (MVA) for a 3-phase alternator: S = sqrt(3) * V * I / 1000
    # V in kV already, I in A -> MVA directly
    out["Calculated_MVA_engine"] = (np.sqrt(3) * out["Voltage_kV"] * out["Current_A"]) / 1000.0

    # Reactive power: Q = S * sin(acos(PF))
    pf_clipped = out["Power_Factor"].clip(-1, 1)
    out["Reactive_Power_MVAr_engine"] = out["Calculated_MVA_engine"] * np.sin(np.arccos(pf_clipped))

    # Load percent relative to rated capacity
    out["Load_Percent_engine"] = (out["MW"] / config.RATED_MW) * 100.0

    # Voltage deviation from nominal
    out["Voltage_Deviation_Percent_engine"] = (
        (out["Voltage_kV"] - config.RATED_KV) / config.RATED_KV
    ) * 100.0

    return out


def compare_engineered_vs_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity-check table: recomputed physics features vs. the ones already
    present in the source workbook. Large deltas would mean the source
    data used different nameplate constants than config.py assumes.
    """
    eng = compute_physics_features(df)
    comparison = pd.DataFrame({
        "MVA_source": df["Calculated_MVA"],
        "MVA_recomputed": eng["Calculated_MVA_engine"],
        "MVA_abs_diff": (df["Calculated_MVA"] - eng["Calculated_MVA_engine"]).abs(),
        "Load_source": df["Load_Percent"],
        "Load_recomputed": eng["Load_Percent_engine"],
        "Load_abs_diff": (df["Load_Percent"] - eng["Load_Percent_engine"]).abs(),
    })
    return comparison


def engineer_for_row(mw: float, current_a: float, power_factor: float, voltage_kv: float) -> dict:
    """Single-row version used by the live /predict endpoint."""
    mva = (np.sqrt(3) * voltage_kv * current_a) / 1000.0
    pf_clipped = max(min(power_factor, 1.0), -1.0)
    q = mva * np.sin(np.arccos(pf_clipped))
    load_pct = (mw / config.RATED_MW) * 100.0
    v_dev = ((voltage_kv - config.RATED_KV) / config.RATED_KV) * 100.0
    return {
        "Calculated_MVA": round(float(mva), 3),
        "Reactive_Power_MVAr": round(float(q), 3),
        "Load_Percent": round(float(load_pct), 2),
        "Voltage_Deviation_Percent": round(float(v_dev), 2),
    }
