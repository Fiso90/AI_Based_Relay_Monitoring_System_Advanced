"""
AI Decision Engine — the operational fusion layer.

This is the piece that turns three independent signals into ONE actionable
verdict for an operator, which is the step the underlying ML pipeline
(EDA/CDA/anomaly/classifiers) was always missing:

    1. Engineering Protection Rules  -- deterministic setpoint thresholds
       (config.SETPOINTS), the same logic a physical protection relay uses.
       Zero ML involved; this is the ground truth an operator already trusts.
    2. AI Anomaly Detection          -- Isolation Forest score: does this
       reading look like anything the system has seen before, independent
       of whether it crosses a hard threshold?
    3. AI Risk Prediction            -- the trained supervised ensemble's
       class vote + confidence.

The engine does NOT just return the ML model's answer. It returns a
severity verdict, a confidence, and an explicit breakdown of which signals
agreed or disagreed, so a decision can be audited rather than trusted
blindly. This directly implements the "AI Decision Engine" and
"Alarm / Warning / Critical Event" stages of the target operational
workflow, sitting on top of the existing rule + anomaly + ML stages rather
than replacing any of them.
"""
import math
from . import config
from . import state
from . import feature_engineering as fe
from . import anomaly
from . import models as ml_models

SEVERITY_ORDER = ["NORMAL", "WARNING", "CRITICAL"]


def _rule_verdict(mw, current_a, power_factor, voltage_dev_pct):
    """
    Engineering protection-rule verdict: the deterministic setpoint logic,
    same thresholds used everywhere else in this project (config.SETPOINTS),
    reframed as a relay-style verdict with a human-readable reason.
    """
    sp = config.SETPOINTS
    reasons = []
    score = 0

    if mw > sp["MW"]["warning_max"]:
        score += 2; reasons.append(f"MW {mw:.2f} exceeds warning threshold {sp['MW']['warning_max']}")
    elif mw > sp["MW"]["normal_max"]:
        score += 1; reasons.append(f"MW {mw:.2f} above normal threshold {sp['MW']['normal_max']}")

    if current_a > sp["Current_A"]["warning_max"]:
        score += 2; reasons.append(f"Current {current_a:.1f}A exceeds warning threshold {sp['Current_A']['warning_max']}A")
    elif current_a > sp["Current_A"]["normal_max"]:
        score += 1; reasons.append(f"Current {current_a:.1f}A above normal threshold {sp['Current_A']['normal_max']}A")

    if power_factor < sp["Power_Factor"]["warning_min"]:
        score += 2; reasons.append(f"Power factor {power_factor:.2f} below warning floor {sp['Power_Factor']['warning_min']}")
    elif power_factor < sp["Power_Factor"]["normal_min"]:
        score += 1; reasons.append(f"Power factor {power_factor:.2f} below normal floor {sp['Power_Factor']['normal_min']}")

    if abs(voltage_dev_pct) > sp["Voltage_Deviation_Percent"]["warning_max"]:
        score += 2; reasons.append(f"Voltage deviation {voltage_dev_pct:.2f}% exceeds warning threshold {sp['Voltage_Deviation_Percent']['warning_max']}%")
    elif abs(voltage_dev_pct) > sp["Voltage_Deviation_Percent"]["normal_max"]:
        score += 1; reasons.append(f"Voltage deviation {voltage_dev_pct:.2f}% above normal threshold {sp['Voltage_Deviation_Percent']['normal_max']}%")

    if score >= 2:
        severity = "CRITICAL"
    elif score == 1:
        severity = "WARNING"
    else:
        severity = "NORMAL"

    return {"severity": severity, "score": score, "reasons": reasons}


def _anomaly_verdict(anomaly_score):
    """
    Maps the continuous Isolation Forest score onto the same 3-tier
    severity scale using fixed, documented cut points, so it can be fused
    with the other two signals on equal footing.
    """
    if anomaly_score > 0.15:
        severity, note = "CRITICAL", "Reading is a strong outlier vs. historical operating envelope"
    elif anomaly_score > 0.0:
        severity, note = "WARNING", "Reading is mildly atypical vs. historical operating envelope"
    else:
        severity, note = "NORMAL", "Reading is consistent with historical operating envelope"
    return {"severity": severity, "score": round(float(anomaly_score), 4), "note": note}


def normalize_severity(label: str) -> str:
    """
    Maps any free-text class label onto the fixed 3-tier severity
    vocabulary (NORMAL/WARNING/CRITICAL) by keyword, so labels coming from
    different places -- the raw dataset ("HIGH TRIP RISK"), the decision
    engine's own output ("CRITICAL"), or an uploaded dataset's custom
    labels ("DANGER") -- can be compared on equal footing (see the
    monitoring endpoint). Falls back to returning the label unchanged if it
    matches no known keyword, so an exotic uploaded taxonomy still shows up
    as itself rather than being silently miscategorized.
    """
    upper = str(label).upper()
    if "CRITICAL" in upper or "HIGH" in upper or "DANGER" in upper:
        return "CRITICAL"
    if "WARN" in upper:
        return "WARNING"
    if "NORMAL" in upper or "SAFE" in upper:
        return "NORMAL"
    return str(label)


def _ml_verdict(prediction, probabilities):
    """Maps the ensemble's predicted class + confidence onto the 3-tier scale."""
    severity = normalize_severity(prediction)
    confidence = max(probabilities.values()) if probabilities else None
    return {"severity": severity, "predicted_label": str(prediction), "confidence": confidence}


def _combine_severity(*severities):
    """Overall verdict = the single most severe signal, never averaged down.
    A relay-monitoring system should never let a calm majority vote mask one
    alarming signal."""
    known = [s for s in severities if s in SEVERITY_ORDER]
    if not known:
        return severities[0]
    return max(known, key=lambda s: SEVERITY_ORDER.index(s))


def evaluate(mw, current_a, power_factor, voltage_kv, model_key=None):
    """
    Run the full fused decision: physics engine -> rule verdict -> anomaly
    verdict -> ML verdict -> combined severity + explainability breakdown.
    This is what both the Operator Dashboard and the event log call.

    Designed to degrade gracefully: the engineering protection-rule verdict
    never depends on trained models and always works. The anomaly and ML
    verdicts fall back to "not available" if Isolation Forest / the
    supervised bundle haven't been trained yet (e.g. a fresh checkout before
    anyone has visited the Train page), rather than raising and taking the
    whole decision down with them.
    """
    physics = fe.engineer_for_row(mw, current_a, power_factor, voltage_kv)
    row = {"MW": mw, "Current_A": current_a, "Power_Factor": power_factor,
           "Voltage_kV": voltage_kv, **physics}

    rule = _rule_verdict(mw, current_a, power_factor, physics["Voltage_Deviation_Percent"])

    anomaly_available = True
    try:
        a_score = anomaly.score_single(row)
        anom = _anomaly_verdict(a_score)
    except FileNotFoundError:
        anomaly_available = False
        anom = {"severity": "NORMAL", "score": None, "note": "Isolation Forest not trained yet — train models first for anomaly scoring."}

    ml_available = True
    try:
        ml_result = ml_models.predict_single(mw, current_a, power_factor, voltage_kv, model_key=model_key)
        ml = _ml_verdict(ml_result["prediction"], ml_result["probabilities"] or {})
    except FileNotFoundError:
        ml_available = False
        ml = {"severity": None, "predicted_label": None, "confidence": None}

    overall = _combine_severity(rule["severity"], anom["severity"], ml["severity"] or "NORMAL")

    all_severities = [rule["severity"]]
    if anomaly_available:
        all_severities.append(anom["severity"])
    if ml_available:
        all_severities.append(ml["severity"])
    agreement = len(set(all_severities)) == 1

    recommended_action = _recommend_action(overall, rule, anom, ml)

    return {
        "overall_severity": overall,
        "agreement": agreement,
        "recommended_action": recommended_action,
        "rule_based": rule,
        "anomaly": anom,
        "ml": ml,
        "ml_available": ml_available,
        "anomaly_available": anomaly_available,
        "engineered_features": physics,
        "inputs": {"mw": mw, "current_a": current_a, "power_factor": power_factor, "voltage_kv": voltage_kv},
    }


def _recommend_action(overall, rule, anom, ml):
    if overall == "CRITICAL":
        base = "Immediate operator review recommended. "
        if rule["severity"] == "CRITICAL":
            return base + "Deterministic protection threshold(s) breached: " + "; ".join(rule["reasons"]) + "."
        if anom["severity"] == "CRITICAL":
            return base + "No threshold breached, but the reading is a strong statistical outlier — verify sensors before dismissing."
        return base + "AI ensemble predicts high trip risk with high confidence, though no hard threshold is breached yet."
    if overall == "WARNING":
        return "Monitor closely. " + ("; ".join(rule["reasons"]) if rule["reasons"] else "Signals are mildly elevated but within tolerance.")
    return "Normal operation. No action required."
