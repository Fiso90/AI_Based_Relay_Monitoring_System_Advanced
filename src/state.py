"""
Mutable runtime state for the currently active dataset.

Unlike config.py (fixed defaults), this module holds values that can change
while the app is running -- specifically after a user uploads their own
telemetry file. Kept as a single small module so every other file reads/writes
the SAME state rather than each having its own stale copy.
"""
import os
import shutil
from . import config

ACTIVE_DATA_PATH = os.path.join(config.RUNTIME_DATA_DIR, "active_dataset.xlsx")
SAMPLE_DATA_PATH = os.path.join(config.BASE_DIR, "data", "alternator_data.xlsx")

# Runtime-tunable nameplate constants (can be overridden at upload time to
# match a different unit than the bundled 3 MW sample).
_state = {
    "rated_mw": config.RATED_MW,
    "rated_kv": config.RATED_KV,
    "classes": list(config.CLASSES),
    "source_label": "Bundled sample (1,000 observations)",
}


def ensure_active_dataset():
    """On first run, the 'active' dataset is just a copy of the bundled sample."""
    if not os.path.exists(ACTIVE_DATA_PATH):
        shutil.copyfile(SAMPLE_DATA_PATH, ACTIVE_DATA_PATH)


def get_rated_mw():
    return _state["rated_mw"]


def get_rated_kv():
    return _state["rated_kv"]


def get_classes():
    return list(_state["classes"])


def get_source_label():
    return _state["source_label"]


def set_active_dataset(rated_mw=None, rated_kv=None, classes=None, source_label=None):
    if rated_mw is not None:
        _state["rated_mw"] = float(rated_mw)
    if rated_kv is not None:
        _state["rated_kv"] = float(rated_kv)
    if classes is not None:
        _state["classes"] = list(classes)
    if source_label is not None:
        _state["source_label"] = source_label


def reset_to_sample():
    shutil.copyfile(SAMPLE_DATA_PATH, ACTIVE_DATA_PATH)
    _state["rated_mw"] = config.RATED_MW
    _state["rated_kv"] = config.RATED_KV
    _state["classes"] = list(config.CLASSES)
    _state["source_label"] = "Bundled sample (1,000 observations)"


# A fixed color/order is used for the three labels this project was designed
# around; anything else uploaded gets colors assigned from this fallback
# palette by first-seen order, so the dashboard still renders sensibly for a
# dataset that uses different risk-label names.
KNOWN_CLASS_COLORS = {
    "NORMAL": "#4ecb71",
    "WARNING": "#f2a93b",
    "HIGH TRIP RISK": "#f2555a",
}
FALLBACK_PALETTE = ["#4ecb71", "#f2a93b", "#f2555a", "#a385f0", "#3fa9f5", "#f27f3f"]


def class_color_map():
    classes = get_classes()
    out = {}
    fallback_i = 0
    for c in classes:
        if c in KNOWN_CLASS_COLORS:
            out[c] = KNOWN_CLASS_COLORS[c]
        else:
            out[c] = FALLBACK_PALETTE[fallback_i % len(FALLBACK_PALETTE)]
            fallback_i += 1
    return out
