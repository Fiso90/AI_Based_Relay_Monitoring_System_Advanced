"""
Event / alarm log — persisted history of every decision-engine evaluation.

This is what makes "Historical Event Analysis" and "Operator Dashboard"
real rather than a live-only demo: every time the decision engine is run
(manually from the Operator Dashboard, or via the simulated telemetry
stepper), the full verdict is written to a local SQLite database so it
survives page reloads and app restarts. Uses only the standard library
(sqlite3) -- no new dependency for a feature this small.
"""
import json
import os
import sqlite3
import time
from . import config

DB_PATH = os.path.join(config.RUNTIME_DATA_DIR, "events.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            source TEXT NOT NULL,
            severity TEXT NOT NULL,
            agreement INTEGER NOT NULL,
            predicted_label TEXT,
            confidence REAL,
            anomaly_score REAL,
            recommended_action TEXT,
            inputs_json TEXT NOT NULL,
            verdict_json TEXT NOT NULL,
            acknowledged INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_event(verdict: dict, source: str = "manual") -> int:
    """Persist one decision-engine verdict as an event row. Returns the new row id."""
    init_db()
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO events
           (ts, source, severity, agreement, predicted_label, confidence, anomaly_score,
            recommended_action, inputs_json, verdict_json, acknowledged)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            time.time(), source, verdict["overall_severity"], int(verdict["agreement"]),
            verdict["ml"].get("predicted_label"), verdict["ml"].get("confidence"),
            verdict["anomaly"]["score"], verdict["recommended_action"],
            json.dumps(verdict["inputs"]), json.dumps(verdict),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def list_events(limit: int = 100, severity: str = None) -> list:
    init_db()
    conn = _connect()
    if severity:
        rows = conn.execute(
            "SELECT * FROM events WHERE severity = ? ORDER BY ts DESC LIMIT ?", (severity, limit)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def acknowledge_event(event_id: int) -> bool:
    init_db()
    conn = _connect()
    cur = conn.execute("UPDATE events SET acknowledged = 1 WHERE id = ?", (event_id,))
    conn.commit()
    changed = cur.rowcount > 0
    conn.close()
    return changed


def severity_counts(since_seconds: float = None) -> dict:
    """Counts by severity, optionally restricted to the last N seconds -- the
    input to the monitoring page's live severity-mix view."""
    init_db()
    conn = _connect()
    if since_seconds is not None:
        cutoff = time.time() - since_seconds
        rows = conn.execute(
            "SELECT severity, COUNT(*) as n FROM events WHERE ts >= ? GROUP BY severity", (cutoff,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT severity, COUNT(*) as n FROM events GROUP BY severity").fetchall()
    conn.close()
    return {r["severity"]: r["n"] for r in rows}


def clear_events():
    init_db()
    conn = _connect()
    conn.execute("DELETE FROM events")
    conn.commit()
    conn.close()


def _row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "ts": row["ts"],
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"])),
        "source": row["source"],
        "severity": row["severity"],
        "agreement": bool(row["agreement"]),
        "predicted_label": row["predicted_label"],
        "confidence": row["confidence"],
        "anomaly_score": row["anomaly_score"],
        "recommended_action": row["recommended_action"],
        "inputs": json.loads(row["inputs_json"]),
        "acknowledged": bool(row["acknowledged"]),
    }
