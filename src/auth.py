"""
Authentication & authorisation.

* Users live in SQLite (data/users.db, stdlib only).
* Passwords are salted + hashed with werkzeug (scrypt/pbkdf2) -- never stored plain.
* Sessions use Flask's signed cookie. The signing key comes from the
  SECRET_KEY env var, or is generated once and persisted to data/.secret_key
  so every gunicorn worker (and restart) shares the same key.
* CSRF: a per-session token is required on every state-changing request
  (form field `csrf_token` or header `X-CSRF-Token`).
* Roles: admin > operator > viewer.
    viewer   - read-only (dashboards, reports)
    operator - + run decisions, acknowledge alarms
    admin    - + upload data, train models, clear log, manage users
  The FIRST account registered becomes admin; later sign-ups are viewers
  until an admin promotes them (safe default for a protection system).
"""
import os
import re
import secrets
import sqlite3
import time

from werkzeug.security import generate_password_hash, check_password_hash

from . import config

DB_PATH = os.path.join(config.RUNTIME_DATA_DIR, "users.db")
SECRET_PATH = os.path.join(config.RUNTIME_DATA_DIR, ".secret_key")
ROLES = ("viewer", "operator", "admin")
ROLE_RANK = {r: i for i, r in enumerate(ROLES)}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8
MAX_FAILS, LOCK_SECONDS = 5, 300


class AuthError(Exception):
    pass


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            full_name TEXT,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            created_at REAL NOT NULL,
            last_login REAL,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until REAL NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1
        )""")
    conn.commit()
    conn.close()


def bootstrap_admin():
    """Optional: create the first admin from env vars if no users exist yet.
    Useful on hosts with an ephemeral disk (accounts vanish on restart).
    Set BOOTSTRAP_ADMIN_USER, BOOTSTRAP_ADMIN_EMAIL, BOOTSTRAP_ADMIN_PASSWORD."""
    u = os.environ.get("BOOTSTRAP_ADMIN_USER")
    e = os.environ.get("BOOTSTRAP_ADMIN_EMAIL")
    p = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD")
    if not (u and e and p) or count_users() > 0:
        return False
    try:
        create_user(u, e, p, "Administrator")
        return True
    except AuthError:
        return False  # another worker created it first, or values invalid


def get_secret_key():
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    os.makedirs(os.path.dirname(SECRET_PATH), exist_ok=True)
    if not os.path.exists(SECRET_PATH):
        try:  # O_EXCL: if two workers race, exactly one creates the file
            fd = os.open(SECRET_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_hex(32))
        except FileExistsError:
            pass
    for _ in range(20):
        with open(SECRET_PATH) as f:
            key = f.read().strip()
        if key:
            return key
        time.sleep(0.05)
    raise RuntimeError("Could not read secret key")


def _row(r):
    if r is None:
        return None
    d = dict(r)
    d.pop("password_hash", None)
    return d


def count_users():
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return n


def get_user(user_id):
    conn = _connect()
    r = conn.execute("SELECT * FROM users WHERE id=? AND active=1", (user_id,)).fetchone()
    conn.close()
    return _row(r)


def list_users():
    conn = _connect()
    rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()
    return [_row(r) for r in rows]


def create_user(username, email, password, full_name=""):
    username, email, full_name = (username or "").strip(), (email or "").strip(), (full_name or "").strip()
    if not USERNAME_RE.match(username):
        raise AuthError("Username must be 3-32 characters: letters, digits, '.', '_' or '-'.")
    if not EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError("Password must be at least %d characters." % MIN_PASSWORD)
    if password.lower() in (username.lower(), "password", "12345678", "password1"):
        raise AuthError("That password is too easy to guess.")
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")  # makes "first user = admin" race-free
        first = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        role = "admin" if first else "viewer"
        cur = conn.execute(
            "INSERT INTO users (username,email,full_name,password_hash,role,created_at) VALUES (?,?,?,?,?,?)",
            (username, email, full_name, generate_password_hash(password), role, time.time()))
        conn.commit()
        return cur.lastrowid, role
    except sqlite3.IntegrityError:
        conn.rollback()
        raise AuthError("That username or email is already registered.")
    finally:
        conn.close()


def authenticate(identifier, password):
    """Return user dict or raise AuthError. Includes brute-force lockout."""
    conn = _connect()
    try:
        r = conn.execute("SELECT * FROM users WHERE (username=? OR email=?) AND active=1",
                         ((identifier or "").strip(), (identifier or "").strip())).fetchone()
        generic = AuthError("Invalid username or password.")
        if r is None:
            check_password_hash(generate_password_hash("x"), password or "")  # constant-ish time
            raise generic
        now = time.time()
        if r["locked_until"] > now:
            raise AuthError("Account temporarily locked after repeated failures. Try again in %d min."
                            % (int((r["locked_until"] - now) // 60) + 1))
        if not check_password_hash(r["password_hash"], password or ""):
            fails = r["failed_attempts"] + 1
            lock = now + LOCK_SECONDS if fails >= MAX_FAILS else 0
            conn.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
                         (0 if lock else fails, lock, r["id"]))
            conn.commit()
            raise generic
        conn.execute("UPDATE users SET failed_attempts=0, locked_until=0, last_login=? WHERE id=?",
                     (now, r["id"]))
        conn.commit()
        return _row(r)
    finally:
        conn.close()


def set_role(user_id, role):
    if role not in ROLES:
        raise AuthError("Unknown role.")
    conn = _connect()
    cur_role = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if cur_role is None:
        conn.close()
        raise AuthError("User not found.")
    if cur_role["role"] == "admin" and role != "admin":
        admins = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]
        if admins <= 1:
            conn.close()
            raise AuthError("Cannot demote the last administrator.")
    conn.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    conn.commit()
    conn.close()


def set_active(user_id, active):
    conn = _connect()
    row = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None:
        conn.close()
        raise AuthError("User not found.")
    if not active and row["role"] == "admin":
        admins = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]
        if admins <= 1:
            conn.close()
            raise AuthError("Cannot deactivate the last administrator.")
    conn.execute("UPDATE users SET active=? WHERE id=?", (1 if active else 0, user_id))
    conn.commit()
    conn.close()


def has_role(user, minimum):
    return bool(user) and ROLE_RANK.get(user["role"], -1) >= ROLE_RANK[minimum]
