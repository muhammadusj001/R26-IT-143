"""
SQLite user store (backend/data/app.db).

Plain sqlite3 with one short-lived connection per call, which is safe to
use from Flask request threads, Socket.IO handlers and the CLI alike.
"""

import sqlite3
from datetime import datetime, timezone

from config import settings
from core.plans import PLANS

PLAN_STATUSES = ("active", "pending", "cancelled")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    plan            TEXT NOT NULL DEFAULT 'basic',
    plan_status     TEXT NOT NULL DEFAULT 'active'
                    CHECK (plan_status IN ({", ".join(f"'{s}'" for s in PLAN_STATUSES)})),
    subscription_id TEXT,
    plan_renews_at  TEXT,
    is_admin        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
"""

USER_FIELDS = ("name", "email", "password_hash", "plan", "plan_status",
               "subscription_id", "plan_renews_at", "is_admin")


def connect():
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def create_user(name, email, password_hash, plan="basic", is_admin=False):
    """Insert a user and return its id. Raises sqlite3.IntegrityError if
    the email is already registered."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, plan, is_admin, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name.strip(), email.strip().lower(), password_hash, plan, int(is_admin), _now()),
        )
        return cur.lastrowid


def get_user(user_id):
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_email(email):
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()


def list_users():
    with connect() as conn:
        return conn.execute("SELECT * FROM users ORDER BY created_at, id").fetchall()


def update_user(user_id, **fields):
    unknown = set(fields) - set(USER_FIELDS)
    if unknown:
        raise ValueError(f"unknown user fields: {sorted(unknown)}")
    if "plan" in fields and fields["plan"] not in PLANS:
        raise ValueError(f"unknown plan {fields['plan']!r}")
    if "plan_status" in fields and fields["plan_status"] not in PLAN_STATUSES:
        raise ValueError(f"unknown plan_status {fields['plan_status']!r}")
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE users SET {assignments} WHERE id = ?", (*fields.values(), user_id))


def effective_plan(user):
    """The plan a user's features are gated on right now.

    Only an "active" subscription counts, and a paid plan whose
    plan_renews_at date has passed drops back to basic."""
    if user is None or user["plan_status"] != "active" or user["plan"] not in PLANS:
        return "basic"
    renews = user["plan_renews_at"]
    if renews and user["plan"] != "basic":
        try:
            if datetime.fromisoformat(renews).date() < datetime.now().date():
                return "basic"
        except ValueError:
            return "basic"
    return user["plan"]
