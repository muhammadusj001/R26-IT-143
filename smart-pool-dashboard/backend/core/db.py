"""
SQLite user store (backend/data/app.db).

Plain sqlite3 with one short-lived connection per call, which is safe to
use from Flask request threads, Socket.IO handlers and the CLI alike.
"""

import sqlite3
from contextlib import contextmanager
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
    created_at      TEXT NOT NULL,
    phone           TEXT,
    address         TEXT,
    city            TEXT,
    -- plan to switch to at plan_renews_at: 'basic' = cancelled at period
    -- end, a paid plan = downgrade scheduled (completed by a new checkout)
    scheduled_plan  TEXT
);

-- One row per subscription order (UNIQUE order_id): created "pending" at
-- checkout, updated by the verified PayHere notification / Retrieval API;
-- later monthly charges update the same row (latest payment id, paid_at).
CREATE TABLE IF NOT EXISTS payments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id),
    order_id            TEXT NOT NULL,
    plan                TEXT NOT NULL,
    amount              TEXT NOT NULL,
    currency            TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    payhere_payment_id  TEXT,
    subscription_id     TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT,
    paid_at             TEXT,
    -- subscription this order replaces (upgrade): cancelled once it is paid
    old_subscription_id TEXT
);
CREATE INDEX IF NOT EXISTS payments_order_id ON payments(order_id);
CREATE UNIQUE INDEX IF NOT EXISTS payments_payhere_payment_id
    ON payments(payhere_payment_id) WHERE payhere_payment_id IS NOT NULL;

-- PayHere subscriptions we asked to cancel (upgrade, downgrade, cancel).
-- status 'cancel_pending' = the API call failed: shown to the admin so a
-- subscription is never left charging silently.
CREATE TABLE IF NOT EXISTS subscription_cancellations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    subscription_id TEXT NOT NULL UNIQUE,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL,
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT
);

-- Every notification PayHere sends (valid or not). dedupe_key is set only
-- once a notification has been processed, so a repeat is detected.
CREATE TABLE IF NOT EXISTS payhere_notifications (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at  TEXT NOT NULL,
    order_id     TEXT,
    payment_id   TEXT,
    status_code  TEXT,
    message_type TEXT,
    signature_ok INTEGER NOT NULL,
    dedupe_key   TEXT UNIQUE,
    result       TEXT NOT NULL,
    payload      TEXT NOT NULL
);
"""

# Columns added after the first release (added to existing databases)
_MIGRATIONS = {
    "users": {"phone": "TEXT", "address": "TEXT", "city": "TEXT", "scheduled_plan": "TEXT"},
    "payments": {"paid_at": "TEXT", "old_subscription_id": "TEXT"},
}

USER_FIELDS = ("name", "email", "password_hash", "plan", "plan_status",
               "subscription_id", "plan_renews_at", "is_admin", "phone", "address", "city",
               "scheduled_plan")


def raw_connect():
    """A connection the caller must close (used for manual transactions)."""
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect():
    """`with connect() as conn:` — commits on success, rolls back on error
    and always closes (sqlite3's own context manager never closes)."""
    conn = raw_connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        for table, columns in _MIGRATIONS.items():
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for column, sql_type in columns.items():
                if existing and column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        conn.executescript(SCHEMA)
        _merge_duplicate_orders(conn)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS payments_order_unique ON payments(order_id)")


def _merge_duplicate_orders(conn):
    """Older versions added a row per monthly charge (and PayHere sends an
    AUTHORIZATION_SUCCESS plus a first RECURRING_INSTALLMENT_SUCCESS for a
    new subscription). Keep one row per order: the first, carrying the
    latest payment id."""
    for dup in conn.execute("SELECT order_id FROM payments GROUP BY order_id HAVING COUNT(*) > 1").fetchall():
        rows = conn.execute("SELECT * FROM payments WHERE order_id = ? ORDER BY id", (dup["order_id"],)).fetchall()
        keep, latest = rows[0], rows[-1]
        conn.execute("DELETE FROM payments WHERE order_id = ? AND id != ?", (dup["order_id"], keep["id"]))
        conn.execute("UPDATE payments SET payhere_payment_id = ?, status = ?, updated_at = ?, "
                     "paid_at = COALESCE(paid_at, ?) WHERE id = ?",
                     (latest["payhere_payment_id"], latest["status"], latest["updated_at"],
                      (latest["updated_at"] or latest["created_at"])[:10], keep["id"]))


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def create_user(name, email, password_hash, plan="basic", is_admin=False):
    """Insert a user and return its id. Raises sqlite3.IntegrityError if
    the email is already registered."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, plan, is_admin, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name.strip(), email.strip().lower(), password_hash, plan, int(is_admin), now()),
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


def create_payment(user_id, order_id, plan, amount, currency, old_subscription_id=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO payments (user_id, order_id, plan, amount, currency, status, created_at, "
            "old_subscription_id) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
            (user_id, order_id, plan, amount, currency, now(), old_subscription_id),
        )


def record_cancellation(user_id, subscription_id, reason, status, error=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO subscription_cancellations (user_id, subscription_id, reason, status, error, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(subscription_id) DO UPDATE SET status = excluded.status, "
            "error = excluded.error, updated_at = excluded.updated_at",
            (user_id, subscription_id, reason, status, error, now(), now()),
        )


def get_cancellation(subscription_id):
    with connect() as conn:
        return conn.execute("SELECT * FROM subscription_cancellations WHERE subscription_id = ?",
                            (subscription_id,)).fetchone()


def list_cancellations(status=None):
    with connect() as conn:
        sql = ("SELECT c.*, u.email FROM subscription_cancellations c JOIN users u ON u.id = c.user_id"
               + (" WHERE c.status = ?" if status else "") + " ORDER BY c.id DESC")
        return conn.execute(sql, (status,) if status else ()).fetchall()


def apply_due_changes(user):
    """Run a scheduled cancel/downgrade once its date has passed. Called on
    every request for the logged-in user; returns the (possibly updated) row."""
    if user is None or not user["scheduled_plan"] or not user["plan_renews_at"]:
        return user
    try:
        due = datetime.fromisoformat(user["plan_renews_at"][:10]).date() <= datetime.now().date()
    except ValueError:
        due = True
    if not due:
        return user
    # Paid period over: drop to Basic. For a downgrade, scheduled_plan stays
    # set so the account page prompts the checkout for the lower plan.
    keep = None if user["scheduled_plan"] == "basic" else user["scheduled_plan"]
    update_user(user["id"], plan="basic", plan_status="active", plan_renews_at=None,
                subscription_id=None, scheduled_plan=keep)
    return get_user(user["id"])


def abandon_pending_orders(user_id):
    """A new checkout supersedes the user's older unpaid orders. (A late
    payment for an abandoned order is still honoured by notify/verify.)"""
    with connect() as conn:
        return conn.execute(
            "UPDATE payments SET status = 'abandoned', updated_at = ? "
            "WHERE user_id = ? AND status = 'pending'", (now(), user_id),
        ).rowcount


def list_open_orders(limit=50):
    """Checkout rows still waiting for PayHere (admin page)."""
    with connect() as conn:
        return conn.execute(
            "SELECT p.*, u.email FROM payments p JOIN users u ON u.id = p.user_id "
            "WHERE p.status IN ('pending', 'abandoned') AND p.payhere_payment_id IS NULL "
            "ORDER BY p.id DESC LIMIT ?", (limit,),
        ).fetchall()


def get_order(order_id):
    """The checkout row for an order (the first payment with that order_id)."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM payments WHERE order_id = ? ORDER BY id LIMIT 1", (order_id,)
        ).fetchone()


def list_payments(user_id):
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM payments WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()


def has_active_subscription(user):
    """A paid PayHere subscription that is still running."""
    return bool(user["subscription_id"]) and user["plan"] != "basic" \
        and user["plan_status"] == "active"


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
