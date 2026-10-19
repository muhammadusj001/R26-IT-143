"""
PayHere recurring billing for Pro / Plus (USD, monthly, forever).

Flow:
  /billing/checkout/<plan>  creates a pending order (older open orders of the
                            user become "abandoned") and auto-submits a
                            signed form to PayHere
  /billing/notify           PayHere -> server (no login, no CSRF, no
                            redirect, always 200): applies a payment after
                            md5sig, merchant id, amount and currency check
  /billing/status/<order>   {status, plan}; while open, asks PayHere's
                            Retrieval API (verify_order) — so a lost
                            notification can't leave an order pending
  /billing/success          return page; polls /billing/status
  /billing/cancel           back to /pricing

A plan only changes after PayHere itself confirmed the payment (signed
notify or the authenticated Retrieval API), never from the return URL.
Every event is stored in payhere_notifications; repeats are ignored.
"""

import calendar
import json
import logging
import secrets
import threading
import time
from datetime import date
from decimal import Decimal, InvalidOperation

import requests
from flask import (Blueprint, abort, flash, jsonify, redirect, render_template, request,
                   session, url_for)

from config import settings
from core import db, payhere_api
from core.auth import _check_csrf, _notify_plan_change, admin_required, current_user, user_required
from core.payhere import checkout_hash, format_amount, notify_signature, signature_valid
from core.plans import PLANS

bp = Blueprint("billing", __name__)
log = logging.getLogger("billing")

CURRENCY = "USD"
COUNTRY = "Sri Lanka"
# Fields we never store (card details PayHere includes for VISA/MASTER)
_SENSITIVE = {"card_holder_name", "card_no", "card_expiry", "md5sig"}

STATUS_SUCCESS, STATUS_PENDING = "2", "0"
STATUS_CANCELED, STATUS_FAILED, STATUS_CHARGEDBACK = "-1", "-2", "-3"
_FAILURE_STATUS = {STATUS_CANCELED: "cancelled", STATUS_FAILED: "failed",
                   STATUS_CHARGEDBACK: "chargedback"}


def billing_configured():
    return settings.BILLING_CONFIGURED


def mask_id(value):
    """Show only the ends of an id: 420075054277 -> 4200…4277."""
    value = str(value or "")
    return value if len(value) <= 8 else f"{value[:4]}…{value[-4:]}"


@bp.app_context_processor
def _inject():
    user = current_user()
    return {"billing_configured": billing_configured(),
            "payhere_sandbox": settings.PAYHERE_SANDBOX,
            "has_subscription": bool(user) and db.has_active_subscription(user),
            "mask_id": mask_id}


# ── Checkout ─────────────────────────────────────────────────
def _new_order_id(user_id):
    return f"AG{user_id}-{secrets.token_hex(5).upper()}"


@bp.route("/billing/checkout/<plan_key>", methods=["GET", "POST"])
@user_required
def checkout(plan_key):
    if plan_key not in PLANS:
        abort(404)
    plan = PLANS[plan_key]
    if plan["price_usd"] == 0:
        return redirect(url_for("auth.pricing"))
    if not billing_configured():
        return render_template("billing/unavailable.html", plan_key=plan_key)

    user = current_user()
    old_subscription_id = None
    if db.has_active_subscription(user):
        current = PLANS[user["plan"]]
        if user["plan"] == plan_key:
            flash(f"You're already on {plan['name']}.", "error")
            return redirect(url_for("auth.account"))
        if plan["price_usd"] < current["price_usd"]:
            # Downgrades are scheduled, not charged: see /billing/downgrade
            flash(f"Use “Downgrade” to switch to {plan['name']} at the end of your billing period.", "error")
            return redirect(url_for("auth.pricing"))
        # Upgrade: the new subscription replaces this one once it is paid
        old_subscription_id = user["subscription_id"]

    if request.method == "POST":
        _check_csrf()
        details = {k: request.form.get(k, "").strip() for k in ("phone", "address", "city")}
        if not all(details.values()):
            flash("Please fill in phone, address and city.", "error")
            return render_template("billing/details.html", plan_key=plan_key, details=details)
        db.update_user(user["id"], **details)
        user = db.get_user(user["id"])
    elif not (user["phone"] and user["address"] and user["city"]):
        # PayHere requires these customer fields; ask once and remember them
        return render_template("billing/details.html", plan_key=plan_key,
                               details={k: user[k] or "" for k in ("phone", "address", "city")})

    amount = format_amount(plan["price_usd"])
    order_id = _new_order_id(user["id"])
    abandoned = db.abandon_pending_orders(user["id"])
    if abandoned:
        log.info("checkout user=%s: %s older pending order(s) marked abandoned", user["id"], abandoned)
    db.create_payment(user["id"], order_id, plan_key, amount, CURRENCY,
                      old_subscription_id=old_subscription_id)
    session["last_order_id"] = order_id

    first_name, _, last_name = user["name"].strip().partition(" ")
    # The browser comes back to the host the user is logged in on (their
    # session cookie lives there); only PayHere's server-to-server
    # notification needs the public URL.
    here = request.host_url.rstrip("/")
    return_url = f"{here}{url_for('billing.success')}"
    cancel_url = f"{here}{url_for('billing.cancel')}"
    notify_url = f"{settings.PUBLIC_BASE_URL}{url_for('billing.notify')}"
    fields = {
        "merchant_id": settings.PAYHERE_MERCHANT_ID,
        "return_url": return_url,
        "cancel_url": cancel_url,
        "notify_url": notify_url,
        "order_id": order_id,
        "items": f"AquaGuard {plan['name']}",
        "currency": CURRENCY,
        "amount": amount,
        "recurrence": "1 Month",
        "duration": "Forever",
        "first_name": first_name,
        "last_name": last_name or first_name,
        "email": user["email"],
        "phone": user["phone"],
        "address": user["address"],
        "city": user["city"],
        "country": COUNTRY,
        "custom_1": str(user["id"]),
        "custom_2": plan_key,
        "hash": checkout_hash(settings.PAYHERE_MERCHANT_ID, order_id, amount, CURRENCY,
                              settings.PAYHERE_MERCHANT_SECRET),
    }
    log.info("checkout order=%s user=%s plan=%s amount=%s %s return_url=%s cancel_url=%s "
             "notify_url=%s", order_id, user["id"], plan_key, amount, CURRENCY,
             return_url, cancel_url, notify_url)
    return render_template("billing/redirect.html", action=settings.PAYHERE_CHECKOUT_URL,
                           fields=fields, plan_key=plan_key)


OPEN_STATUSES = ("pending", "abandoned")  # no result from PayHere yet


def _order_status(order):
    """Public, read-only status of an order (no personal data)."""
    renews_at = None
    if order["status"] == "paid":
        user = db.get_user(order["user_id"])
        if user is not None and user["plan"] == order["plan"]:
            renews_at = (user["plan_renews_at"] or "")[:10] or None
    return {"status": order["status"], "plan": order["plan"],
            "plan_name": PLANS[order["plan"]]["name"], "renews_at": renews_at}


@bp.route("/billing/success", methods=["GET", "POST"])
def success():
    """PayHere's return page. Never trusted (the plan only changes after
    notify or the Retrieval API confirm the payment) and never blank: it
    works without a session — PayHere may send the browser to a host the
    user is not logged in on (e.g. ngrok)."""
    order_id = (request.values.get("order_id") or session.get("last_order_id") or "").strip()
    order = db.get_order(order_id) if order_id else None
    user = current_user()
    is_owner = order is not None and user is not None and user["id"] == order["user_id"]
    log.info("success page order=%s found=%s logged_in_owner=%s host=%s",
             order_id or "-", order is not None, is_owner, request.host)
    return render_template(
        "billing/success.html", order=order, order_id=order_id, is_owner=is_owner,
        app_base_url=settings.APP_BASE_URL,
        status=_order_status(order) if order is not None else None,
    )


@bp.route("/billing/status/<order_id>")
def order_status(order_id):
    """{status, plan, ...}; a still-open order is checked with PayHere's
    Retrieval API (throttled), so a lost notification can't block it."""
    order = db.get_order(order_id)
    if order is None:
        return jsonify({"status": "unknown", "plan": None}), 404
    if order["status"] in OPEN_STATUSES:
        verify_order(order_id)
        order = db.get_order(order_id)
    return jsonify(_order_status(order))


@bp.route("/billing/check/<order_id>", methods=["POST"])
@user_required
def check_order(order_id):
    """"Check status" button on /account for the user's own open orders."""
    _check_csrf()
    order = db.get_order(order_id)
    if order is None or order["user_id"] != current_user()["id"]:
        abort(404)
    status, detail = verify_order(order_id, force=True)
    if status == "paid":
        flash(f"Payment confirmed — you're on {PLANS[order['plan']]['name']} now.", "success")
    elif not payhere_api.configured():
        flash("PayHere hasn't confirmed this order yet. If you paid, your plan switches "
              "automatically when PayHere's notification arrives (or ask the admin).", "error")
    else:
        flash(f"PayHere has no completed payment for {order_id} yet ({detail}).", "error")
    return redirect(url_for("auth.account"))


@bp.route("/billing/cancel")
@user_required
def cancel():
    flash("Payment cancelled — you have not been charged.", "error")
    return redirect(url_for("auth.pricing"))


@bp.route("/api/me")
def api_me():
    user = current_user()
    if user is None:
        return jsonify({"error": "login_required"}), 401
    order_id = request.args.get("order_id")
    order = db.get_order(order_id) if order_id else None
    return jsonify({
        "id": user["id"], "name": user["name"], "email": user["email"],
        "plan": user["plan"], "effective_plan": db.effective_plan(user),
        "plan_status": user["plan_status"], "plan_renews_at": user["plan_renews_at"],
        "order": ({"order_id": order["order_id"], "plan": order["plan"], "status": order["status"]}
                  if order is not None and order["user_id"] == user["id"] else None),
    })


# ── Sandbox helper ───────────────────────────────────────────
@bp.route("/admin/payments/<order_id>/simulate", methods=["POST"])
@admin_required
def simulate_confirmation(order_id):
    """Sandbox only: send ourselves a correctly signed "payment successful"
    notification for an open order — exercises the real notify path."""
    _check_csrf()
    if not settings.PAYHERE_SANDBOX:
        abort(404)
    order = db.get_order(order_id)
    if order is None:
        abort(404)
    if order["status"] not in OPEN_STATUSES:
        flash(f"{order_id} is already {order['status']}.", "error")
        return redirect(url_for("auth.admin_users"))
    tag = secrets.token_hex(4).upper()
    form = {
        "merchant_id": settings.PAYHERE_MERCHANT_ID, "order_id": order_id,
        "payment_id": f"SIM{tag}", "payhere_amount": order["amount"],
        "payhere_currency": order["currency"], "status_code": STATUS_SUCCESS,
        "subscription_id": f"SIMSUB{tag}", "message_type": "AUTHORIZATION_SUCCESS",
        "item_rec_status": "0", "item_rec_date_next": _one_month_from(date.today()).isoformat(),
        "status_message": "Simulated by admin (sandbox)", "recurring": "1",
    }
    form["md5sig"] = notify_signature(settings.PAYHERE_MERCHANT_ID, order_id, order["amount"],
                                      order["currency"], STATUS_SUCCESS,
                                      settings.PAYHERE_MERCHANT_SECRET)
    status, result = process_notification(form)
    flash(f"Simulated confirmation for {order_id}: {result}", "success" if status == 200 else "error")
    return redirect(url_for("auth.admin_users"))


# ── Notify (PayHere -> server) ───────────────────────────────
@bp.route("/billing/notify", methods=["POST"])
def notify():
    """Public, CSRF-exempt, never redirects, always answers 200 quickly —
    PayHere only needs an acknowledgement; the verdict is logged."""
    try:
        _status, result = process_notification(request.form.to_dict())
    except Exception:  # noqa: BLE001 — logged in process_notification
        result = "error: see server log"
    return result, 200, {"Content-Type": "text/plain"}


def process_notification(form):
    """Verify and apply one PayHere notification. Returns (status, result):
    200 when it was applied or recognised, 400 when it was rejected."""
    payload = json.dumps({k: v for k, v in form.items() if k not in _SENSITIVE}, sort_keys=True)
    order_id = form.get("order_id", "")
    payment_id = form.get("payment_id") or None
    status_code = form.get("status_code", "")
    message_type = form.get("message_type", "")
    merchant_ok = form.get("merchant_id", "").strip() == settings.PAYHERE_MERCHANT_ID
    sig_ok = bool(settings.PAYHERE_MERCHANT_SECRET) and merchant_ok and \
        signature_valid(form, settings.PAYHERE_MERCHANT_ID, settings.PAYHERE_MERCHANT_SECRET)
    log.info("notify received order_id=%s status_code=%s sig_valid=%s%s payment_id=%s "
             "message_type=%s amount=%s %s", order_id or "-", status_code or "-",
             "yes" if sig_ok else "no", "" if merchant_ok else " (merchant_id mismatch)",
             payment_id or "-", message_type or "-", form.get("payhere_amount", "-"),
             form.get("payhere_currency", "-"))
    dedupe_key = f"{order_id}|{payment_id or ''}|{status_code}|{message_type}"

    def apply(conn):
        return _apply(conn, form, sig_ok, dedupe_key)

    status, result, processed = _in_transaction(
        apply, order_id, payment_id, status_code, message_type, sig_ok, dedupe_key, payload)
    log.info("notify done order_id=%s status_code=%s sig_valid=%s action=%s",
             order_id or "-", status_code or "-", "yes" if sig_ok else "no", result)
    if processed and status == 200:
        _after_payment(order_id)  # cancel a replaced subscription, re-gate dashboards
    return status, result


def _in_transaction(apply, order_id, payment_id, status_code, message_type, sig_ok,
                    dedupe_key, payload):
    """Run apply(conn) and log the event, in one serialised transaction."""
    conn = db.raw_connect()
    conn.isolation_level = None  # manual transaction below
    try:
        conn.execute("BEGIN IMMEDIATE")  # serialise concurrent notifications
        status, result, processed = apply(conn)
        conn.execute(
            "INSERT INTO payhere_notifications (received_at, order_id, payment_id, status_code, "
            "message_type, signature_ok, dedupe_key, result, payload) VALUES (?,?,?,?,?,?,?,?,?)",
            (db.now(), order_id, payment_id, status_code, message_type, int(sig_ok),
             dedupe_key if processed else None, result, payload),
        )
        conn.execute("COMMIT")
        return status, result, processed
    except Exception:
        conn.execute("ROLLBACK")
        log.exception("billing event for order %s failed", order_id or "-")
        raise
    finally:
        conn.close()


def _record(conn, order, status, payment_id, subscription_id, paid_on=None):
    """Upsert the order's single history row (order_id is UNIQUE): monthly
    charges and repeated notifications update it, never add rows."""
    conn.execute(
        "UPDATE payments SET status = ?, payhere_payment_id = COALESCE(?, payhere_payment_id), "
        "subscription_id = COALESCE(?, subscription_id), updated_at = ?, "
        "paid_at = COALESCE(?, paid_at) WHERE id = ?",
        (status, payment_id, subscription_id, db.now(), paid_on, order["id"]))


def _next_billing_date(item_rec_date_next, paid_on):
    """PayHere's item_rec_date_next if it is a future date, else paid_on +
    1 month (the sandbox reports the payment day itself as "next")."""
    try:
        given = date.fromisoformat(str(item_rec_date_next or "")[:10])
        if given > paid_on:
            return given.isoformat()
    except ValueError:
        pass
    return _one_month_from(paid_on).isoformat()


def _activate(conn, order, user, subscription_id, renews_at, new_subscription):
    """Give the user the order's plan. A new subscription also clears any
    scheduled cancel/downgrade (the old subscription is cancelled after
    commit, see _after_payment)."""
    conn.execute("UPDATE users SET plan = ?, plan_status = 'active', "
                 "subscription_id = COALESCE(?, subscription_id), plan_renews_at = ?"
                 + (", scheduled_plan = NULL" if new_subscription else "") + " WHERE id = ?",
                 (order["plan"], subscription_id, renews_at, user["id"]))
    return f"paid: user {user['id']} -> {order['plan']}"


def _amount_matches(amount, order):
    try:
        return Decimal(str(amount).strip()) == Decimal(order["amount"])
    except (InvalidOperation, ValueError):
        return False


def _apply(conn, form, sig_ok, dedupe_key):
    """Returns (status, result, processed)."""
    if not sig_ok:
        return 400, "rejected: bad signature or merchant id", False

    order_id = form.get("order_id", "")
    order = conn.execute("SELECT * FROM payments WHERE order_id = ? ORDER BY id LIMIT 1",
                         (order_id,)).fetchone()
    if order is None:
        return 400, "rejected: unknown order", False
    if not _amount_matches(form.get("payhere_amount", ""), order) or \
            form.get("payhere_currency", "").strip() != order["currency"]:
        return 400, "rejected: amount/currency mismatch", False

    if conn.execute("SELECT 1 FROM payhere_notifications WHERE dedupe_key = ?",
                    (dedupe_key,)).fetchone():
        return 200, "duplicate: already processed", False

    user = conn.execute("SELECT * FROM users WHERE id = ?", (order["user_id"],)).fetchone()
    status_code = form.get("status_code", "")
    message_type = form.get("message_type", "")
    payment_id = form.get("payment_id") or None
    subscription_id = form.get("subscription_id") or None

    # Is this notification about the subscription the user is on now?
    # (an old subscription must never overwrite a newer plan)
    is_first_result = order["status"] in OPEN_STATUSES
    current_sub = user["subscription_id"]
    about_current = (subscription_id is not None and subscription_id == current_sub) or \
        (not current_sub and user["plan"] == order["plan"])

    def downgrade(reason):
        conn.execute("UPDATE users SET plan = 'basic', plan_status = 'cancelled', "
                     "plan_renews_at = NULL, subscription_id = NULL WHERE id = ?", (user["id"],))
        return f"user {user['id']} downgraded to basic ({reason})"

    if status_code == STATUS_SUCCESS:
        paid_on = date.today()
        _record(conn, order, "paid", payment_id, subscription_id, paid_on.isoformat())
        if is_first_result or about_current:
            renews = _next_billing_date(form.get("item_rec_date_next"), paid_on)
            return 200, _activate(conn, order, user, subscription_id, renews,
                                  new_subscription=is_first_result), True
        return 200, "paid: installment of a non-current subscription, plan unchanged", True

    if status_code == STATUS_PENDING:
        return 200, "pending: nothing to do yet", True

    if status_code in _FAILURE_STATUS:
        result = f"{_FAILURE_STATUS[status_code]}"
        if is_first_result or status_code == STATUS_CHARGEDBACK:
            _record(conn, order, _FAILURE_STATUS[status_code], payment_id, subscription_id)
        else:
            # A failed renewal keeps the paid history row; PayHere retries and
            # the plan lapses by itself when plan_renews_at passes.
            result = f"renewal {result}"
        stopped = message_type == "RECURRING_STOPPED" or form.get("item_rec_status") == "-1"
        if status_code == STATUS_CHARGEDBACK and about_current:
            result += "; " + downgrade("chargeback")
        elif stopped and about_current:
            if user["scheduled_plan"]:
                # We cancelled it ourselves (cancel/downgrade): paid access
                # continues until plan_renews_at.
                result += "; subscription stopped as scheduled, access until " \
                          f"{user['plan_renews_at']}"
            else:
                result += "; " + downgrade("subscription cancelled")
        return 200, result, True

    return 400, f"rejected: unknown status_code {status_code!r}", False


# ── Verification via the Retrieval API (independent of notify) ─
VERIFY_INTERVAL_S = 3
_last_verify = {}
_verify_lock = threading.Lock()


def _one_month_from(day):
    month = day.month % 12 + 1
    year = day.year + (day.month == 12)
    return day.replace(year=year, month=month, day=min(day.day, calendar.monthrange(year, month)[1]))


def verify_order(order_id, force=False, http=None):
    """Ask PayHere whether order_id was paid and activate the plan if a
    RECEIVED payment with the order's amount and currency exists.
    Returns (order_status, detail). At most once per 3 s per order unless forced."""
    order = db.get_order(order_id)
    if order is None:
        return "unknown", "no such order"
    if order["status"] not in OPEN_STATUSES:
        return order["status"], "already final"
    if not payhere_api.configured():
        return order["status"], "retrieval API not configured"
    now = time.monotonic()
    with _verify_lock:
        if not force and now - _last_verify.get(order_id, -1e9) < VERIFY_INTERVAL_S:
            return order["status"], "checked recently"
        _last_verify[order_id] = now

    try:
        found = payhere_api.search_payments(order_id, **({"http": http} if http else {}))
    except (payhere_api.RetrievalError, requests.RequestException, ValueError, KeyError) as exc:
        log.warning("verify order_id=%s retrieval error: %s", order_id, exc)
        return order["status"], f"PayHere API error: {exc}"

    match = next((p for p in found
                  if str(p.get("status", "")).upper() == "RECEIVED"
                  and _amount_matches(p.get("amount", ""), order)
                  and str(p.get("currency", "")).upper() == order["currency"]), None)
    log.info("verify order_id=%s payments_found=%s statuses=%s match=%s", order_id, len(found),
             [p.get("status") for p in found], bool(match))
    if match is None:
        return order["status"], f"{len(found)} payment(s) found, none RECEIVED for " \
                                f"{order['currency']} {order['amount']}"

    payment_id = str(match.get("payment_id") or "") or None
    try:
        paid_on = date.fromisoformat(str(match.get("date") or "")[:10])
    except ValueError:
        paid_on = date.today()
    renews_at = _next_billing_date(None, paid_on)
    dedupe_key = f"{order_id}|{payment_id or ''}|RETRIEVAL"
    payload = json.dumps({"source": "retrieval_api", "payment_id": payment_id,
                          "status": match.get("status"), "amount": match.get("amount"),
                          "currency": match.get("currency"), "date": match.get("date")})

    def apply(conn):
        fresh = conn.execute("SELECT * FROM payments WHERE id = ?", (order["id"],)).fetchone()
        if fresh["status"] not in OPEN_STATUSES:  # notify won the race
            return 200, f"already {fresh['status']}", False
        user = conn.execute("SELECT * FROM users WHERE id = ?", (order["user_id"],)).fetchone()
        _record(conn, fresh, "paid", payment_id, None, paid_on.isoformat())
        return 200, _activate(conn, fresh, user, None, renews_at, new_subscription=True) \
            + " (via Retrieval API)", True

    _status, result, processed = _in_transaction(
        apply, order_id, payment_id, STATUS_SUCCESS, "RETRIEVAL_API", True, dedupe_key, payload)
    log.info("verify order_id=%s action=%s", order_id, result)
    if processed:
        _after_payment(order_id)
    return db.get_order(order_id)["status"], result


# ── Plan changes: upgrade clean-up, cancel, downgrade ────────
def _after_payment(order_id):
    """After a confirmed payment (committed): cancel the subscription this
    order replaces (Pro -> Plus upgrade) and re-gate open dashboards."""
    order = db.get_order(order_id)
    if order["status"] == "paid" and order["old_subscription_id"]:
        cancel_subscription_for(order["user_id"], order["old_subscription_id"],
                                f"replaced by {order['plan']} ({order_id})")
    _notify_plan_change(order["user_id"])


def cancel_subscription_for(user_id, subscription_id, reason):
    """Ask PayHere to stop charging subscription_id. Never silent: a failure
    is stored as 'cancel_pending' and listed on the admin page.
    Returns (ok, detail)."""
    existing = db.get_cancellation(subscription_id)
    if existing is not None and existing["status"] == "cancelled":
        return True, "already cancelled"
    if not payhere_api.configured():
        status, error = "cancel_pending", "PayHere API key (PAYHERE_APP_ID/SECRET) not configured"
    else:
        try:
            payhere_api.cancel_subscription(subscription_id)
            status, error = "cancelled", None
        except (payhere_api.RetrievalError, requests.RequestException, ValueError, KeyError) as exc:
            status, error = "cancel_pending", str(exc)
    db.record_cancellation(user_id, subscription_id, reason, status, error)
    (log.info if status == "cancelled" else log.error)(
        "cancel subscription=%s user=%s reason=%s -> %s%s", subscription_id, user_id, reason,
        status, f" ({error})" if error else "")
    return status == "cancelled", error or "cancelled"


def schedule_change(user, target_plan, reason):
    """Cancel the user's running subscription now and switch to target_plan
    ('basic' = cancel, or a cheaper plan) when the paid period ends.
    Returns (ok, detail)."""
    ok, detail = cancel_subscription_for(user["id"], user["subscription_id"], reason)
    if user["plan_renews_at"]:
        db.update_user(user["id"], scheduled_plan=target_plan)
    else:  # no known period end: the change applies right away
        db.update_user(user["id"], scheduled_plan=target_plan, plan_renews_at=date.today().isoformat())
    db.apply_due_changes(db.get_user(user["id"]))
    _notify_plan_change(user["id"])
    return ok, detail


def _pending_note(ok):
    return "" if ok else " PayHere hasn't confirmed the cancellation yet — the admin has been notified."


@bp.route("/billing/cancel-subscription", methods=["POST"])
@user_required
def cancel_subscription():
    _check_csrf()
    user = current_user()
    if not db.has_active_subscription(user):
        flash("You don't have an active paid subscription.", "error")
        return redirect(url_for("auth.account"))
    ok, _ = schedule_change(user, "basic", "cancelled by user")
    user = db.get_user(user["id"])
    if user["plan"] == "basic":
        flash("Subscription cancelled — you're on Basic now." + _pending_note(ok), "success")
    else:
        flash(f"Subscription cancelled. {PLANS[user['plan']]['name']} stays active until "
              f"{user['plan_renews_at']}, then you move to Basic." + _pending_note(ok), "success")
    return redirect(url_for("auth.account"))


@bp.route("/billing/downgrade/<plan_key>", methods=["POST"])
@user_required
def downgrade(plan_key):
    _check_csrf()
    user = current_user()
    if plan_key not in PLANS or PLANS[plan_key]["price_usd"] == 0:
        abort(404)
    if not db.has_active_subscription(user) or \
            PLANS[plan_key]["price_usd"] >= PLANS[user["plan"]]["price_usd"]:
        flash(f"{PLANS[plan_key]['name']} isn't a downgrade from your current plan.", "error")
        return redirect(url_for("auth.pricing"))
    ok, _ = schedule_change(user, plan_key, f"downgrade to {plan_key} by user")
    user = db.get_user(user["id"])
    flash(f"Your plan will change to {PLANS[plan_key]['name']} on {user['plan_renews_at']}. "
          f"You keep {PLANS[user['plan']]['name']} until then; we'll ask you to confirm the "
          f"{PLANS[plan_key]['name']} payment at that point." + _pending_note(ok), "success")
    return redirect(url_for("auth.account"))


@bp.route("/admin/users/<int:user_id>/cancel-subscription", methods=["POST"])
@admin_required
def admin_cancel_subscription(user_id):
    _check_csrf()
    user = db.get_user(user_id)
    if user is None or not db.has_active_subscription(user):
        flash("That user has no active paid subscription.", "error")
        return redirect(url_for("auth.admin_users"))
    ok, detail = schedule_change(user, "basic", "cancelled by admin")
    flash(f"{user['email']}: subscription cancelled ({'confirmed by PayHere' if ok else detail}); "
          f"paid access until {user['plan_renews_at'] or 'now'}.", "success" if ok else "error")
    return redirect(url_for("auth.admin_users"))


@bp.route("/admin/cancellations/<subscription_id>/retry", methods=["POST"])
@admin_required
def admin_retry_cancellation(subscription_id):
    _check_csrf()
    pending = db.get_cancellation(subscription_id)
    if pending is None:
        abort(404)
    ok, detail = cancel_subscription_for(pending["user_id"], subscription_id, pending["reason"])
    flash(f"Subscription {subscription_id}: {'cancelled' if ok else 'still pending — ' + detail}",
          "success" if ok else "error")
    return redirect(url_for("auth.admin_users"))
