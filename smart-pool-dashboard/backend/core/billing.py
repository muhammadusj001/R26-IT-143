"""
PayHere recurring billing for Pro / Plus (USD, monthly, forever).

Flow:
  /billing/checkout/<plan>  creates a pending order and auto-submits a
                            signed form to PayHere
  /billing/notify           PayHere -> server (no login, no CSRF): the ONLY
                            place a plan is changed, after the md5sig,
                            merchant id, amount and currency are verified
  /billing/success          return page; polls /api/me until notify landed
  /billing/cancel           back to /pricing

Every notification is stored in payhere_notifications; a repeat of an
already processed notification is detected and ignored (idempotent).
"""

import json
import logging
import secrets
from decimal import Decimal, InvalidOperation

from flask import (Blueprint, abort, flash, jsonify, redirect, render_template, request,
                   session, url_for)

from config import settings
from core import db
from core.auth import _check_csrf, _notify_plan_change, current_user, user_required
from core.payhere import checkout_hash, format_amount, signature_valid
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


@bp.app_context_processor
def _inject():
    return {"billing_configured": billing_configured()}


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
    if db.has_active_subscription(user):
        # Switching a running subscription would leave the old one billing
        # at PayHere (no Subscription Manager API integration yet).
        flash("You already have an active subscription — ask the admin to change or "
              "cancel it.", "error")
        return redirect(url_for("auth.account"))

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
    db.create_payment(user["id"], order_id, plan_key, amount, CURRENCY)
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
    """PayHere's return page. Never trusted (the plan only changes in
    notify) and never blank: it works without a session — PayHere may send
    the browser to a host the user is not logged in on (e.g. ngrok)."""
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
    order = db.get_order(order_id)
    if order is None:
        return jsonify({"status": "unknown", "plan": None}), 404
    return jsonify(_order_status(order))


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


# ── Notify (PayHere -> server) ───────────────────────────────
@bp.route("/billing/notify", methods=["POST"])
def notify():
    status, result = process_notification(request.form.to_dict())
    return (result, status, {"Content-Type": "text/plain"})


def process_notification(form):
    """Verify and apply one PayHere notification. Returns (http_status, result).
    Plans change here and only here."""
    payload = json.dumps({k: v for k, v in form.items() if k not in _SENSITIVE}, sort_keys=True)
    order_id = form.get("order_id", "")
    payment_id = form.get("payment_id") or None
    status_code = form.get("status_code", "")
    message_type = form.get("message_type", "")
    merchant_ok = form.get("merchant_id") == settings.PAYHERE_MERCHANT_ID
    sig_ok = bool(settings.PAYHERE_MERCHANT_SECRET) and merchant_ok and \
        signature_valid(form, settings.PAYHERE_MERCHANT_ID, settings.PAYHERE_MERCHANT_SECRET)
    log.info("notify received order_id=%s status_code=%s signature_valid=%s%s payment_id=%s "
             "message_type=%s amount=%s %s", order_id or "-", status_code or "-",
             "yes" if sig_ok else "no", "" if merchant_ok else " (merchant_id mismatch)",
             payment_id or "-", message_type or "-", form.get("payhere_amount", "-"),
             form.get("payhere_currency", "-"))
    dedupe_key = f"{order_id}|{payment_id or ''}|{status_code}|{message_type}"

    conn = db.raw_connect()
    conn.isolation_level = None  # manual transaction below
    try:
        conn.execute("BEGIN IMMEDIATE")  # serialise concurrent notifications
        status, result, processed = _apply(conn, form, sig_ok, dedupe_key)
        conn.execute(
            "INSERT INTO payhere_notifications (received_at, order_id, payment_id, status_code, "
            "message_type, signature_ok, dedupe_key, result, payload) VALUES (?,?,?,?,?,?,?,?,?)",
            (db.now(), order_id, payment_id, status_code, message_type, int(sig_ok),
             dedupe_key if processed else None, result, payload),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        log.exception("notify order=%s failed", order_id)
        raise
    finally:
        conn.close()

    log.info("notify done order_id=%s status_code=%s signature_valid=%s -> HTTP %s: %s",
             order_id or "-", status_code or "-", "yes" if sig_ok else "no", status, result)
    if processed and status == 200:
        user_id = db.get_order(order_id)["user_id"]
        _notify_plan_change(user_id)  # re-gate that user's open dashboards
    return status, result


def _apply(conn, form, sig_ok, dedupe_key):
    """Returns (http_status, result, processed)."""
    if not sig_ok:
        return 400, "rejected: bad signature or merchant id", False

    order_id = form.get("order_id", "")
    order = conn.execute("SELECT * FROM payments WHERE order_id = ? ORDER BY id LIMIT 1",
                         (order_id,)).fetchone()
    if order is None:
        return 400, "rejected: unknown order", False
    try:
        amount_ok = Decimal(form.get("payhere_amount", "")) == Decimal(order["amount"])
    except InvalidOperation:
        amount_ok = False
    if not amount_ok or form.get("payhere_currency") != order["currency"]:
        return 400, "rejected: amount/currency mismatch", False

    if conn.execute("SELECT 1 FROM payhere_notifications WHERE dedupe_key = ?",
                    (dedupe_key,)).fetchone():
        return 200, "duplicate: already processed", False

    user = conn.execute("SELECT * FROM users WHERE id = ?", (order["user_id"],)).fetchone()
    status_code = form.get("status_code", "")
    message_type = form.get("message_type", "")
    payment_id = form.get("payment_id") or None
    subscription_id = form.get("subscription_id") or None
    now = db.now()

    # Is this notification about the subscription the user is on now?
    # (an old subscription must never overwrite a newer plan)
    is_first_result = order["status"] == "pending"
    current_sub = user["subscription_id"]
    about_current = (subscription_id is not None and subscription_id == current_sub) or \
        (not current_sub and user["plan"] == order["plan"])

    def record(status):
        """Update the checkout row on its first result, else add a charge row."""
        existing = payment_id and conn.execute(
            "SELECT id FROM payments WHERE payhere_payment_id = ? AND order_id = ?",
            (payment_id, order_id)).fetchone()
        if existing:
            conn.execute("UPDATE payments SET status = ?, subscription_id = COALESCE(?, subscription_id), "
                         "updated_at = ? WHERE id = ?", (status, subscription_id, now, existing["id"]))
        elif is_first_result:
            conn.execute("UPDATE payments SET status = ?, payhere_payment_id = ?, subscription_id = ?, "
                         "updated_at = ? WHERE id = ?",
                         (status, payment_id, subscription_id, now, order["id"]))
        else:
            conn.execute("INSERT INTO payments (user_id, order_id, plan, amount, currency, status, "
                         "payhere_payment_id, subscription_id, created_at, updated_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (order["user_id"], order_id, order["plan"], order["amount"],
                          order["currency"], status, payment_id, subscription_id, now, now))

    def downgrade(reason):
        conn.execute("UPDATE users SET plan = 'basic', plan_status = 'cancelled', "
                     "plan_renews_at = NULL, subscription_id = NULL WHERE id = ?", (user["id"],))
        return f"user {user['id']} downgraded to basic ({reason})"

    if status_code == STATUS_SUCCESS:
        record("paid")
        if is_first_result or about_current:
            if is_first_result and current_sub and current_sub != subscription_id:
                log.warning("user %s started subscription %s while %s is still on file — "
                            "cancel the old one in the PayHere portal",
                            user["id"], subscription_id, current_sub)
            conn.execute("UPDATE users SET plan = ?, plan_status = 'active', subscription_id = ?, "
                         "plan_renews_at = ? WHERE id = ?",
                         (order["plan"], subscription_id,
                          form.get("item_rec_date_next") or None, user["id"]))
            return 200, f"paid: user {user['id']} -> {order['plan']}", True
        return 200, "paid: installment of a non-current subscription, plan unchanged", True

    if status_code == STATUS_PENDING:
        return 200, "pending: nothing to do yet", True

    if status_code in _FAILURE_STATUS:
        record(_FAILURE_STATUS[status_code])
        result = f"{_FAILURE_STATUS[status_code]}"
        stopped = message_type == "RECURRING_STOPPED" or form.get("item_rec_status") == "-1"
        if status_code == STATUS_CHARGEDBACK and about_current:
            result += "; " + downgrade("chargeback")
        elif stopped and about_current:
            result += "; " + downgrade("subscription cancelled")
        # A failed renewal does not downgrade at once: PayHere retries, and the
        # plan lapses by itself when plan_renews_at passes (effective_plan).
        return 200, result, True

    return 400, f"rejected: unknown status_code {status_code!r}", False
