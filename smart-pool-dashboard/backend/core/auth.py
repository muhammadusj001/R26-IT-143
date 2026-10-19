"""
User accounts: register / login / logout, the login_required and
admin_required decorators, a small CSRF guard for the HTML forms, and
the admin page for changing a user's plan by hand (viva demo, offline
payments).
"""

import secrets
import sqlite3
from datetime import date
from functools import wraps

from flask import (Blueprint, abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from config import settings
from core import db
from core.plans import (FEATURE_SHORT, FEATURES, PLANS, has_feature, plan_change_kind, price_label,
                        required_plan, resolve_plan_key)

bp = Blueprint("auth", __name__)

MIN_PASSWORD_LENGTH = 8
# Called as fn(user_id) after an admin changes a user's plan, so open
# dashboards can be re-gated without waiting for a reconnect.
plan_change_listeners = []


# ── Current user / plan ──────────────────────────────────────
def current_user():
    if "user" not in g:
        user_id = session.get("user_id")
        g.user = db.get_user(user_id) if user_id is not None else None
        if user_id is not None and g.user is None:
            session.clear()  # account was deleted
    return g.user


def plan_for(user):
    """Plan to gate on: the user's plan, or ACTIVE_PLAN in DEMO_MODE when
    nobody is logged in."""
    if user is not None:
        return db.effective_plan(user)
    if settings.DEMO_MODE:
        return resolve_plan_key(settings.ACTIVE_PLAN)
    return None


def current_plan():
    return plan_for(current_user())


# ── Decorators ───────────────────────────────────────────────
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None and not settings.DEMO_MODE:
            if request.path.startswith("/api/"):
                return jsonify({"error": "login_required"}), 401
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    """Admin pages always need a real admin login, even in DEMO_MODE."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        if not user["is_admin"]:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


# ── CSRF for the HTML forms ──────────────────────────────────
def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


@bp.app_context_processor
def _inject():
    return {
        "csrf_token": csrf_token,
        "current_user": current_user(),
        "current_plan": current_plan(),  # None when logged out (outside DEMO_MODE)
        "plans": PLANS,
        "features": FEATURES,
        "feature_short": FEATURE_SHORT,
        "has_feature": has_feature,
        "required_plan": required_plan,
        "price_label": price_label,
        "plan_change_kind": plan_change_kind,
    }


def _check_csrf():
    sent = request.form.get("csrf", "")
    if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
        abort(400, "Invalid or missing form token — reload the page and try again.")


def _safe_next(target):
    """Only allow redirects to local paths (no open redirect)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return url_for("index")


# ── Pages ────────────────────────────────────────────────────
@bp.route("/register", methods=["GET", "POST"])
def register():
    form = {"name": "", "email": ""}
    if request.method == "POST":
        _check_csrf()
        form["name"] = request.form.get("name", "").strip()
        form["email"] = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        error = None
        if not form["name"]:
            error = "Please enter your name."
        elif "@" not in form["email"] or "." not in form["email"].split("@")[-1]:
            error = "Please enter a valid email address."
        elif len(password) < MIN_PASSWORD_LENGTH:
            error = f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            try:
                db.create_user(form["name"], form["email"], generate_password_hash(password))
            except sqlite3.IntegrityError:
                error = "An account with that email already exists."

        if error is None:
            flash("Account created — please sign in.", "success")
            return redirect(url_for("auth.login", email=form["email"]))
        flash(error, "error")
    return render_template("auth/register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    email = request.args.get("email", "")
    if request.method == "POST":
        _check_csrf()
        email = request.form.get("email", "").strip().lower()
        user = db.get_user_by_email(email)
        if user is not None and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()  # new session id contents on login
            session["user_id"] = user["id"]
            return redirect(_safe_next(request.form.get("next")))
        flash("Incorrect email or password.", "error")
    return render_template("auth/login.html", email=email, next=request.args.get("next", ""))


@bp.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("auth.login"))


def _notify_plan_change(user_id):
    for listener in plan_change_listeners:
        listener(user_id)


def user_required(view):
    """Pages about the user's own account need a real login, even in DEMO_MODE."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)
    return wrapped


@bp.route("/pricing")
def pricing():
    """Public: logged-out visitors see the plans with sign-up buttons."""
    return render_template("auth/pricing.html")


@bp.route("/account")
@user_required
def account():
    user = current_user()
    return render_template("auth/account.html", user=user, payments=db.list_payments(user["id"]))


@bp.route("/account/plan/basic", methods=["POST"])
@user_required
def select_basic():
    """Basic is free, so switching to it is instant (no checkout)."""
    _check_csrf()
    user = current_user()
    if db.has_active_subscription(user):
        # Downgrading here would not stop PayHere from charging the card
        flash("You have an active paid subscription — ask the admin to cancel it "
              "(it is then switched to Basic).", "error")
        return redirect(url_for("auth.account"))
    db.update_user(user["id"], plan="basic", plan_status="active", plan_renews_at=None)
    _notify_plan_change(user["id"])
    flash("You are now on the Basic plan.", "success")
    return redirect(url_for("auth.account"))


@bp.route("/admin/users")
@admin_required
def admin_users():
    users = [dict(u, effective_plan=db.effective_plan(u)) for u in db.list_users()]
    return render_template("auth/admin_users.html", users=users, statuses=db.PLAN_STATUSES)


@bp.route("/admin/users/<int:user_id>/plan", methods=["POST"])
@admin_required
def admin_set_plan(user_id):
    _check_csrf()
    user = db.get_user(user_id)
    if user is None:
        abort(404)
    plan = request.form.get("plan", "")
    status = request.form.get("plan_status", "")
    renews = request.form.get("plan_renews_at", "").strip() or None
    try:
        if renews:
            date.fromisoformat(renews)
        db.update_user(user_id, plan=plan, plan_status=status, plan_renews_at=renews)
    except ValueError as exc:
        flash(f"Not saved: {exc}", "error")
        return redirect(url_for("auth.admin_users"))
    _notify_plan_change(user_id)
    flash(f"{user['email']}: plan set to {PLANS[plan]['name']} ({status}).", "success")
    return redirect(url_for("auth.admin_users"))
