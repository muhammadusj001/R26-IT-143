"""
Account management commands. Run from smart-pool-dashboard/backend:

    python manage.py create-admin                       # prompts for details
    python manage.py create-admin --email a@b.com --name Admin --plan plus
    python manage.py list-users
    python manage.py set-plan a@b.com pro [--status active]

The password is always asked for interactively (or read from the
ADMIN_PASSWORD environment variable for scripted setups), never taken
as a command-line argument where it would end up in shell history.
"""

import argparse
import getpass
import os
import sqlite3
import sys

from werkzeug.security import generate_password_hash

from core import db
from core.auth import MIN_PASSWORD_LENGTH
from core.plans import PLANS


def _ask(prompt, default=None):
    value = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return value or default


def _ask_password():
    password = os.environ.get("ADMIN_PASSWORD")
    if password:
        return password
    while True:
        password = getpass.getpass("Password: ")
        if len(password) < MIN_PASSWORD_LENGTH:
            print(f"  Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        elif password != getpass.getpass("Repeat password: "):
            print("  Passwords do not match.")
        else:
            return password


def create_admin(args):
    email = (args.email or _ask("Email") or "").strip().lower()
    if "@" not in email:
        sys.exit("A valid email is required.")
    name = args.name or _ask("Name", "Admin")
    password = _ask_password()
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    existing = db.get_user_by_email(email)
    if existing:
        # Promote an existing account instead of failing
        db.update_user(existing["id"], is_admin=1, plan=args.plan, plan_status="active",
                       password_hash=generate_password_hash(password))
        print(f"Updated {email}: now an admin on the {PLANS[args.plan]['name']} plan.")
        return
    try:
        db.create_user(name, email, generate_password_hash(password), plan=args.plan, is_admin=True)
    except sqlite3.IntegrityError:
        sys.exit(f"{email} already exists.")
    print(f"Created admin {email} on the {PLANS[args.plan]['name']} plan.")


def list_users(_args):
    users = db.list_users()
    if not users:
        print("No users yet.")
    for u in users:
        flag = " (admin)" if u["is_admin"] else ""
        print(f"{u['id']:>3}  {u['email']:<32} {u['plan']:<6} {u['plan_status']:<9} "
              f"in effect: {db.effective_plan(u)}{flag}")


def set_plan(args):
    user = db.get_user_by_email(args.email)
    if user is None:
        sys.exit(f"No user with email {args.email}.")
    db.update_user(user["id"], plan=args.plan, plan_status=args.status)
    print(f"{user['email']}: plan {args.plan} ({args.status}).")


def main():
    parser = argparse.ArgumentParser(description="Smart Pool account management")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-admin", help="create (or promote) an admin user")
    p.add_argument("--email")
    p.add_argument("--name")
    p.add_argument("--plan", choices=list(PLANS), default="plus")
    p.set_defaults(func=create_admin)

    p = sub.add_parser("list-users", help="list all users")
    p.set_defaults(func=list_users)

    p = sub.add_parser("set-plan", help="change a user's plan")
    p.add_argument("email")
    p.add_argument("plan", choices=list(PLANS))
    p.add_argument("--status", choices=db.PLAN_STATUSES, default="active")
    p.set_defaults(func=set_plan)

    args = parser.parse_args()
    db.init_db()
    args.func(args)


if __name__ == "__main__":
    main()
