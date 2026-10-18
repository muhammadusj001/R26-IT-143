"""
PayHere billing tests — hashes, checkout form and the notify handler.

Run from smart-pool-dashboard/backend:
    python -m unittest discover -s tests -v

Uses a throwaway SQLite database and dummy PayHere credentials; nothing
is sent to PayHere.
"""

import hashlib
import os
import re
import shutil
import sys
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="billing-test-")
os.environ.update(
    DB_PATH=os.path.join(_TMP, "app.db"), CAMERA_SOURCE="simulate", DEMO_MODE="0",
    SECRET_KEY="test-secret", PAYHERE_MERCHANT_ID="1211149", PAYHERE_MERCHANT_SECRET="test-merchant-secret",
    PAYHERE_SANDBOX="1", PUBLIC_BASE_URL="https://example.ngrok.app",
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werkzeug.security import generate_password_hash  # noqa: E402

import app as A  # noqa: E402
from config import settings  # noqa: E402
from core import db  # noqa: E402
from core.payhere import checkout_hash, format_amount, notify_signature  # noqa: E402

MERCHANT_ID = "1211149"
SECRET = "test-merchant-secret"


def md5u(text):
    return hashlib.md5(text.encode()).hexdigest().upper()


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


class HashTests(unittest.TestCase):
    def test_format_amount(self):
        self.assertEqual(format_amount(15), "15.00")
        self.assertEqual(format_amount("25.5"), "25.50")
        self.assertEqual(format_amount(1000), "1000.00")  # no thousands separator
        self.assertEqual(format_amount("0.005"), "0.01")

    def test_checkout_hash_formula(self):
        expected = md5u(MERCHANT_ID + "AG1-ABC" + "15.00" + "USD" + md5u(SECRET))
        self.assertEqual(checkout_hash(MERCHANT_ID, "AG1-ABC", 15, "USD", SECRET), expected)

    def test_checkout_hash_known_vector(self):
        # Fixed vector (regression guard): uppercase hex, amount formatted to 2 dp
        h = checkout_hash("1211149", "ItemNo12345", "1000", "LKR", "MjM0NTY3ODkw")
        self.assertEqual(h, md5u("1211149ItemNo123451000.00LKR" + md5u("MjM0NTY3ODkw")))
        self.assertRegex(h, r"^[0-9A-F]{32}$")

    def test_notify_signature_formula(self):
        expected = md5u(MERCHANT_ID + "AG1-ABC" + "15.00" + "USD" + "2" + md5u(SECRET))
        self.assertEqual(notify_signature(MERCHANT_ID, "AG1-ABC", "15.00", "USD", "2", SECRET), expected)

    def test_notify_signature_depends_on_every_field(self):
        base = notify_signature(MERCHANT_ID, "AG1-ABC", "15.00", "USD", "2", SECRET)
        for args in [("999", "AG1-ABC", "15.00", "USD", "2", SECRET),
                     (MERCHANT_ID, "AG1-XYZ", "15.00", "USD", "2", SECRET),
                     (MERCHANT_ID, "AG1-ABC", "25.00", "USD", "2", SECRET),
                     (MERCHANT_ID, "AG1-ABC", "15.00", "LKR", "2", SECRET),
                     (MERCHANT_ID, "AG1-ABC", "15.00", "USD", "-2", SECRET),
                     (MERCHANT_ID, "AG1-ABC", "15.00", "USD", "2", "other-secret")]:
            self.assertNotEqual(notify_signature(*args), base, args)


class BillingFlowTests(unittest.TestCase):
    _n = 0

    def setUp(self):
        settings.BILLING_CONFIGURED = True
        BillingFlowTests._n += 1
        self.email = f"user{self._n}@example.com"
        self.user_id = db.create_user("Nimal Perera", self.email, generate_password_hash("secret123"))
        db.update_user(self.user_id, phone="0771234567", address="1 Galle Rd", city="Colombo")
        self.web = A.app.test_client()
        page = self.web.get("/login").get_data(as_text=True)
        token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        self.web.post("/login", data={"csrf": token, "email": self.email,
                                      "password": "secret123", "next": "/"})

    # helpers
    def checkout(self, plan="pro"):
        r = self.web.get(f"/billing/checkout/{plan}")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        fields = dict(re.findall(r'<input type="hidden" name="(\w+)" value="([^"]*)"', html))
        return html, fields

    def notify(self, order_id, status_code="2", amount=None, currency="USD", **extra):
        order = db.get_order(order_id)
        amount = amount or order["amount"]
        # PayHere payment ids are globally unique; make them unique per user here
        payment_id = f"{self.user_id}{extra.pop('payment_id', '320025')}"
        form = {
            "merchant_id": MERCHANT_ID, "order_id": order_id, "payment_id": payment_id,
            "payhere_amount": amount, "payhere_currency": currency, "status_code": status_code,
            "subscription_id": "420075", "message_type": "AUTHORIZATION_SUCCESS",
            "item_rec_status": "0", "item_rec_date_next": "2026-11-18", "recurring": "1",
            "custom_1": str(self.user_id), "custom_2": order["plan"], "card_no": "************1292",
        }
        form.update(extra)
        form.setdefault("md5sig", notify_signature(MERCHANT_ID, order_id, amount, currency,
                                                   status_code, SECRET))
        return A.app.test_client().post("/billing/notify", data=form)  # no session/login

    def user(self):
        return db.get_user(self.user_id)

    def payments(self):
        return db.list_payments(self.user_id)

    def notifications(self, order_id):
        with db.connect() as conn:
            return conn.execute("SELECT * FROM payhere_notifications WHERE order_id = ? ORDER BY id",
                                (order_id,)).fetchall()

    # checkout
    def test_checkout_renders_signed_form(self):
        html, f = self.checkout("pro")
        self.assertIn('action="https://sandbox.payhere.lk/pay/checkout"', html)
        self.assertNotIn(SECRET, html)
        self.assertEqual((f["items"], f["amount"], f["currency"], f["recurrence"], f["duration"]),
                         ("AquaGuard Pro", "15.00", "USD", "1 Month", "Forever"))
        self.assertEqual(f["notify_url"], "https://example.ngrok.app/billing/notify")
        self.assertEqual(f["return_url"], "https://example.ngrok.app/billing/success")
        self.assertEqual(f["cancel_url"], "https://example.ngrok.app/billing/cancel")
        self.assertEqual((f["first_name"], f["last_name"], f["country"]), ("Nimal", "Perera", "Sri Lanka"))
        self.assertEqual((f["custom_1"], f["custom_2"]), (str(self.user_id), "pro"))
        self.assertEqual(f["hash"], checkout_hash(MERCHANT_ID, f["order_id"], "15.00", "USD", SECRET))
        order = db.get_order(f["order_id"])
        self.assertEqual((order["plan"], order["amount"], order["status"]), ("pro", "15.00", "pending"))
        _, f2 = self.checkout("plus")
        self.assertEqual((f2["items"], f2["amount"]), ("AquaGuard Plus", "25.00"))

    def test_checkout_asks_for_missing_billing_details(self):
        db.update_user(self.user_id, phone=None)
        r = self.web.get("/billing/checkout/pro")
        self.assertIn("Continue to PayHere", r.get_data(as_text=True))
        self.assertIn('name="phone"', r.get_data(as_text=True))
        self.assertEqual(self.payments(), [])  # no order until details are given

    def test_billing_not_configured(self):
        settings.BILLING_CONFIGURED = False
        self.assertIn("Billing not configured — ask admin", self.web.get("/pricing").get_data(as_text=True))
        r = self.web.get("/billing/checkout/pro")
        self.assertIn("Billing not configured — ask admin", r.get_data(as_text=True))
        self.assertEqual(self.payments(), [])

    # notify
    def test_valid_success_activates_plan(self):
        _, f = self.checkout("pro")
        r = self.notify(f["order_id"])
        self.assertEqual(r.status_code, 200, r.data)
        u = self.user()
        self.assertEqual((u["plan"], u["plan_status"], u["subscription_id"], u["plan_renews_at"]),
                         ("pro", "active", "420075", "2026-11-18"))
        p = self.payments()[0]
        self.assertEqual((p["status"], p["payhere_payment_id"], p["subscription_id"]),
                         ("paid", f"{self.user_id}320025", "420075"))
        log = self.notifications(f["order_id"])
        self.assertEqual(len(log), 1)
        self.assertTrue(log[0]["signature_ok"])
        self.assertNotIn("card_no", log[0]["payload"])  # card details are not stored

    def test_bad_signature_rejected(self):
        _, f = self.checkout("pro")
        r = self.notify(f["order_id"], md5sig="0" * 32)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.user()["plan"], "basic")
        self.assertEqual(self.payments()[0]["status"], "pending")
        log = self.notifications(f["order_id"])
        self.assertEqual((len(log), log[0]["signature_ok"], log[0]["dedupe_key"]), (1, 0, None))

    def test_wrong_merchant_id_rejected(self):
        _, f = self.checkout("pro")
        sig = notify_signature("999999", f["order_id"], "15.00", "USD", "2", SECRET)
        r = self.notify(f["order_id"], merchant_id="999999", md5sig=sig)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.user()["plan"], "basic")

    def test_amount_or_currency_mismatch_rejected(self):
        _, f = self.checkout("plus")
        self.assertEqual(self.notify(f["order_id"], amount="15.00").status_code, 400)  # Pro price for Plus
        self.assertEqual(self.notify(f["order_id"], currency="LKR").status_code, 400)
        self.assertEqual(self.user()["plan"], "basic")
        self.assertEqual(self.payments()[0]["status"], "pending")

    def test_unknown_order_rejected(self):
        form = {"merchant_id": MERCHANT_ID, "order_id": "NOPE", "payhere_amount": "15.00",
                "payhere_currency": "USD", "status_code": "2",
                "md5sig": notify_signature(MERCHANT_ID, "NOPE", "15.00", "USD", "2", SECRET)}
        self.assertEqual(A.app.test_client().post("/billing/notify", data=form).status_code, 400)

    def test_duplicate_notification_processed_once(self):
        _, f = self.checkout("pro")
        first, second = self.notify(f["order_id"]), self.notify(f["order_id"])
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertIn(b"duplicate", second.data)
        self.assertEqual(len(self.payments()), 1)
        log = self.notifications(f["order_id"])
        self.assertEqual(len(log), 2)  # both logged...
        self.assertEqual(sum(1 for n in log if n["dedupe_key"]), 1)  # ...processed once

    def test_failed_status_marks_payment_failed(self):
        _, f = self.checkout("pro")
        r = self.notify(f["order_id"], status_code="-2", message_type="AUTHORIZATION_FAILED")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.payments()[0]["status"], "failed")
        self.assertEqual(self.user()["plan"], "basic")

    def test_cancelled_status(self):
        _, f = self.checkout("pro")
        self.notify(f["order_id"], status_code="-1", message_type="AUTHORIZATION_FAILED")
        self.assertEqual(self.payments()[0]["status"], "cancelled")
        self.assertEqual(self.user()["plan"], "basic")

    def test_pending_status_changes_nothing(self):
        _, f = self.checkout("pro")
        self.assertEqual(self.notify(f["order_id"], status_code="0").status_code, 200)
        self.assertEqual((self.payments()[0]["status"], self.user()["plan"]), ("pending", "basic"))
        self.notify(f["order_id"])  # the success that follows still activates
        self.assertEqual(self.user()["plan"], "pro")

    def test_renewal_installment_adds_payment_and_extends(self):
        _, f = self.checkout("pro")
        self.notify(f["order_id"])
        r = self.notify(f["order_id"], payment_id="320026", message_type="RECURRING_INSTALLMENT_SUCCESS",
                        item_rec_date_next="2026-12-18")
        self.assertEqual(r.status_code, 200)
        self.assertEqual([p["status"] for p in self.payments()], ["paid", "paid"])
        self.assertEqual(self.user()["plan_renews_at"], "2026-12-18")

    def test_failed_renewal_keeps_plan_until_renewal_date(self):
        _, f = self.checkout("pro")
        self.notify(f["order_id"])
        self.notify(f["order_id"], status_code="-2", payment_id="320026",
                    message_type="RECURRING_INSTALLMENT_FAILED")
        self.assertEqual(self.payments()[0]["status"], "failed")
        self.assertEqual(self.user()["plan"], "pro")  # lapses when plan_renews_at passes

    def test_chargeback_downgrades(self):
        _, f = self.checkout("plus")
        self.notify(f["order_id"])
        self.assertEqual(self.user()["plan"], "plus")
        self.notify(f["order_id"], status_code="-3", message_type="RECURRING_INSTALLMENT_FAILED")
        u = self.user()
        self.assertEqual((u["plan"], u["subscription_id"]), ("basic", None))
        self.assertEqual(self.payments()[0]["status"], "chargedback")

    def test_stopped_subscription_downgrades(self):
        _, f = self.checkout("pro")
        self.notify(f["order_id"])
        self.notify(f["order_id"], status_code="-2", payment_id="320027",
                    message_type="RECURRING_STOPPED", item_rec_status="-1")
        self.assertEqual(self.user()["plan"], "basic")

    def test_return_url_never_changes_plan(self):
        _, f = self.checkout("pro")
        r = self.web.get(f"/billing/success?order_id={f['order_id']}")
        self.assertIn("activating your plan", r.get_data(as_text=True))
        self.assertEqual(self.user()["plan"], "basic")
        me = self.web.get(f"/api/me?order_id={f['order_id']}").get_json()
        self.assertEqual((me["effective_plan"], me["order"]["status"]), ("basic", "pending"))
        self.notify(f["order_id"])
        me = self.web.get(f"/api/me?order_id={f['order_id']}").get_json()
        self.assertEqual((me["effective_plan"], me["order"]["status"]), ("pro", "paid"))

    def test_cancel_returns_to_pricing(self):
        r = self.web.get("/billing/cancel", follow_redirects=True)
        self.assertIn("Payment cancelled", r.get_data(as_text=True))
        self.assertIn("Choose your plan", r.get_data(as_text=True))

    def test_active_subscription_blocks_second_checkout_and_free_downgrade(self):
        _, f = self.checkout("pro")
        self.notify(f["order_id"])
        r = self.web.get("/billing/checkout/plus", follow_redirects=True)
        self.assertIn("already have an active subscription", r.get_data(as_text=True))
        page = self.web.get("/pricing").get_data(as_text=True)
        token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        r = self.web.post("/account/plan/basic", data={"csrf": token}, follow_redirects=True)
        self.assertIn("ask the admin to cancel", r.get_data(as_text=True))
        self.assertEqual(self.user()["plan"], "pro")

    def test_account_shows_payment_history(self):
        _, f = self.checkout("pro")
        self.notify(f["order_id"])
        page = self.web.get("/account").get_data(as_text=True)
        self.assertIn("Payment history", page)
        self.assertIn(f["order_id"], page)
        self.assertIn("PAID", page)

    def test_api_me_requires_login(self):
        self.assertEqual(A.app.test_client().get("/api/me").status_code, 401)


if __name__ == "__main__":
    unittest.main()
