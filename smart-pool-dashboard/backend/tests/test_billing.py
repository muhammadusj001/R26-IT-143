"""
PayHere billing tests — hashes, checkout form and the notify handler.

Run from smart-pool-dashboard/backend:
    python -m unittest discover -s tests -v

Uses a throwaway SQLite database and dummy PayHere credentials; nothing
is sent to PayHere.
"""

import base64
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
from core import billing, db, payhere_api  # noqa: E402
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
        settings.PAYHERE_SANDBOX = True
        settings.PAYHERE_APP_ID = settings.PAYHERE_APP_SECRET = ""
        settings.PAYHERE_RETRIEVAL_CONFIGURED = False
        payhere_api._token["value"] = None
        billing._last_verify.clear()
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
        # notify goes to the public URL; the browser returns to the host it is on
        self.assertEqual(f["notify_url"], "https://example.ngrok.app/billing/notify")
        self.assertEqual(f["return_url"], "http://localhost/billing/success")
        self.assertEqual(f["cancel_url"], "http://localhost/billing/cancel")
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
        # acknowledged (always 200) but ignored
        self.assertEqual((r.status_code, r.data), (200, b"rejected: bad signature or merchant id"))
        self.assertEqual(self.user()["plan"], "basic")
        self.assertEqual(self.payments()[0]["status"], "pending")
        log = self.notifications(f["order_id"])
        self.assertEqual((len(log), log[0]["signature_ok"], log[0]["dedupe_key"]), (1, 0, None))

    def test_wrong_merchant_id_rejected(self):
        _, f = self.checkout("pro")
        sig = notify_signature("999999", f["order_id"], "15.00", "USD", "2", SECRET)
        r = self.notify(f["order_id"], merchant_id="999999", md5sig=sig)
        self.assertIn(b"rejected", r.data)
        self.assertEqual(self.user()["plan"], "basic")

    def test_amount_or_currency_mismatch_rejected(self):
        _, f = self.checkout("plus")
        self.assertEqual(self.notify(f["order_id"], amount="15.00").data,  # Pro price for Plus
                         b"rejected: amount/currency mismatch")
        self.assertEqual(self.notify(f["order_id"], currency="LKR").data,
                         b"rejected: amount/currency mismatch")
        self.assertEqual(self.user()["plan"], "basic")
        self.assertEqual(self.payments()[0]["status"], "pending")

    def test_unknown_order_rejected(self):
        form = {"merchant_id": MERCHANT_ID, "order_id": "NOPE", "payhere_amount": "15.00",
                "payhere_currency": "USD", "status_code": "2",
                "md5sig": notify_signature(MERCHANT_ID, "NOPE", "15.00", "USD", "2", SECRET)}
        r = A.app.test_client().post("/billing/notify", data=form)
        self.assertEqual((r.status_code, r.data), (200, b"rejected: unknown order"))

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

    def test_return_urls_follow_the_users_host(self):
        r = self.web.get("/billing/checkout/plus", base_url="http://localhost:5000")
        f = dict(re.findall(r'<input type="hidden" name="(\w+)" value="([^"]*)"', r.get_data(as_text=True)))
        self.assertEqual(f["return_url"], "http://localhost:5000/billing/success")
        self.assertEqual(f["cancel_url"], "http://localhost:5000/billing/cancel")
        self.assertEqual(f["notify_url"], "https://example.ngrok.app/billing/notify")
        # behind a TLS-terminating tunnel the scheme comes from X-Forwarded-Proto
        # (the session cookie is per host, so sign in on the tunnel host first)
        tunnel, base = A.app.test_client(), "http://abc.ngrok-free.dev"
        page = tunnel.get("/login", base_url=base).get_data(as_text=True)
        token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        tunnel.post("/login", base_url=base, data={"csrf": token, "email": self.email,
                                                   "password": "secret123", "next": "/"})
        r = tunnel.get("/billing/checkout/plus", base_url=base, headers={"X-Forwarded-Proto": "https"})
        f = dict(re.findall(r'<input type="hidden" name="(\w+)" value="([^"]*)"', r.get_data(as_text=True)))
        self.assertEqual(f["return_url"], "https://abc.ngrok-free.dev/billing/success")

    def test_status_endpoint_and_success_page_after_valid_notify(self):
        _, f = self.checkout("pro")
        oid = f["order_id"]
        anon = A.app.test_client()
        self.assertEqual(anon.get(f"/billing/status/{oid}").get_json(),
                         {"status": "pending", "plan": "pro", "plan_name": "Pro", "renews_at": None})
        self.assertEqual(self.notify(oid).status_code, 200)
        status = anon.get(f"/billing/status/{oid}").get_json()
        self.assertEqual(status, {"status": "paid", "plan": "pro", "plan_name": "Pro",
                                  "renews_at": "2026-11-18"})

        page = self.web.get(f"/billing/success?order_id={oid}").get_data(as_text=True)
        for text in ("activating your Pro plan", "Welcome to AquaGuard PRO", "Drowning detection &amp; alerts",
                     "PDF session reports", "Go to Dashboard", "View Account", 'data-owner="1"'):
            self.assertIn(text, page)
        self.assertNotIn("Garbage &amp; littering detection", page)  # not part of Pro
        self.assertIn(">PRO</a>", page)  # header badge (plan already active)

    def test_success_page_without_session(self):
        _, f = self.checkout("plus")
        self.notify(f["order_id"])
        anon = A.app.test_client()  # e.g. PayHere sent the browser to the ngrok host
        r = anon.get(f"/billing/success?order_id={f['order_id']}", base_url="https://abc.ngrok-free.dev")
        page = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Welcome to AquaGuard PLUS", page)
        self.assertIn('href="http://localhost:5000/">Open dashboard', page)
        self.assertIn('data-owner="0"', page)
        self.assertNotIn(self.email, page)  # no personal data
        self.assertNotIn("Nimal", page)

    def test_success_page_never_blank(self):
        anon = A.app.test_client()
        for url in ("/billing/success", "/billing/success?order_id=NOPE"):
            r = anon.get(url)
            self.assertEqual(r.status_code, 200)
            self.assertIn("couldn't find this payment", r.get_data(as_text=True))
        r = anon.post("/billing/success", data={"order_id": "NOPE"})  # PayHere POST-back
        self.assertEqual(r.status_code, 200)
        self.assertEqual(anon.get("/billing/status/NOPE").status_code, 404)

    def test_return_url_never_changes_plan(self):
        _, f = self.checkout("pro")
        r = self.web.get(f"/billing/success?order_id={f['order_id']}")
        self.assertIn("activating your Pro plan", r.get_data(as_text=True))
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

    # ── notify reachability ──────────────────────────────────
    def test_notify_without_session_from_any_host_no_redirect(self):
        for base in ("https://abrasion-grafting-oxford.ngrok-free.dev", "http://127.0.0.1:5000"):
            with self.subTest(base=base):
                _, f = self.checkout("pro")
                order = db.get_order(f["order_id"])
                form = {"merchant_id": MERCHANT_ID, "order_id": f["order_id"],
                        "payment_id": f"{self.user_id}{len(base)}", "payhere_amount": "15.00",
                        "payhere_currency": "USD", "status_code": "2", "subscription_id": "S1",
                        "md5sig": notify_signature(MERCHANT_ID, f["order_id"], order["amount"],
                                                   "USD", "2", SECRET)}
                r = A.app.test_client().post("/billing/notify", data=form, base_url=base)
                self.assertEqual(r.status_code, 200)
                self.assertNotIn("Location", r.headers)
                self.assertEqual(db.get_order(f["order_id"])["status"], "paid")
                db.update_user(self.user_id, plan="basic", subscription_id=None)

    def test_notify_answers_200_even_on_internal_error(self):
        _, f = self.checkout("pro")
        original = billing._apply
        billing._apply = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            with self.assertLogs("billing", "ERROR"):
                r = self.notify(f["order_id"])
        finally:
            billing._apply = original
        self.assertEqual((r.status_code, r.data), (200, b"error: see server log"))

    # ── Retrieval API verification ───────────────────────────
    def enable_retrieval(self):
        settings.PAYHERE_APP_ID, settings.PAYHERE_APP_SECRET = "4OVx", "app-secret-xyz"
        settings.PAYHERE_RETRIEVAL_CONFIGURED = True

    class FakeHttp:
        """Stands in for `requests`: records calls, answers like PayHere."""
        def __init__(self, payments, search_status=1):
            self.calls, self.payments, self.search_status = [], payments, search_status

        class Resp:
            def __init__(self, body, code=200):
                self.status_code, self._body = code, body

            def json(self):
                return self._body

        def post(self, url, headers=None, data=None, timeout=None):
            self.calls.append(("POST", url, headers, data))
            return self.Resp({"access_token": "tok-123", "token_type": "bearer", "expires_in": 599})

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append(("GET", url, headers, params))
            if self.search_status == -1:
                return self.Resp({"status": -1, "msg": "No payments found", "data": None})
            return self.Resp({"status": 1, "msg": "ok", "data": self.payments})

    def received(self, order_id, **over):
        p = {"payment_id": int(f"{self.user_id}320025"), "order_id": order_id,
             "date": "2026-10-19 10:15:00", "status": "RECEIVED", "currency": "USD", "amount": 15}
        p.update(over)
        return p

    def test_verify_order_with_mocked_retrieval_api_upgrades(self):
        self.enable_retrieval()
        _, f = self.checkout("pro")
        http = self.FakeHttp([self.received(f["order_id"])])
        status, detail = billing.verify_order(f["order_id"], http=http)
        self.assertEqual(status, "paid", detail)
        # OAuth: Basic base64(app_id:app_secret) + client_credentials, then Bearer search
        (m1, url1, h1, d1), (m2, url2, h2, p2) = http.calls
        self.assertEqual((m1, url1), ("POST", "https://sandbox.payhere.lk/merchant/v1/oauth/token"))
        self.assertEqual(h1["Authorization"], "Basic " + base64.b64encode(b"4OVx:app-secret-xyz").decode())
        self.assertEqual(d1, {"grant_type": "client_credentials"})
        self.assertEqual((m2, url2, p2), ("GET", "https://sandbox.payhere.lk/merchant/v1/payment/search",
                                          {"order_id": f["order_id"]}))
        self.assertEqual(h2["Authorization"], "Bearer tok-123")
        u = self.user()
        self.assertEqual((u["plan"], u["plan_status"], u["plan_renews_at"]), ("pro", "active", "2026-11-19"))
        p = self.payments()[0]
        self.assertEqual((p["status"], p["payhere_payment_id"]), ("paid", f"{self.user_id}320025"))
        # PayHere's late notify for the same payment: no second charge row, adds the subscription id
        self.assertEqual(self.notify(f["order_id"]).status_code, 200)
        self.assertEqual(len(self.payments()), 1)
        self.assertEqual(self.user()["subscription_id"], "420075")

    def test_verify_order_rejects_wrong_amount_currency_or_status(self):
        self.enable_retrieval()
        _, f = self.checkout("pro")
        for bad in (self.received(f["order_id"], amount=25),
                    self.received(f["order_id"], currency="LKR", amount=15),
                    self.received(f["order_id"], status="REFUNDED")):
            with self.subTest(bad=bad):
                status, _ = billing.verify_order(f["order_id"], force=True, http=self.FakeHttp([bad]))
                self.assertEqual(status, "pending")
        status, detail = billing.verify_order(f["order_id"], force=True,
                                              http=self.FakeHttp([], search_status=-1))
        self.assertEqual((status, detail.startswith("0 payment")), ("pending", True))
        self.assertEqual(self.user()["plan"], "basic")

    def test_verify_order_without_api_key_does_nothing(self):
        _, f = self.checkout("pro")
        http = self.FakeHttp([self.received(f["order_id"])])
        self.assertEqual(billing.verify_order(f["order_id"], http=http),
                         ("pending", "retrieval API not configured"))
        self.assertEqual(http.calls, [])

    def test_status_endpoint_verifies_pending_order_at_most_every_3s(self):
        self.enable_retrieval()
        _, f = self.checkout("pro")
        calls = []
        original = payhere_api.search_payments
        payhere_api.search_payments = lambda order_id, **kw: calls.append(order_id) or []
        try:
            for _ in range(3):
                self.assertEqual(A.app.test_client().get(f"/billing/status/{f['order_id']}").get_json()["status"],
                                 "pending")
            self.assertEqual(len(calls), 1)  # throttled
            billing._last_verify.clear()
            payhere_api.search_payments = lambda order_id, **kw: [self.received(order_id)]
            status = A.app.test_client().get(f"/billing/status/{f['order_id']}").get_json()
        finally:
            payhere_api.search_payments = original
        self.assertEqual((status["status"], status["plan"]), ("paid", "pro"))
        self.assertEqual(self.user()["plan"], "pro")

    # ── account / admin ──────────────────────────────────────
    def test_new_checkout_abandons_old_pending_but_late_payment_counts(self):
        _, first = self.checkout("pro")
        _, second = self.checkout("pro")
        self.assertEqual(db.get_order(first["order_id"])["status"], "abandoned")
        self.assertEqual(db.get_order(second["order_id"])["status"], "pending")
        self.assertIn("ABANDONED", self.web.get("/account").get_data(as_text=True))
        self.notify(first["order_id"])  # the user did pay the first one after all
        self.assertEqual((db.get_order(first["order_id"])["status"], self.user()["plan"]), ("paid", "pro"))

    def test_check_status_button(self):
        self.enable_retrieval()
        _, f = self.checkout("plus")
        page = self.web.get("/account").get_data(as_text=True)
        self.assertIn(f'/billing/check/{f["order_id"]}', page)
        token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        original = payhere_api.search_payments
        payhere_api.search_payments = lambda order_id, **kw: [self.received(order_id, amount="25.00")]
        try:
            r = self.web.post(f"/billing/check/{f['order_id']}", data={"csrf": token}, follow_redirects=True)
        finally:
            payhere_api.search_payments = original
        self.assertIn("Payment confirmed", r.get_data(as_text=True))
        self.assertEqual(self.user()["plan"], "plus")
        self.assertNotIn("Check status", self.web.get("/account").get_data(as_text=True))

    def admin_client(self):
        email = f"admin{self._n}@example.com"
        db.create_user("Admin", email, generate_password_hash("adminpass1"), is_admin=True)
        client = A.app.test_client()
        page = client.get("/login").get_data(as_text=True)
        token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        client.post("/login", data={"csrf": token, "email": email, "password": "adminpass1", "next": "/"})
        return client

    def test_simulate_confirmation_sandbox_only(self):
        _, f = self.checkout("pro")
        admin = self.admin_client()
        page = admin.get("/admin/users").get_data(as_text=True)
        self.assertIn(f["order_id"], page)
        self.assertIn("Simulate confirmation", page)
        token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        r = admin.post(f"/admin/payments/{f['order_id']}/simulate", data={"csrf": token}, follow_redirects=True)
        self.assertIn("paid: user", r.get_data(as_text=True))
        self.assertEqual((db.get_order(f["order_id"])["status"], self.user()["plan"]), ("paid", "pro"))
        # not available on live
        settings.PAYHERE_SANDBOX = False
        self.assertNotIn("Simulate confirmation", admin.get("/admin/users").get_data(as_text=True))
        self.assertEqual(admin.post(f"/admin/payments/{f['order_id']}/simulate",
                                    data={"csrf": token}).status_code, 404)
        # and a normal user can't use it
        self.assertEqual(self.web.post(f"/admin/payments/{f['order_id']}/simulate",
                                       data={"csrf": token}).status_code, 403)


if __name__ == "__main__":
    unittest.main()
