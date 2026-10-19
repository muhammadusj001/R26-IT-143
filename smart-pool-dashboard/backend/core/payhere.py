"""
PayHere hashing (server-side only — the merchant secret never reaches
the browser). Formulas from the PayHere Checkout / Recurring API docs.
"""

import hashlib
import hmac
from decimal import Decimal, ROUND_HALF_UP


def _md5_upper(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest().upper()


def format_amount(amount):
    """PayHere amounts: 2 decimals, '.' separator, no thousands separator."""
    return str(Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def checkout_hash(merchant_id, order_id, amount, currency, merchant_secret):
    """hash = UPPER(MD5(merchant_id + order_id + amount_2dp + currency + UPPER(MD5(secret))))"""
    return _md5_upper(
        f"{merchant_id}{order_id}{format_amount(amount)}{currency}{_md5_upper(merchant_secret)}"
    )


def notify_signature(merchant_id, order_id, payhere_amount, payhere_currency, status_code,
                     merchant_secret):
    """md5sig = UPPER(MD5(merchant_id + order_id + payhere_amount + payhere_currency
    + status_code + UPPER(MD5(secret)))) — amount is used exactly as PayHere sent it."""
    return _md5_upper(
        f"{merchant_id}{order_id}{payhere_amount}{payhere_currency}{status_code}"
        f"{_md5_upper(merchant_secret)}"
    )


def signature_valid(form, merchant_id, merchant_secret):
    expected = notify_signature(
        merchant_id, form.get("order_id", ""), form.get("payhere_amount", ""),
        form.get("payhere_currency", ""), form.get("status_code", ""), merchant_secret,
    )
    return hmac.compare_digest(expected, form.get("md5sig", "").upper())
