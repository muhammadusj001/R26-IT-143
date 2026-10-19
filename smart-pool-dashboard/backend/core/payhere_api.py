"""
PayHere Retrieval API client (merchant/v1): look up the payments PayHere
received for an order, independent of the notify_url callback.

  POST {base}/merchant/v1/oauth/token    Basic base64(app_id:app_secret),
                                         grant_type=client_credentials
  GET  {base}/merchant/v1/payment/search?order_id=...   Bearer token

Search answers {"status": 1, "data": [{payment_id, order_id, status:
"RECEIVED", currency, amount, ...}]} or {"status": -1, "msg": "No payments
found"}. Rate limit: 20 requests / 10 s (callers throttle).
"""

import base64
import logging
import threading
import time

import requests

from config import settings

log = logging.getLogger("billing")
TIMEOUT_S = 10

_token = {"value": None, "expires": 0.0}
_token_lock = threading.Lock()


class RetrievalError(Exception):
    pass


def configured():
    return settings.PAYHERE_RETRIEVAL_CONFIGURED


def _access_token(http=requests):
    with _token_lock:
        if _token["value"] and time.time() < _token["expires"]:
            return _token["value"]
        basic = base64.b64encode(
            f"{settings.PAYHERE_APP_ID}:{settings.PAYHERE_APP_SECRET}".encode()).decode()
        resp = http.post(
            f"{settings.PAYHERE_API_BASE}/merchant/v1/oauth/token",
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials"}, timeout=TIMEOUT_S,
        )
        if resp.status_code != 200:
            raise RetrievalError(f"token request failed: HTTP {resp.status_code}")
        body = resp.json()
        _token["value"] = body["access_token"]
        _token["expires"] = time.time() + max(int(body.get("expires_in", 600)) - 60, 30)
        return _token["value"]


def search_payments(order_id, http=requests):
    """Payments PayHere has for order_id (list of dicts; [] if none)."""
    token = _access_token(http)
    resp = http.get(
        f"{settings.PAYHERE_API_BASE}/merchant/v1/payment/search",
        params={"order_id": order_id},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=TIMEOUT_S,
    )
    if resp.status_code == 401:
        _token["value"] = None  # expired/revoked — next call fetches a new one
    if resp.status_code != 200:
        raise RetrievalError(f"payment search failed: HTTP {resp.status_code}")
    body = resp.json()
    if body.get("status") == -1:
        return []
    return body.get("data") or []
