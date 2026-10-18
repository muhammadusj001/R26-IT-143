"""
Subscription plans and feature gating.

Each plan unlocks a set of features. The backend enforces them (locked
vision modules are not run, the decision engine returns a neutral result,
the report endpoint returns 403) — the frontend only reflects what the
"plan" field in the state payload says is unlocked.
"""

PLANS = {
    "basic": {"name": "Basic", "price_usd": 0,  "features": ["crowd", "water_quality"]},
    "pro":   {"name": "Pro",   "price_usd": 15, "features": ["crowd", "water_quality", "drowning", "reports"]},
    "plus":  {"name": "Plus",  "price_usd": 25, "features": ["crowd", "water_quality", "drowning", "reports",
                                                            "garbage", "decision_engine"]},
}

# Unknown plan keys fall back to the most restrictive plan (fail closed)
FALLBACK_PLAN = "basic"


def get_plan(key):
    """Return the plan dict for key, or None if there is no such plan."""
    return PLANS.get(str(key).strip().lower()) if key is not None else None


def has_feature(plan_key, feature):
    plan = get_plan(plan_key)
    return plan is not None and feature in plan["features"]


def required_plan(feature):
    """Cheapest plan that includes feature (e.g. "reports" -> "pro")."""
    candidates = [k for k, p in PLANS.items() if feature in p["features"]]
    return min(candidates, key=lambda k: PLANS[k]["price_usd"]) if candidates else None


def resolve_plan_key(key):
    """Normalise a configured plan key, falling back to FALLBACK_PLAN."""
    plan_key = str(key).strip().lower()
    if plan_key not in PLANS:
        print(f"  WARNING: unknown ACTIVE_PLAN {key!r} — using {FALLBACK_PLAN!r}")
        return FALLBACK_PLAN
    return plan_key


def plan_payload(plan_key):
    """The {key, name, features} dict sent to the frontend."""
    plan = PLANS[plan_key]
    return {"key": plan_key, "name": plan["name"], "features": list(plan["features"])}
