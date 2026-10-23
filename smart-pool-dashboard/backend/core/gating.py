"""
Per-viewer filtering of shared data.

There is one camera and one vision loop, but every viewer has their own
plan. Before anything leaves the server it is filtered for the viewer's
plan: locked modules are replaced by their default (empty) values with
"locked": True, locked modules' alerts are dropped and the decision
engine is neutral without "decision_engine".
"""

import copy

from core.decision_engine import decision_engine
from core.plans import PLANS, has_feature, plan_payload
from core.state import SystemState

MODULES = ("crowd", "drowning", "garbage", "water_quality")
# Every feature any plan can lock
GATED = {f for p in PLANS.values() for f in p["features"]}
_DEFAULTS = SystemState().data  # a fresh, never-mutated default state


def _visible(plan_key, module):
    return module not in GATED or has_feature(plan_key, module)


def filter_state(snapshot, plan_key):
    out = dict(snapshot)
    for module in MODULES:
        if has_feature(plan_key, module):
            out[module] = {**snapshot[module], "locked": False}
        else:
            out[module] = {**copy.deepcopy(_DEFAULTS[module]), "locked": True}
    if has_feature(plan_key, "decision_engine"):
        out["decision"] = {**snapshot["decision"], "locked": False}
    else:
        out["decision"] = {**decision_engine.neutral(), "locked": True}
    out["alerts"] = [a for a in snapshot["alerts"] if _visible(plan_key, a["module"])]
    out["plan"] = plan_payload(plan_key)
    return out


def filter_summary(summary, plan_key):
    """Session summary / report data without locked modules."""
    out = dict(summary)
    alerts = [a for a in summary.get("alerts", []) if _visible(plan_key, a["module"])]
    by_module, by_severity = {}, {}
    for a in alerts:
        by_module[a["module"]] = by_module.get(a["module"], 0) + 1
        by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1
    out.update({
        "alerts": alerts,
        "total_alerts": len(alerts),
        "alert_counts_by_module": by_module,
        "alert_counts_by_severity": by_severity,
    })
    if not has_feature(plan_key, "drowning"):
        out["drowning_alerts"] = 0
    if not has_feature(plan_key, "garbage"):
        out["garbage_events"] = 0
    if "model_status" in out:
        out["model_status"] = {
            m: (s if _visible(plan_key, m) else "locked")
            for m, s in out["model_status"].items()
        }
    return out
