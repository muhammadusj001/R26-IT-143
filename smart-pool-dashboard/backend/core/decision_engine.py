"""
Central Decision Engine.

This is the ONLY place where information from multiple modules is
combined. Each AI module stays fully independent; the engine reads their
outputs from SystemState and produces system-level intelligence:

  - overall risk level  (drowning status escalated by crowd density)
  - maintenance urgency (bather load combined with water quality trend)

This implements the "Decision Intelligence Layer" of the project's
five-layer architecture.
"""


class DecisionEngine:
    def evaluate(self, snapshot: dict) -> dict:
        crowd = snapshot["crowd"]
        drowning = snapshot["drowning"]
        garbage = snapshot["garbage"]
        water = snapshot["water_quality"]

        notes = []

        # ── Overall safety risk ──────────────────────────────
        crowd_at_high_density = crowd["density_level"] in ("HIGH", "OVER CAPACITY")
        garbage_high_risk = garbage.get("risk_level") == "HIGH"

        overall_risk = "NORMAL"
        if drowning["status"] == "DANGER":
            overall_risk = "EMERGENCY"
            if crowd_at_high_density:
                notes.append(
                    "Drowning alert during HIGH crowd density — "
                    "visual confirmation may be slower, priority escalated."
                )
        elif water["status"] == "CRITICAL":
            overall_risk = "HIGH"
            notes.append("Water quality CRITICAL — swimming not advised.")
        elif crowd_at_high_density or water["status"] == "WARNING" or garbage_high_risk:
            overall_risk = "ELEVATED"

        if crowd["density_level"] == "OVER CAPACITY":
            notes.append(
                f"Pool is OVER CAPACITY ({crowd.get('swimmer_count', 0)}/"
                f"{crowd.get('max_capacity', 0)} bathers) — restrict entry."
            )

        if garbage_high_risk:
            notes.append(
                f"Intentional littering detected (source: {garbage.get('source', 'unknown')}, "
                f"{garbage.get('intent_confidence', 0)}% confidence) — risk escalated, "
                "consider dispatching staff."
            )

        # ── Maintenance urgency (crowd load × water trend) ───
        urgency = "LOW"
        load_flag = crowd.get("pending_actions", 0) > 0
        if water["status"] == "CRITICAL":
            urgency = "IMMEDIATE"
            notes.append("Immediate water treatment required (CRITICAL reading).")
        elif load_flag and water["status"] == "WARNING":
            urgency = "HIGH"
            notes.append(
                "High bather load combined with WARNING water quality — "
                "schedule maintenance earlier than planned."
            )
        elif load_flag or water["status"] == "WARNING":
            urgency = "MEDIUM"

        if garbage["alert_status"] != "CLEAR":
            notes.append("Foreign objects in pool — cleaning required.")
            if urgency == "LOW":
                urgency = "MEDIUM"

        return {
            "overall_risk": overall_risk,
            "maintenance_urgency": urgency,
            "notes": notes[:5],
        }


decision_engine = DecisionEngine()
