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
        overall_risk = "NORMAL"
        if drowning["status"] == "DANGER":
            overall_risk = "EMERGENCY"
            if crowd["density_level"] == "HIGH":
                notes.append(
                    "Drowning alert during HIGH crowd density — "
                    "visual confirmation may be slower, priority escalated."
                )
        elif water["status"] == "CRITICAL":
            overall_risk = "HIGH"
            notes.append("Water quality CRITICAL — swimming not advised.")
        elif crowd["density_level"] == "HIGH" or water["status"] == "WARNING":
            overall_risk = "ELEVATED"

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
