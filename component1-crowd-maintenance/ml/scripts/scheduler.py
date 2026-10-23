"""
Maintenance Scheduler
SLIIT FYP - Component 1
Core Novelty: AI-driven maintenance scheduling

NOTE: The original scheduler.py source was missing from the uploaded
archive (only the compiled __pycache__/scheduler.cpython-313.pyc was
present). This file is a faithful reconstruction recovered from that
bytecode: same THRESHOLDS (person-hours), same priority escalation
(MEDIUM -> HIGH at 1.5x -> CRITICAL at 2.0x of threshold), same
per-action completion tracking and JSON report output.
"""

import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


class MaintenanceScheduler:
    # Bather-load thresholds (person-hours) per maintenance action —
    # recovered exactly from the original compiled module.
    THRESHOLDS = {
        "chlorine_dose": 20,
        "filter_backwash": 50,
        "skimmer_clean": 30,
        "shock_treatment": 100,
        "deep_clean": 200,
    }

    def __init__(self):
        self.maintenance_log = []
        self.last_maintenance = {action: 0 for action in self.THRESHOLDS}
        self.total_load = 0

    def update_load(self, bather_load_hours):
        self.total_load = bather_load_hours

    def get_recommendations(self):
        recommendations = []
        for action, threshold in self.THRESHOLDS.items():
            load_since = self.total_load - self.last_maintenance[action]
            if load_since >= threshold:
                ratio = load_since / threshold
                if ratio >= 2.0:
                    priority = "CRITICAL 🔴"
                elif ratio >= 1.5:
                    priority = "HIGH 🟠"
                else:
                    priority = "MEDIUM 🟡"
                recommendations.append(
                    {
                        "action": action.replace("_", " ").title(),
                        "priority": priority,
                        "overdue_by": round(load_since - threshold, 2),
                    }
                )
        return sorted(recommendations, key=lambda r: r["priority"], reverse=True)

    def mark_completed(self, action):
        self.last_maintenance[action] = self.total_load
        self.maintenance_log.append(
            {
                "action": action,
                "completed_at": datetime.now().isoformat(),
                "load_at_completion": self.total_load,
            }
        )

    def print_report(self):
        recommendations = self.get_recommendations()
        print("\n" + "=" * 50)
        print("🏊 POOL MAINTENANCE SCHEDULE REPORT")
        print("=" * 50)
        print(f"Current Bather Load: {self.total_load} person-hours")
        print(f"Pending Actions:     {len(recommendations)}")
        if not recommendations:
            print("✅ No maintenance needed!")
        else:
            print("\n📋 RECOMMENDED ACTIONS:")
            print("-" * 50)
            for rec in recommendations:
                print(f"  Action:   {rec['action']}")
                print(f"  Priority: {rec['priority']}")
                print(f"  Overdue:  {rec['overdue_by']}")
        out_dir = BASE_DIR / "results"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "maintenance_schedule.json", "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "bather_load": self.total_load,
                    "recommendations": recommendations,
                },
                f,
                indent=2,
            )
        print("\n✅ Report saved!")
