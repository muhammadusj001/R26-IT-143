"""
Bather Load Calculator
SLIIT FYP - Component 1
Crowd-Aware Maintenance Scheduling Module

NOTE: The original bather_load.py source was missing from the uploaded
archive (only the compiled __pycache__/bather_load.cpython-313.pyc was
present). This file is a faithful reconstruction of the original module
recovered from that bytecode: same class, methods, field names, hour-key
format, person-seconds accumulation and JSON output.
"""

import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


class BatherLoadCalculator:
    def __init__(self):
        self.records = []
        self.hourly_loads = {}

    def add_reading(self, swimmer_count, interval_seconds=5):
        """Record one detection reading and accumulate person-seconds."""
        now = datetime.now()
        hour = now.strftime("%Y-%m-%d %H:00")
        person_seconds = swimmer_count * interval_seconds
        self.records.append(
            {
                "timestamp": now.isoformat(),
                "swimmer_count": swimmer_count,
                "person_seconds": person_seconds,
                "hour": hour,
            }
        )
        self.hourly_loads[hour] = self.hourly_loads.get(hour, 0) + person_seconds

    def get_current_load(self):
        """Today's cumulative bather load in person-hours."""
        today = datetime.now().strftime("%Y-%m-%d")
        return (
            sum(
                r["person_seconds"]
                for r in self.records
                if r["timestamp"].startswith(today)
            )
            / 3600
        )

    def get_peak_hour(self):
        if not self.hourly_loads:
            return None
        return max(self.hourly_loads, key=self.hourly_loads.get)

    def get_summary(self):
        return {
            "total_readings": len(self.records),
            "current_bather_load_hours": round(self.get_current_load(), 2),
            "peak_hour": self.get_peak_hour(),
        }

    def save_to_file(self):
        out_dir = BASE_DIR / "results"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / "bather_load_data.json"
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"✅ Saved to: {path}")
