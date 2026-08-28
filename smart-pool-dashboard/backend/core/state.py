"""
Central system state.

Each module writes ONLY into its own section; the dashboard receives the
whole object. Modules never read or modify another module's section —
cross-module reasoning happens exclusively in the DecisionEngine.
"""

import threading
from datetime import datetime

from config import settings


class SystemState:
    def __init__(self):
        self._lock = threading.Lock()
        self.data = {
            "running": False,
            "fps": 0.0,
            "frame_num": 0,
            "session_time": "00:00",
            "camera_connected": False,
            "crowd": {
                "swimmer_count": 0,
                "density_level": "LOW",
                "bather_load": 0.0,
                "peak_hour": None,
                "pending_actions": 0,
                "recommendations": [],
                "maintenance_recommendation": "No maintenance needed",
                "model_status": "not_loaded",
                "max_capacity": 0,
                "occupancy_pct": 0.0,
                "total_people_detected": 0,
                "people_outside_pool": 0,
            },
            "drowning": {
                "swimming": 0,
                "drowning": 0,
                "out_of_water": 0,
                "status": "SAFE",
                "total_alerts": 0,
                "model_status": "not_loaded",
            },
            "garbage": {
                "objects_detected": 0,
                "object_labels": [],
                "non_garbage_labels": [],
                "alert_status": "CLEAR",
                "total_events": 0,
                "model_status": "not_loaded",
                "source": "unknown",
                "intent": "none",
                "intent_score": 0.0,
                "intent_confidence": 0,
                "risk_level": "NORMAL",
            },
            "water_quality": {
                "ph": None,
                "temperature": None,
                "chlorine": None,
                "turbidity": None,
                "tds": None,
                "status": "UNKNOWN",
                "sensor_connected": False,
                "model_status": "not_loaded",
            },
            "decision": {
                "overall_risk": "NORMAL",
                "maintenance_urgency": "LOW",
                "notes": [],
            },
            "alerts": [],  # unified recent-alert feed (all modules)
        }

    def update_module(self, module: str, values: dict):
        with self._lock:
            self.data[module].update(values)

    def update_root(self, values: dict):
        with self._lock:
            self.data.update(values)

    def add_alert(self, module: str, message: str, severity: str = "warning"):
        with self._lock:
            self.data["alerts"].insert(
                0,
                {
                    "module": module,
                    "message": message,
                    "severity": severity,
                    "time": datetime.now().strftime("%H:%M:%S"),
                },
            )
            self.data["alerts"] = self.data["alerts"][: settings.MAX_ALERT_HISTORY]

    def snapshot(self) -> dict:
        with self._lock:
            # Shallow copy is enough for JSON serialisation via socketio
            return {
                k: (v.copy() if isinstance(v, (dict, list)) else v)
                for k, v in self.data.items()
            }


state = SystemState()
