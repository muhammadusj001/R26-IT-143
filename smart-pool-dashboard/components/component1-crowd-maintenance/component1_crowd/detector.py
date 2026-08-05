"""
Component 1 — Crowd-Aware Maintenance Scheduling (runtime service).

Uses the team's REAL Component 1 assets (from the second upload):
  - trained single-class YOLOv8 swimmer model
    (models/crowd/best_swimmer_model.pt — v1, the higher-accuracy model;
     v2_combined is also provided and selectable via CROWD_MODEL_PATH)
  - BatherLoadCalculator (recovered from bytecode, see bather_load.py)
  - MaintenanceScheduler with its original person-hour thresholds
    (recovered from bytecode, see scheduler.py)

Pipeline: swimmer detection -> count -> density level ->
BatherLoadCalculator.add_reading -> MaintenanceScheduler.get_recommendations
Falls back to simulation mode when the model file is absent so the
unified dashboard still runs.
"""

import random
import time

from component1_crowd.drawing import draw_box
from component1_crowd.bather_load import BatherLoadCalculator
from component1_crowd.scheduler import MaintenanceScheduler

# Standalone defaults (the dashboard can override via constructor)
DEFAULT_CONF = 0.4
DENSITY_MODERATE = 4
DENSITY_HIGH = 8

SWIMMER_COLOR = (255, 200, 0)


class CrowdDetector:
    def __init__(self, model_path=None, conf_threshold=DEFAULT_CONF,
                 density_moderate=DENSITY_MODERATE, density_high=DENSITY_HIGH):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.density_moderate = density_moderate
        self.density_high = density_high
        self.model = None
        self.model_status = "not_loaded"
        self.bather_load = BatherLoadCalculator()
        self.scheduler = MaintenanceScheduler()
        self._last_ts = None
        self._sim_count = 3

    def load(self):
        if self.model_path is None or not self.model_path.exists():
            self.model_status = "simulation"
            return False
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.model_status = "loaded"
            return True
        except Exception as exc:  # noqa: BLE001
            self.model_status = f"error: {exc}"
            return False

    def density_level(self, count: int) -> str:
        if count >= self.density_high:
            return "HIGH"
        if count >= self.density_moderate:
            return "MODERATE"
        return "LOW"

    def process(self, frame):
        count = self._detect(frame) if self.model is not None else self._simulate()

        # ── Original C1 logic: reading -> load -> recommendations ──
        now = time.time()
        interval = (now - self._last_ts) if self._last_ts else 5
        self._last_ts = now
        self.bather_load.add_reading(count, interval_seconds=interval)

        load_hours = round(self.bather_load.get_current_load(), 2)
        self.scheduler.update_load(load_hours)
        recs = self.scheduler.get_recommendations()

        if recs:
            top = recs[0]
            recommendation = f"{top['action']} — {top['priority']} (overdue by {top['overdue_by']} ph)"
        else:
            recommendation = "No maintenance needed"

        return frame, {
            "swimmer_count": count,
            "density_level": self.density_level(count),
            "bather_load": load_hours,
            "peak_hour": self.bather_load.get_peak_hour(),
            "pending_actions": len(recs),
            "recommendations": recs[:3],
            "maintenance_recommendation": recommendation,
            "model_status": self.model_status,
        }

    def _detect(self, frame) -> int:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        count = 0
        for result in results:
            for box in result.boxes:
                name = self.model.names[int(box.cls[0])]
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                draw_box(frame, x1, y1, x2, y2,
                         f"{name} {float(box.conf[0]):.0%}", SWIMMER_COLOR)
        return count

    def _simulate(self) -> int:
        self._sim_count = max(0, min(12, self._sim_count + random.choice([-1, 0, 0, 1])))
        return self._sim_count
