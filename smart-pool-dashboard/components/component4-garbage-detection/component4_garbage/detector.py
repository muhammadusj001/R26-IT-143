"""
Component 4 — Garbage Intrusion Detection (runtime service).

Built from the member's REAL assets:
  - trained model: models/swimming_pool_garbage_yolo.pt
    (YOLOv8n, 6 classes, trained on the Roboflow "swimming-pool" dataset,
     CC BY 4.0 — MUST be cited in the thesis)
  - business logic preserved from the original legacy_flask_app/app.py:
    only GARBAGE_CLASSES trigger alerts; 'ball' (a toy) and 'leaf' are
    detected and drawn but do NOT raise a garbage alert. This class
    disambiguation reduces false positives on legitimate pool objects.

Falls back to simulation mode when the model file is absent.
"""

import random

from component4_garbage.drawing import draw_box

ALL_CLASSES = ["aluminium foil", "ball", "bottle", "juice", "leaf", "thermocol"]
# Preserved exactly from the member's original app.py:
GARBAGE_CLASSES = ["aluminium foil", "bottle", "juice", "thermocol"]

GARBAGE_COLOR = (0, 255, 255)
NON_GARBAGE_COLOR = (200, 200, 200)
SIM_OBJECTS = ["bottle", "aluminium foil", "juice", "thermocol"]


class GarbageDetector:
    def __init__(self, model_path=None, conf_threshold=0.25, imgsz=None):
        # conf 0.25 preserved from the original app.py
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz            # None = ultralytics default inference size
        self.model = None
        self.model_status = "not_loaded"
        self.total_events = 0
        self._sim_cooldown = 0

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

    def process(self, frame):
        """Returns (frame, result_dict, alert_message_or_None)."""
        if self.model is not None:
            garbage_labels, other_labels = self._detect(frame)
        else:
            garbage_labels, other_labels = self._simulate(), []

        alert = None
        if garbage_labels:
            self.total_events += 1
            alert = f"Garbage detected: {', '.join(garbage_labels)}"

        return frame, {
            "objects_detected": len(garbage_labels) + len(other_labels),
            "object_labels": garbage_labels,
            "non_garbage_labels": other_labels,
            "alert_status": "ALERT" if garbage_labels else "CLEAR",
            "total_events": self.total_events,
            "model_status": self.model_status,
        }, alert

    def _detect(self, frame):
        """Original class-disambiguation logic: only GARBAGE_CLASSES alert."""
        kwargs = {"conf": self.conf_threshold, "verbose": False}
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        garbage, others = [], []
        results = self.model(frame, **kwargs)
        for result in results:
            for box in result.boxes:
                name = self.model.names[int(box.cls[0])]
                is_garbage = name in GARBAGE_CLASSES
                (garbage if is_garbage else others).append(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                draw_box(
                    frame, x1, y1, x2, y2,
                    f"{name} {float(box.conf[0]):.0%}",
                    GARBAGE_COLOR if is_garbage else NON_GARBAGE_COLOR,
                )
        return garbage, others

    def _simulate(self):
        if self._sim_cooldown > 0:
            self._sim_cooldown -= 1
            return [random.choice(SIM_OBJECTS)]
        if random.random() < 0.01:
            self._sim_cooldown = 20
        return []
