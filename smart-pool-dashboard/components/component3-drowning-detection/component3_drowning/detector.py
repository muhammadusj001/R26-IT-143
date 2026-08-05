"""
Component 3 — Swimmer Behaviour & Drowning Detection (runtime service).

Business logic ported UNCHANGED from the original
component3_drowning_detection/dashboard/app.py:
  - three classes: Drowning / Person out of water / Swimming
  - same colours, same confidence threshold, same consecutive-frame
    alert logic (DROWNING_FRAMES) before raising an alert
  - SAFE/DANGER status behaviour preserved

Only the surrounding architecture changed: the service now receives
frames from the shared CameraManager instead of opening its own camera,
and reports results into SystemState instead of a private global dict.
If the trained model file (best.pt) is absent, the service degrades to
"model_missing" mode instead of crashing.
"""

from datetime import datetime

from component3_drowning.drawing import draw_box, draw_alert_banner

DEFAULT_CONF = 0.35        # preserved from original dashboard/app.py
DEFAULT_CONSEC_FRAMES = 3  # preserved from original scripts

# Colours preserved exactly from Component 3
COLORS = {
    "Drowning": (0, 0, 255),
    "Person out of water": (0, 165, 255),
    "Swimming": (0, 255, 0),
}


class DrowningDetector:
    def __init__(self, model_path=None, conf_threshold=DEFAULT_CONF,
                 consec_frames=DEFAULT_CONSEC_FRAMES):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.consec_frames = consec_frames
        self.model = None
        self.model_status = "not_loaded"
        self.consec_drown = 0
        self.total_alerts = 0
        self.status = "SAFE"

    def load(self):
        if self.model_path is None or not self.model_path.exists():
            self.model_status = "model_missing"
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
        """
        Run detection on one shared frame.
        Returns (frame, result_dict, alert_or_None).
        """
        counts = {"Drowning": 0, "Person out of water": 0, "Swimming": 0}

        if self.model is None:
            return frame, self._result(counts), None

        results = self.model(
            frame,
            imgsz=416,
            conf=self.conf_threshold,
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = COLORS.get(cls_name, (255, 255, 255))
                thickness = 4 if cls_name == "Drowning" else 2
                draw_box(
                    frame, x1, y1, x2, y2, f"{cls_name} {conf:.0%}", color, thickness
                )
                if cls_name in counts:
                    counts[cls_name] += 1

        # ── Alert logic preserved from Component 3 ───────────
        alert = None
        if counts["Drowning"] > 0:
            self.consec_drown += 1
            if self.consec_drown >= self.consec_frames:
                self.total_alerts += 1
                self.status = "DANGER"
                alert = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "count": counts["Drowning"],
                }
                draw_alert_banner(frame, "DROWNING DETECTED!")
        else:
            self.consec_drown = 0
            if self.status == "DANGER":
                self.status = "SAFE"

        return frame, self._result(counts), alert

    def _result(self, counts):
        return {
            "swimming": counts["Swimming"],
            "drowning": counts["Drowning"],
            "out_of_water": counts["Person out of water"],
            "status": self.status,
            "total_alerts": self.total_alerts,
            "model_status": self.model_status,
        }
