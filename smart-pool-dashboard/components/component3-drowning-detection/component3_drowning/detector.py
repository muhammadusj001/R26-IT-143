"""
Component 3 — Swimmer Behaviour & Drowning Detection (runtime service).

Based on the original component3_drowning_detection/dashboard/app.py:
  - three classes: Drowning / Person out of water / Swimming
  - same colours and SAFE/DANGER status semantics

Robustness changes on top of the original logic:
  - class-conflict filter: a Drowning box is ignored when a stronger
    Swimming box overlaps it (IoU > 0.5), which removes most of the
    Swimming/Drowning flicker on the same person
  - sliding-window smoothing with hysteresis instead of a plain
    consecutive-frame counter (DANGER at >= 60% of the window,
    back to SAFE at <= 20%)
  - one alert per incident (only on the SAFE -> DANGER transition)
  - last drawn boxes are kept so the dashboard can redraw them on
    frames where the model is not run (draw_last)

Only the surrounding architecture changed: the service now receives
frames from the shared CameraManager instead of opening its own camera,
and reports results into SystemState instead of a private global dict.
If the trained model file (best.pt) is absent, the service degrades to
"model_missing" mode instead of crashing.
"""

from collections import deque
from datetime import datetime

from component3_drowning.drawing import draw_box, draw_alert_banner

DEFAULT_CONF = 0.5         # filters the 40-55% Drowning false positives
DEFAULT_CONSEC_FRAMES = 3  # preserved from original scripts (window min is 5)
DEFAULT_IMGSZ = 640        # best.pt (YOLOv8s) was trained at imgsz=640

# Sliding-window thresholds (fraction of recent frames with a Drowning box)
DANGER_RATIO = 0.6
SAFE_RATIO = 0.2
CONFLICT_IOU = 0.5

# Colours preserved exactly from Component 3
COLORS = {
    "Drowning": (0, 0, 255),
    "Person out of water": (0, 165, 255),
    "Swimming": (0, 255, 0),
}


def _iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


class DrowningDetector:
    def __init__(self, model_path=None, conf_threshold=DEFAULT_CONF,
                 consec_frames=DEFAULT_CONSEC_FRAMES, imgsz=DEFAULT_IMGSZ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.consec_frames = consec_frames
        self.imgsz = imgsz
        self.model = None
        self.model_status = "not_loaded"
        self.history = deque(maxlen=max(5, consec_frames))
        self.last_boxes = []
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
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                xyxy = tuple(map(int, box.xyxy[0].tolist()))
                detections.append((cls_name, conf, xyxy))

        # ── Class-conflict filter ────────────────────────────
        # Drop a Drowning box if a more confident Swimming box covers
        # the same person (the model often fires both classes at once).
        swimming = [(c, b) for n, c, b in detections if n == "Swimming"]
        kept = []
        for cls_name, conf, xyxy in detections:
            if cls_name == "Drowning" and any(
                s_conf > conf and _iou(xyxy, s_box) > CONFLICT_IOU
                for s_conf, s_box in swimming
            ):
                continue
            kept.append((cls_name, conf, xyxy))

        self.last_boxes = []
        for cls_name, conf, (x1, y1, x2, y2) in kept:
            color = COLORS.get(cls_name, (255, 255, 255))
            thickness = 4 if cls_name == "Drowning" else 2
            self.last_boxes.append(
                (x1, y1, x2, y2, f"{cls_name} {conf:.0%}", color, thickness)
            )
            if cls_name in counts:
                counts[cls_name] += 1

        # ── Sliding window + hysteresis ──────────────────────
        self.history.append(1 if counts["Drowning"] > 0 else 0)
        ratio = self._ratio()
        alert = None
        if len(self.history) == self.history.maxlen and ratio >= DANGER_RATIO:
            # Alert once per incident: only on the SAFE -> DANGER transition
            if self.status != "DANGER":
                self.total_alerts += 1
                alert = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "count": counts["Drowning"],
                }
            self.status = "DANGER"
        elif ratio <= SAFE_RATIO:
            self.status = "SAFE"

        self.draw_last(frame)
        return frame, self._result(counts), alert

    def draw_last(self, frame):
        """Redraw the last boxes (and the banner while DANGER) on a frame
        the model did not run on, so the dashboard overlay doesn't flicker."""
        for x1, y1, x2, y2, label, color, thickness in self.last_boxes:
            draw_box(frame, x1, y1, x2, y2, label, color, thickness)
        if self.status == "DANGER":
            draw_alert_banner(frame, "DROWNING DETECTED!")
        return frame

    def _ratio(self):
        return sum(self.history) / len(self.history) if self.history else 0.0

    def _result(self, counts):
        return {
            "swimming": counts["Swimming"],
            "drowning": counts["Drowning"],
            "out_of_water": counts["Person out of water"],
            "status": self.status,
            "total_alerts": self.total_alerts,
            "model_status": self.model_status,
            "drown_ratio": round(self._ratio(), 2),
        }
