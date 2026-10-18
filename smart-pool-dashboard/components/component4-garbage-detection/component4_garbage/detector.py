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

INTENT-AWARE GARBAGE DETECTION (step 1 — source identification):
An optional second model (a stock COCO model, e.g. yolov8n.pt) detects
people and animals near a piece of detected garbage. identify_source()
finds the nearest person/bird/cat/dog to the highest-confidence garbage
box, as groundwork for later distinguishing "someone just littered"
from "debris blew in." This is entirely optional — if the source model
isn't configured or fails to load, garbage detection itself is
completely unaffected; source identification is just skipped.

INTENT-AWARE GARBAGE DETECTION (step 3 — intent + risk level):
When the identified source is a person, frames are fed into an
IntentAnalyzer (see intent_analyzer.py) to score throwing-motion
signals (arm raise, wrist speed, forward reach) — never for
animals/unknown sources, both to save CPU and because a stale human
pose shouldn't be attributed to an unrelated animal/unknown event.
compute_risk_level() then maps (source, intent) to NORMAL/MEDIUM/HIGH.
If MediaPipe is unavailable or no pose was found, intent safely
defaults to "accidental" with 0 confidence — see intent_analyzer.py.
"""

import math
import random

from component4_garbage.drawing import draw_box
from component4_garbage.intent_analyzer import IntentAnalyzer

ALL_CLASSES = ["aluminium foil", "ball", "bottle", "juice", "leaf", "thermocol"]
# Preserved exactly from the member's original app.py:
GARBAGE_CLASSES = ["aluminium foil", "bottle", "juice", "thermocol"]

GARBAGE_COLOR = (0, 255, 255)
NON_GARBAGE_COLOR = (200, 200, 200)
SIM_OBJECTS = ["bottle", "aluminium foil", "juice", "thermocol"]

# ── Source identification (COCO person/animal model) ──────────
SOURCE_COLOR = (255, 0, 255)  # magenta (BGR)
DEFAULT_MAX_SOURCE_DISTANCE = 400  # pixels
# COCO class id -> reported source label
SOURCE_CLASS_MAP = {0: "human", 14: "bird", 15: "cat", 16: "dog"}

# Safe "no intent signal" placeholder — used whenever intent can't be
# (or shouldn't be) assessed: no garbage this frame, non-human source,
# or MediaPipe/pose data unavailable. Matches IntentAnalyzer.classify_intent()'s
# own safe-default shape exactly.
NO_INTENT_SIGNAL = {
    "intent": "accidental",
    "intent_score": 0.0,
    "confidence": 0,
    "signals": {"arm_raise": 0.0, "peak_speed": 0.0, "forward_motion": 0.0},
}


class GarbageDetector:
    def __init__(self, model_path=None, conf_threshold=0.25, imgsz=None,
                 source_model_path=None):
        # conf 0.25 preserved from the original app.py
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz            # None = ultralytics default inference size
        self.model = None
        self.model_status = "not_loaded"
        self.total_events = 0
        self._sim_cooldown = 0

        # Optional COCO person/animal model for identify_source().
        self.source_model_path = source_model_path
        self.source_model = None
        self.source_model_status = "not_loaded"
        self.source_identification_enabled = False

        # Throwing-motion intent analyzer (step 3) — fed frames only
        # when identify_source() reports a human nearby.
        self.intent_analyzer = IntentAnalyzer()

    def load(self):
        # Source model is independent of the garbage model — attempt it
        # regardless of whether the garbage model itself loads.
        self._load_source_model()

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

    def _load_source_model(self):
        """Loads the optional COCO source-identification model. Missing
        file or load failure just disables source identification — it
        never affects garbage detection itself."""
        if self.source_model_path is None or not self.source_model_path.exists():
            self.source_model_status = "not_available"
            self.source_identification_enabled = False
            return False
        try:
            from ultralytics import YOLO

            self.source_model = YOLO(str(self.source_model_path))
            self.source_model_status = "loaded"
            self.source_identification_enabled = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.source_model_status = f"error: {exc}"
            self.source_identification_enabled = False
            return False

    def process(self, frame):
        """Returns (frame, result_dict, alert_message_or_None)."""
        if self.model is not None:
            garbage_labels, other_labels, best_garbage_box = self._detect(frame)
        else:
            garbage_labels, other_labels, best_garbage_box = self._simulate(), [], None

        alert = None
        source = "unknown"
        source_distance = None
        intent_info = NO_INTENT_SIGNAL

        if garbage_labels:
            self.total_events += 1

            if best_garbage_box is not None:
                src = self.identify_source(best_garbage_box, frame)
                source = src["source"]
                source_distance = src["distance"]
                if src["source_box"] is not None:
                    sx1, sy1, sx2, sy2 = src["source_box"]
                    draw_box(frame, sx1, sy1, sx2, sy2,
                             f"SOURCE: {source}", SOURCE_COLOR, thickness=2)

                # Only feed frames / assess intent for a human source —
                # saves CPU (skips MediaPipe entirely for animals/unknown)
                # and avoids attributing a stale human pose to an
                # unrelated animal/unknown-source event.
                if source == "human":
                    self.intent_analyzer.add_frame(frame)
                    intent_info = self.intent_analyzer.classify_intent()

            risk_level = self.compute_risk_level(source, intent_info["intent"])
            alert = (f"Garbage detected: {', '.join(garbage_labels)} — "
                     f"risk: {risk_level}, intent: {intent_info['intent']} (source: {source})")
        else:
            risk_level = self.compute_risk_level(None, None)

        return frame, {
            "objects_detected": len(garbage_labels) + len(other_labels),
            "object_labels": garbage_labels,
            "non_garbage_labels": other_labels,
            "alert_status": "ALERT" if garbage_labels else "CLEAR",
            "total_events": self.total_events,
            "model_status": self.model_status,
            "source": source,
            "source_distance": source_distance,
            "intent": intent_info["intent"],
            "intent_score": intent_info["intent_score"],
            "intent_confidence": intent_info["confidence"],
            "risk_level": risk_level,
            "signals": intent_info["signals"],
        }, alert

    def compute_risk_level(self, source, intent):
        """Maps a detection event's (source, intent) pair to an overall
        risk level:
            human + intentional -> HIGH
            human + accidental  -> MEDIUM
            animal (any)        -> MEDIUM
            unknown source      -> MEDIUM
            no garbage (source=None) -> NORMAL
        """
        if source is None:
            return "NORMAL"
        if source == "human":
            return "HIGH" if intent == "intentional" else "MEDIUM"
        return "MEDIUM"  # bird / cat / dog / unknown

    def _detect(self, frame):
        """Original class-disambiguation logic: only GARBAGE_CLASSES alert.
        Also tracks the highest-confidence garbage box for identify_source()."""
        kwargs = {"conf": self.conf_threshold, "verbose": False}
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        garbage, others = [], []
        best_conf = -1.0
        best_box = None
        results = self.model(frame, **kwargs)
        for result in results:
            for box in result.boxes:
                name = self.model.names[int(box.cls[0])]
                is_garbage = name in GARBAGE_CLASSES
                (garbage if is_garbage else others).append(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                draw_box(
                    frame, x1, y1, x2, y2,
                    f"{name} {conf:.0%}",
                    GARBAGE_COLOR if is_garbage else NON_GARBAGE_COLOR,
                )
                if is_garbage and conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)
        return garbage, others, best_box

    def identify_source(self, garbage_box, frame, max_distance=DEFAULT_MAX_SOURCE_DISTANCE):
        """Finds the nearest person/bird/cat/dog to a detected garbage box.

        garbage_box = (x1, y1, x2, y2) in pixel coordinates.
        Returns {"source": "human"|"bird"|"cat"|"dog"|"unknown",
                 "distance": float or None, "source_box": (x1,y1,x2,y2) or None}.
        "unknown" (source_box=None, distance=None) if source identification
        isn't available, nothing was detected, or the nearest match is
        further than max_distance pixels from the garbage box's centre.
        """
        result = {"source": "unknown", "distance": None, "source_box": None}
        if not self.source_identification_enabled or self.source_model is None:
            return result

        gx1, gy1, gx2, gy2 = garbage_box
        g_center = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)

        kwargs = {
            "conf": self.conf_threshold, "verbose": False,
            "classes": list(SOURCE_CLASS_MAP.keys()),
        }
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz

        results = self.source_model(frame, **kwargs)
        best_distance = None
        best_box = None
        best_label = None
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in SOURCE_CLASS_MAP:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                distance = math.hypot(center[0] - g_center[0], center[1] - g_center[1])
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_box = (x1, y1, x2, y2)
                    best_label = SOURCE_CLASS_MAP[cls_id]

        if best_distance is None or best_distance > max_distance:
            return result

        result["source"] = best_label
        result["distance"] = round(best_distance, 1)
        result["source_box"] = best_box
        return result

    def _simulate(self):
        if self._sim_cooldown > 0:
            self._sim_cooldown -= 1
            return [random.choice(SIM_OBJECTS)]
        if random.random() < 0.01:
            self._sim_cooldown = 20
        return []
