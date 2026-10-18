"""
Component 4 — Intent-aware garbage detection: throwing-motion pose analysis.

Step 2 of intent-aware garbage detection (see identify_source() in
detector.py for step 1). This module is NOT wired into detector.py yet
— it's a self-contained pose analyzer that will later help distinguish
"someone just threw litter into the pool" from "debris blew in" by
looking at arm-raise, wrist speed, and forward reach over a short
rolling window of frames.

Uses MediaPipe's Tasks API (mediapipe.tasks.python.vision.PoseLandmarker)
in IMAGE running mode — one synchronous detect() call per add_frame().
The older "Solutions" API (mp.solutions.pose.Pose(static_image_mode=...,
model_complexity=...)) was targeted originally, but no mediapipe release
with a Python 3.13 wheel still ships it (confirmed: 0.10.30 through
1.0.1, the only versions available for 3.13, all lack mp.solutions
entirely) — Google removed it in favour of the Tasks API. This rewrite
keeps every public method's name, signature and behaviour identical to
the original spec; only the internal MediaPipe wiring changed.

Landmarks used (33-point pose model, same indices in both APIs):
  11 / 12 = left / right shoulder
  13 / 14 = left / right elbow
  15 / 16 = left / right wrist

Model asset: the Tasks API needs a .task model file that the mediapipe
pip package does NOT bundle (unlike the old Solutions API, which had
weights compiled in). Default location is models/pose_landmarker_lite.task
next to this component's other models. Download it from:
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

Self-contained: no imports from the dashboard backend or any other
component. Degrades gracefully in every failure mode — missing
mediapipe install, missing model asset file, or a load/inference error
all just disable the analyzer (available=False) instead of raising;
every method keeps returning its documented safe default.
"""

import math
from collections import deque
from pathlib import Path

import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    mp_vision = None
    BaseOptions = None
    _MEDIAPIPE_AVAILABLE = False

# MediaPipe Pose landmark indices used for throwing-motion analysis
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

DEFAULT_BUFFER_SIZE = 20
# This component's own models/ folder, matching every other model file
# in this repo (best_swimmer_model.pt, swimming_pool_garbage_yolo.pt, ...).
DEFAULT_MODEL_ASSET_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"

# ── Intent classification: weights & thresholds ────────────────
# Step 3 — fuses the three throwing-motion signals into one score.
# These numbers are EMPIRICALLY CHOSEN starting points (not fitted to a
# labelled dataset) and are deliberately exposed as module constants so
# they're easy to re-tune per-venue if real-world false positives or
# negatives show up:
#   ARM_RAISE_WEIGHT / SPEED_WEIGHT / MOTION_WEIGHT — how much each of
#     the three normalised 0-1 signals contributes to intent_score.
#     They sum to 1.0 so intent_score itself stays in 0-1.
#   PEAK_SPEED_NORM_MAX / FORWARD_MOTION_NORM_MAX — the raw (normalised
#     image-coordinate) signal value treated as "maximum" when scaling
#     compute_peak_wrist_speed()/compute_forward_motion() into 0-1
#     before weighting; values above this clamp at 1.0.
#   INTENT_SCORE_THRESHOLD — intent_score strictly above this is
#     classified "intentional", otherwise "accidental".
ARM_RAISE_WEIGHT = 0.40
SPEED_WEIGHT = 0.35
MOTION_WEIGHT = 0.25
PEAK_SPEED_NORM_MAX = 0.15
FORWARD_MOTION_NORM_MAX = 0.10
INTENT_SCORE_THRESHOLD = 0.55


class IntentAnalyzer:
    def __init__(self, buffer_size=DEFAULT_BUFFER_SIZE, model_asset_path=None):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)  # landmark history
        self.model_asset_path = Path(model_asset_path) if model_asset_path else DEFAULT_MODEL_ASSET_PATH

        self.available = _MEDIAPIPE_AVAILABLE
        self.landmarker = None
        self.status = "not_available"

        if not self.available:
            self.status = "mediapipe not installed"
            return

        if not self.model_asset_path.exists():
            self.available = False
            self.status = f"pose model asset missing: {self.model_asset_path}"
            return

        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_asset_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self.status = "ready"
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.landmarker = None
            self.status = f"error: {exc}"

    # ── Frame ingestion ────────────────────────────────────────
    def add_frame(self, frame):
        """Run pose detection on one BGR frame and, if a pose is
        detected, append shoulder/elbow/wrist landmarks (normalised
        x, y, visibility) to the rolling buffer. Frames where no pose
        is detected are skipped entirely (not appended). Returns True
        if a landmark set was added, False otherwise."""
        if not self.available or self.landmarker is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
        except Exception:  # noqa: BLE001
            return False

        if not result.pose_landmarks:
            return False

        lm = result.pose_landmarks[0]  # single pose (num_poses=1)
        entry = {
            "left_shoulder": (lm[LEFT_SHOULDER].x, lm[LEFT_SHOULDER].y, lm[LEFT_SHOULDER].visibility),
            "right_shoulder": (lm[RIGHT_SHOULDER].x, lm[RIGHT_SHOULDER].y, lm[RIGHT_SHOULDER].visibility),
            "left_elbow": (lm[LEFT_ELBOW].x, lm[LEFT_ELBOW].y, lm[LEFT_ELBOW].visibility),
            "right_elbow": (lm[RIGHT_ELBOW].x, lm[RIGHT_ELBOW].y, lm[RIGHT_ELBOW].visibility),
            "left_wrist": (lm[LEFT_WRIST].x, lm[LEFT_WRIST].y, lm[LEFT_WRIST].visibility),
            "right_wrist": (lm[RIGHT_WRIST].x, lm[RIGHT_WRIST].y, lm[RIGHT_WRIST].visibility),
        }
        self.buffer.append(entry)
        return True

    # ── Metrics ──────────────────────────────────────────────
    def compute_arm_raise_score(self):
        """Fraction (0.0-1.0) of buffered frames where either wrist's y
        is less than its corresponding shoulder's y — i.e. the wrist is
        higher on screen than the shoulder (arm raised)."""
        if len(self.buffer) < 2:
            return 0.0
        raised = 0
        for entry in self.buffer:
            left_raised = entry["left_wrist"][1] < entry["left_shoulder"][1]
            right_raised = entry["right_wrist"][1] < entry["right_shoulder"][1]
            if left_raised or right_raised:
                raised += 1
        return raised / len(self.buffer)

    def compute_peak_wrist_speed(self):
        """Maximum Euclidean distance (normalised coordinates) moved by
        either wrist between any two consecutive buffered frames."""
        if len(self.buffer) < 2:
            return 0.0
        entries = list(self.buffer)
        peak = 0.0
        for prev, curr in zip(entries, entries[1:]):
            for wrist_key in ("left_wrist", "right_wrist"):
                dx = curr[wrist_key][0] - prev[wrist_key][0]
                dy = curr[wrist_key][1] - prev[wrist_key][1]
                dist = math.hypot(dx, dy)
                if dist > peak:
                    peak = dist
        return peak

    def compute_forward_motion(self):
        """Maximum absolute horizontal (x) displacement of either wrist
        between any two consecutive buffered frames."""
        if len(self.buffer) < 2:
            return 0.0
        entries = list(self.buffer)
        peak = 0.0
        for prev, curr in zip(entries, entries[1:]):
            for wrist_key in ("left_wrist", "right_wrist"):
                dx = abs(curr[wrist_key][0] - prev[wrist_key][0])
                if dx > peak:
                    peak = dx
        return peak

    # ── Intent classification ───────────────────────────────────
    def classify_intent(self):
        """Fuses arm-raise, peak wrist speed, and forward wrist motion
        into a single intent score and classification. See the module-
        level comment above for the (empirically chosen, configurable)
        weights and thresholds.

        If MediaPipe is unavailable, or no pose has been observed yet,
        this safely defaults to "accidental" with 0 confidence instead
        of raising — compute_arm_raise_score()/compute_peak_wrist_speed()/
        compute_forward_motion() already return 0.0 in that situation,
        which naturally yields intent_score 0.0 (not > threshold), but
        the mediapipe-unavailable case is also checked explicitly below
        for clarity."""
        arm_raise = self.compute_arm_raise_score()
        peak_speed = self.compute_peak_wrist_speed()
        forward_motion = self.compute_forward_motion()
        signals = {
            "arm_raise": round(arm_raise, 3),
            "peak_speed": round(peak_speed, 3),
            "forward_motion": round(forward_motion, 3),
        }

        if not self.available:
            return {"intent": "accidental", "intent_score": 0.0, "confidence": 0, "signals": signals}

        norm_speed = min(peak_speed / PEAK_SPEED_NORM_MAX, 1.0)
        norm_motion = min(forward_motion / FORWARD_MOTION_NORM_MAX, 1.0)

        intent_score = round(
            (ARM_RAISE_WEIGHT * arm_raise) + (SPEED_WEIGHT * norm_speed) + (MOTION_WEIGHT * norm_motion),
            3,
        )
        intent = "intentional" if intent_score > INTENT_SCORE_THRESHOLD else "accidental"
        confidence = round(intent_score * 100)  # percentage, 0-100

        return {"intent": intent, "intent_score": intent_score, "confidence": confidence, "signals": signals}

    # ── Reset / cleanup ─────────────────────────────────────────
    def reset(self):
        """Clear the landmark history buffer."""
        self.buffer.clear()

    def close(self):
        """Release the underlying PoseLandmarker's native resources.
        Not part of the original spec's 4 compute methods, but required
        by the Tasks API (unlike the old Solutions API, these objects
        hold native handles); safe to call multiple times or never."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self.landmarker = None
