"""
Component 4 — Intent analysis v2: was the garbage THROWN or DROPPED by accident?

WHY THE OLD VERSION ALWAYS SAID "accidental"
---------------------------------------------
1. It only started reading body pose AFTER garbage was already floating in
   the pool. By then the throw is over, the arm is down, so arm-raise and
   wrist speed are ~0 -> score below 0.55 -> "accidental" every time.
2. standalone_demo.py never passed a person model (source_model_path), so
   the source was always "unknown" and pose analysis never even ran.
3. Pose was run on the whole frame with num_poses=1, so with several people
   it could measure the wrong person.
4. Speeds were "per frame" in whole-image units, so the same throw gave a
   different number at 10 FPS and 30 FPS, and near vs far from the camera.

WHAT THIS VERSION DOES
----------------------
* PoseHistory runs pose on EVERY tracked person, ALL the time (not only
  after garbage appears), on a crop of that person, and keeps the last few
  seconds of shoulder/elbow/wrist/hip points with timestamps.
* When a garbage object first appears, we LOOK BACK in time (~2.5 s) at the
  nearest person's arm movement, and also at how the OBJECT itself started
  moving (launched sideways/upwards fast = thrown; slipped straight down
  = dropped).
* All speeds are in body-heights per second, so they don't depend on the
  camera distance or the FPS.
* If you record labelled clips and train tools/train_intent_classifier.py,
  the trained model (models/intent_clf.joblib) is used automatically
  instead of the hand-set rule. Without it, the rule below is used.

Landmark indices (MediaPipe 33-point pose): 11/12 shoulders, 13/14 elbows,
15/16 wrists, 23/24 hips.
"""

import math
from collections import deque

import os

import cv2

from component4_garbage import config

# MediaPipe is imported LAZILY (inside PoseHistory), never at module import.
# On some macOS machines in the team, importing / starting MediaPipe crashed
# the whole dashboard backend. Kill switch: set DISABLE_INTENT_ANALYSIS=1 and
# MediaPipe is never imported; garbage detection keeps working without intent.
mp = mp_vision = BaseOptions = None


def _intent_disabled():
    return os.environ.get("DISABLE_INTENT_ANALYSIS", "0") == "1"


def _load_mediapipe():
    """Returns None on success, or a short reason string on failure."""
    global mp, mp_vision, BaseOptions
    if mp is not None:
        return None
    try:
        import mediapipe as _mp
        from mediapipe.tasks.python import vision as _vision
        from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions
    except BaseException as exc:  # noqa: BLE001 - includes native init failures
        return f"mediapipe unavailable ({type(exc).__name__})"
    mp, mp_vision, BaseOptions = _mp, _vision, _BaseOptions
    return None

KP = {"l_sh": 11, "r_sh": 12, "l_el": 13, "r_el": 14,
      "l_wr": 15, "r_wr": 16, "l_hip": 23, "r_hip": 24}

# Order matters: the trained classifier expects exactly these columns.
FEATURE_ORDER = [
    "wrist_peak_speed", "wrist_mean_speed", "arm_raise", "arm_extension_max",
    "arm_extension_delta", "wrist_above_hip", "obj_launch_speed",
    "obj_horizontal_ratio", "obj_upward", "obj_origin_to_wrist", "has_pose", "has_object_path",
]

NO_INTENT_SIGNAL = {
    "intent": "unknown", "intent_score": 0.0, "confidence": 0,
    "signals": {k: 0.0 for k in FEATURE_ORDER}, "reason": "no person near the object",
}


def _plausible(kps, body_h):
    """Reject pose results that cannot be a real upper body (MediaPipe still
    returns landmarks for half-visible or tiny people; those jitter a lot and
    look like fast arm swings). Returns the torso length to use, or None.
    Wrists that are impossibly far from their shoulder are marked invisible."""
    l_sh, r_sh, l_hip, r_hip = kps["l_sh"], kps["r_sh"], kps["l_hip"], kps["r_hip"]
    if max(l_sh[2], r_sh[2]) < 0.5 or min(l_sh[2], r_sh[2]) < 0.2:
        return None       # (side-on throwers hide one shoulder: allow a weak one)
    shoulder_w = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
    torso = abs((l_hip[1] + r_hip[1]) / 2 - (l_sh[1] + r_sh[1]) / 2)
    if max(l_hip[2], r_hip[2]) < 0.3 or torso < 0.15 * body_h:
        # hips not in view (upper-body shot, e.g. standing close to the
        # camera): estimate the torso from the shoulder width instead
        torso = 1.4 * shoulder_w
        if torso < 0.1 * body_h:
            return None
    for side in ("l", "r"):
        sh, wr = kps[f"{side}_sh"], kps[f"{side}_wr"]
        if math.hypot(wr[0] - sh[0], wr[1] - sh[1]) > 1.8 * torso:
            kps[f"{side}_wr"] = (wr[0], wr[1], 0.0)
    return torso


def _clamp01(x):
    return max(0.0, min(1.0, x))


class PoseHistory:
    """Runs MediaPipe pose on person crops and remembers recent keypoints."""

    def __init__(self, model_asset_path=None, history_sec=None):
        self.history_sec = history_sec or config.POSE_HISTORY_SEC
        self.model_asset_path = model_asset_path or config.POSE_MODEL_PATH
        self.people = {}            # person_track_id -> deque of entries
        self.available = False
        self.status = "not_available"
        self.landmarker = None
        if _intent_disabled():
            self.status = "disabled (DISABLE_INTENT_ANALYSIS=1)"
            print("Intent analysis disabled - DISABLE_INTENT_ANALYSIS=1")
            return
        if not self.model_asset_path.exists():
            self.status = f"pose model missing: {self.model_asset_path.name}"
            print(f"Intent analysis disabled - pose model not found at {self.model_asset_path}")
            return
        reason = _load_mediapipe()
        if reason:
            self.status = reason
            print("Intent analysis disabled - MediaPipe unavailable on this platform")
            return
        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_asset_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self.available = True
            self.status = "ready"
        except BaseException as exc:  # noqa: BLE001
            self.status = f"error: {type(exc).__name__}"
            print("Intent analysis disabled - MediaPipe failed to start:", exc)

    def update(self, frame, person_tracks, ts):
        """person_tracks: tracker Track objects for people seen this frame."""
        if not self.available:
            return
        h, w = frame.shape[:2]
        # Largest (closest) people first; cap the number for speed.
        people = sorted(person_tracks,
                        key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]),
                        reverse=True)[: config.MAX_POSE_PEOPLE]
        for tr in people:
            x1, y1, x2, y2 = tr.box
            bw, bh = x2 - x1, y2 - y1
            if bh < 40:
                continue
            # Margin so raised arms are not cut off.
            mx, my = int(bw * 0.35), int(bh * 0.15)
            cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
            cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.copy())
                result = self.landmarker.detect(image)
            except Exception:  # noqa: BLE001
                continue
            if not result.pose_landmarks:
                continue
            lm = result.pose_landmarks[0]
            ch, cw = crop.shape[:2]
            kps = {}
            for name, idx in KP.items():
                p = lm[idx]
                kps[name] = (cx1 + p.x * cw, cy1 + p.y * ch, p.visibility)
            torso = _plausible(kps, bh)
            if torso is None:
                continue          # cut-off / far person: landmarks are guesses
            entry = {"ts": ts, "box": tr.box, "body_h": float(bh), "kp": kps,
                     "torso": float(torso)}
            self.people.setdefault(tr.id, deque()).append(entry)

        # Forget old entries / people.
        for pid in list(self.people):
            dq = self.people[pid]
            while dq and ts - dq[0]["ts"] > self.history_sec:
                dq.popleft()
            if not dq:
                del self.people[pid]

    def window(self, person_id, t0, t1):
        return [e for e in self.people.get(person_id, ()) if t0 <= e["ts"] <= t1]

    def latest(self, person_id):
        dq = self.people.get(person_id)
        return dq[-1] if dq else None

    def close(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self.landmarker = None


class IntentAnalyzer:
    """Turns a pose window + object path into intentional / accidental."""

    def __init__(self, clf_path=None):
        self.clf = None
        self.mode = "rule"
        path = clf_path or config.INTENT_CLF_PATH
        if path.exists():
            try:
                import joblib
                self.clf = joblib.load(path)
                self.mode = "trained"
            except Exception:  # noqa: BLE001
                self.clf = None

    # ── features ──────────────────────────────────────────────
    @staticmethod
    def pose_features(entries):
        f = {"wrist_peak_speed": 0.0, "wrist_mean_speed": 0.0, "arm_raise": 0.0,
             "arm_extension_max": 0.0, "arm_extension_delta": 0.0,
             "wrist_above_hip": 0.0, "has_pose": 0.0}
        if len(entries) < 2:
            return f
        f["has_pose"] = 1.0
        speeds, raised, ext, above = [], 0, [], 0
        for prev, cur in zip(entries, entries[1:]):
            dt = cur["ts"] - prev["ts"]
            if dt <= 0 or dt > 0.6:
                continue
            body = max(cur["body_h"], 1.0)
            for wr in ("l_wr", "r_wr"):
                (px, py, pv), (cx, cy, cv) = prev["kp"][wr], cur["kp"][wr]
                if pv < 0.3 or cv < 0.3:
                    continue
                v = math.hypot(cx - px, cy - py) / body / dt
                if v <= 15.0:     # faster than any human arm = a landmark jump
                    speeds.append(v)
        for e in entries:
            kp = e["kp"]
            torso = e.get("torso") or max(1.0, abs(((kp["l_hip"][1] + kp["r_hip"][1]) / 2)
                                                - ((kp["l_sh"][1] + kp["r_sh"][1]) / 2)))
            best_ext = None
            for side in ("l", "r"):
                sh, wr, hip = kp[f"{side}_sh"], kp[f"{side}_wr"], kp[f"{side}_hip"]
                if wr[2] < 0.3:
                    continue
                if wr[1] < sh[1]:
                    raised += 1
                if wr[1] < hip[1] - 0.15 * torso:
                    above += 1
                d = math.hypot(wr[0] - sh[0], wr[1] - sh[1]) / torso
                best_ext = d if best_ext is None else max(best_ext, d)
            if best_ext is not None:      # no visible wrist = no measurement (not 0)
                ext.append(best_ext)
        n = len(entries)
        if speeds:
            f["wrist_peak_speed"] = max(speeds)
            f["wrist_mean_speed"] = sum(speeds) / len(speeds)
        f["arm_raise"] = min(1.0, raised / n)
        f["wrist_above_hip"] = min(1.0, above / n)
        if ext:
            f["arm_extension_max"] = max(ext)
            f["arm_extension_delta"] = max(ext) - min(ext)
        return f

    @staticmethod
    def object_features(points, frame_h, wrist_points=None):
        """points: [(ts, cx, cy)] of the object from its FIRST sighting.
        wrist_points: [(ts, x, y, body_h)] of the person's wrists."""
        f = {"obj_launch_speed": 0.0, "obj_horizontal_ratio": 0.0, "obj_upward": 0.0,
             "obj_origin_to_wrist": 9.0, "has_object_path": 0.0}
        pts = list(points)
        if len(pts) >= 2:
            t0 = pts[0][0]
            early = [p for p in pts if p[0] - t0 <= 0.6] or pts[:3]
            if len(early) >= 2 and early[-1][0] > early[0][0]:
                dt = early[-1][0] - early[0][0]
                dx = (early[-1][1] - early[0][1]) / frame_h
                dy = (early[-1][2] - early[0][2]) / frame_h
                f["obj_launch_speed"] = math.hypot(dx, dy) / dt
                f["obj_horizontal_ratio"] = abs(dx) / (abs(dx) + abs(dy) + 1e-6)
                min_y = min(p[2] for p in early)
                f["obj_upward"] = 1.0 if (early[0][2] - min_y) / frame_h > 0.02 else 0.0
                f["has_object_path"] = 1.0
        if pts and wrist_points:
            t0, ox, oy = pts[0]
            near = [w for w in wrist_points if abs(w[0] - t0) < 1.0]
            if near:
                f["obj_origin_to_wrist"] = min(
                    math.hypot(ox - wx, oy - wy) / max(bh, 1.0) for _, wx, wy, bh in near)
        return f

    # ── decision ──────────────────────────────────────────────
    def classify(self, pose_entries, object_points, frame_h):
        wrists = []
        for e in pose_entries:
            for wr in ("l_wr", "r_wr"):
                x, y, v = e["kp"][wr]
                if v >= 0.3:
                    wrists.append((e["ts"], x, y, e["body_h"]))
        feats = {}
        feats.update(self.pose_features(pose_entries))
        feats.update(self.object_features(object_points, frame_h, wrists))
        signals = {k: round(float(feats[k]), 3) for k in FEATURE_ORDER}

        if not feats["has_pose"] and not feats["has_object_path"]:
            out = dict(NO_INTENT_SIGNAL)
            out["signals"] = signals
            out["reason"] = "no pose and no object path in the look-back window"
            return out

        if self.clf is not None:
            x = [[feats[k] for k in FEATURE_ORDER]]
            proba = self.clf.predict_proba(x)[0]
            classes = list(self.clf.classes_)
            p_int = float(proba[classes.index("intentional")]) if "intentional" in classes else 0.0
            intent = "intentional" if p_int >= 0.5 else "accidental"
            conf = p_int if intent == "intentional" else 1 - p_int
            return {"intent": intent, "intent_score": round(p_int, 3),
                    "confidence": round(conf * 100), "signals": signals,
                    "reason": "trained classifier"}

        # Rule-based fusion (no training data needed).
        speed_n = _clamp01((feats["wrist_peak_speed"] - config.WRIST_SPEED_DROP)
                           / (config.WRIST_SPEED_THROW - config.WRIST_SPEED_DROP))
        swing_n = _clamp01(feats["arm_extension_delta"] / 0.8)
        raise_n = max(feats["arm_raise"], 0.5 * feats["wrist_above_hip"])
        launch_n = _clamp01(feats["obj_launch_speed"] / config.OBJECT_LAUNCH_SPEED)
        # A dropped object falls straight down; a thrown one moves sideways/up.
        direction_n = max(feats["obj_horizontal_ratio"], feats["obj_upward"])
        obj_n = launch_n * (0.4 + 0.6 * direction_n)

        parts, weights = [], []
        if feats["has_pose"]:
            parts += [speed_n, swing_n, raise_n]
            weights += [0.35, 0.15, 0.15]
        if feats["has_object_path"]:
            parts.append(obj_n)
            weights.append(0.35)
        score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)

        intent = "intentional" if score > config.INTENT_SCORE_THRESHOLD else "accidental"
        margin = abs(score - config.INTENT_SCORE_THRESHOLD) / max(
            config.INTENT_SCORE_THRESHOLD, 1 - config.INTENT_SCORE_THRESHOLD)
        reasons = []
        if speed_n > 0.6:
            reasons.append("fast arm swing")
        if raise_n > 0.4:
            reasons.append("arm raised")
        if obj_n > 0.6:
            reasons.append("object launched sideways/up")
        if feats["has_object_path"] and direction_n < 0.3 and launch_n < 0.6:
            reasons.append("object fell straight down")
        if feats["has_pose"] and speed_n < 0.2:
            reasons.append("arm almost still")
        return {"intent": intent, "intent_score": round(score, 3),
                "confidence": round(50 + 50 * _clamp01(margin)), "signals": signals,
                "reason": ", ".join(reasons) or "weak signals"}

    def throw_motion_score(self, pose_entries):
        """Pose-only score, used to catch a throw even when no floating
        object appears afterwards (e.g. a heavy object that sank)."""
        f = self.pose_features(pose_entries)
        if not f["has_pose"]:
            return 0.0
        speed_n = _clamp01((f["wrist_peak_speed"] - config.WRIST_SPEED_DROP)
                           / (config.WRIST_SPEED_THROW - config.WRIST_SPEED_DROP))
        swing_n = _clamp01(f["arm_extension_delta"] / 0.8)
        return 0.6 * speed_n + 0.25 * swing_n + 0.15 * f["arm_raise"]
