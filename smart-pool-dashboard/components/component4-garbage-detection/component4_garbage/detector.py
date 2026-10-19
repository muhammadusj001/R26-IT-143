"""
Component 4 — Intent-aware Garbage Intrusion Detection (v2, LIVE camera).

Public API is unchanged, so the integrated dashboard keeps working:

    det = GarbageDetector(model_path=..., source_model_path=...)
    det.load()
    frame, result, alert = det.process(frame)      # once per live frame

What happens inside process(frame) for every live frame:

  1. Garbage model (your YOLOv8, 6 classes) finds floating objects.
  2. COCO model (yolov8n.pt) finds people, animals, and hand-held / flying
     bottles & cups (the garbage model never saw objects in hands or air).
  3. A tracker gives every object and person an ID, so one bottle = one
     alert, and we know each object's path over time.
  4. Pose (MediaPipe) runs continuously on people, keeping ~4 s of arm
     history, so when garbage appears we can look BACK at the throw.
  5. Intent = arm movement before the object appeared + how the object
     started moving -> intentional / accidental (+ reason).
  6. Sinking objects: vanished-inside-pool + splash, pool-floor change, and
     throw-with-no-floating-object rules (see submersion.py).
  7. Events are logged to events/events.jsonl with a snapshot image.
"""

import math
import os
import random
import time
from collections import deque
from pathlib import Path

from component4_garbage import config
from component4_garbage import enhance as enh
from component4_garbage.drawing import draw_box, draw_hud
from component4_garbage.events import EventLogger
from component4_garbage.intent_analyzer import (NO_INTENT_SIGNAL, IntentAnalyzer,
                                                PoseHistory)
from component4_garbage.submersion import FloorChangeMonitor, PoolROI, SplashDetector
from component4_garbage.tracker import SimpleTracker, center

ALL_CLASSES = config.ALL_CLASSES
GARBAGE_CLASSES = config.GARBAGE_CLASSES

GARBAGE_COLOR = (0, 255, 255)
NON_GARBAGE_COLOR = (200, 200, 200)
PERSON_COLOR = (255, 0, 255)
LITTER_COLOR = (0, 165, 255)
SUNK_COLOR = (0, 0, 255)
UNDERWATER_COLOR = (255, 120, 0)     # blue-ish (BGR): garbage under the surface
THROW_COLOR = (0, 100, 255)          # orange-red: throwing action
SIM_OBJECTS = ["bottle", "aluminium foil", "juice", "thermocol"]

RISK_ORDER = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _dist_point_box(px, py, box):
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


class GarbageDetector:
    def __init__(self, model_path=None, conf_threshold=None, imgsz=None,
                 source_model_path=None, roi_path=None, log_events=True,
                 show_hud=None):
        # show_hud: status panel + people/pose/hand-held overlays. Off by
        # default so the integrated dashboard frame stays clean (Component 1
        # already draws people); standalone_demo.py turns it on.
        self.show_hud = config.SHOW_HUD if show_hud is None else show_hud
        self.model_path = Path(model_path) if model_path else config.GARBAGE_MODEL_PATH
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.CONF
        # The dashboard passes DETECTION_IMGSZ (640) to every detector. Floating
        # garbage is small, so Component 4 never goes below 640 unless C4_IMGSZ
        # is set explicitly.
        if os.environ.get("C4_IMGSZ"):
            self.imgsz = config.IMGSZ
        else:
            self.imgsz = max(int(imgsz or config.IMGSZ), 640)
        self.source_model_path = (Path(source_model_path) if source_model_path
                                  else config.SOURCE_MODEL_PATH)
        self.model = None
        self.source_model = None
        self.model_status = "not_loaded"
        self.source_model_status = "not_loaded"
        self.source_identification_enabled = False
        self.ov_model = None
        self.ov_status = "off"
        self._ov_cache = []
        self.total_events = 0
        self._sim_cooldown = 0
        self._frame_idx = 0
        self._last_gdets = []
        self._weak_boxes = []
        self._simulating = False
        self._g_runs = 0
        self._tile_cache = []                      # tiny-object detections (tiles)
        self._uw_cache = []                        # [(det, contrast)] from enhanced pass
        self._throw_history = deque(maxlen=50)     # human action: throwing motions
        self._actions = {}                         # person id -> (action, score)
        self._action_tags = []                     # (until_ts, box, text)
        self._feed = deque(maxlen=20)              # plain-language events for the UI

        self.roi = PoolROI(Path(roi_path) if roi_path else None)
        self.g_tracker = SimpleTracker(confirm_frames=config.CONFIRM_FRAMES,
                                       confirm_window=config.CONFIRM_WINDOW,
                                       max_age_sec=config.LOST_TRACK_SEC)
        self.p_tracker = SimpleTracker(confirm_frames=2, confirm_window=4, max_age_sec=1.0,
                                       dist_ratio=0.12)
        self.l_tracker = SimpleTracker(confirm_frames=2, confirm_window=5, max_age_sec=0.8,
                                       iou_thr=0.1, dist_ratio=0.15)
        self.pose = PoseHistory()
        self.intent_analyzer = IntentAnalyzer()
        self.splash = SplashDetector()
        self.floor = FloorChangeMonitor(self.roi) if config.FLOOR_CHANGE_ENABLED else None
        self.logger = EventLogger() if log_events else None

        self._recent_people = deque(maxlen=200)    # ended person tracks (for look-back)
        self._recent_litter = deque(maxlen=200)    # ended COCO litter tracks
        self._pending_vanish = []                  # wait for splash window
        self._pending_throws = []                  # throw motion, waiting for object
        self._person_throw_cooldown = {}
        self._pending_throw_cooldown = {}
        self._markers = []                         # (until_ts, x, y, text)
        self._recent_events = deque(maxlen=6)
        self._last_info = None                     # sticky info for dashboard
        self._fps = deque(maxlen=20)

    # ── loading ─────────────────────────────────────────────────
    def load(self):
        self._load_source_model()
        self._load_open_vocab()
        if not self.model_path.exists():
            for alt in config.GARBAGE_MODEL_FALLBACKS:
                if alt.exists():
                    print(f"Component 4: {self.model_path.name} not found, using {alt.name}")
                    self.model_path = alt
                    break
        if not self.model_path.exists():
            if self.ov_model is not None or self.source_model is not None:
                # Never invent fake garbage on a real camera: run on the
                # open-vocabulary + COCO detectors only.
                self.model_status = "missing (open-vocab + COCO only)"
                print("Component 4: garbage model missing in models/ - put "
                      "swimming_pool_garbage_yolo.pt back for best accuracy")
                return False
            self._simulating = True
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

    def _load_open_vocab(self):
        if not config.OPEN_VOCAB:
            self.ov_status = "off (C4_OPEN_VOCAB=0)"
            return
        if not config.OPEN_VOCAB_MODEL_PATH.exists():
            self.ov_status = "model missing"
            return
        try:
            from ultralytics import YOLO
            self.ov_model = YOLO(str(config.OPEN_VOCAB_MODEL_PATH))
            self.ov_status = "loaded"
        except Exception as exc:  # noqa: BLE001
            self.ov_model = None
            self.ov_status = f"error: {type(exc).__name__}"

    def _load_source_model(self):
        try:
            from ultralytics import YOLO
            # If models/yolov8n.pt is missing, Ultralytics downloads it once.
            path = (str(self.source_model_path) if self.source_model_path.exists()
                    else self.source_model_path.name)
            self.source_model = YOLO(path)
            self.source_model_status = "loaded"
            self.source_identification_enabled = True
        except Exception as exc:  # noqa: BLE001
            self.source_model_status = f"error: {exc}"
            self.source_identification_enabled = False

    def status_summary(self):
        """One line for the backend's start-up banner."""
        return (f"garbage={self.model_status} open_vocab={self.ov_status} "
                f"people={self.source_model_status} "
                f"pose={self.pose.status} intent={self.intent_analyzer.mode} "
                f"pool_area={'set' if self.roi.defined else 'not set'} "
                f"tiny_tiles={'%dx%d' % (config.TILES, config.TILES) if config.TILES > 1 else 'off'} "
                f"underwater={'on' if config.UNDERWATER else 'off'}")

    # ── per-frame ───────────────────────────────────────────────
    def process(self, frame, ts=None, clean_frame=None):
        """Returns (annotated_frame, result_dict, alert_message_or_None).

        clean_frame: in the integrated dashboard, Components 1 and 3 draw
        boxes on `frame` before it reaches us. Pass an untouched copy here
        (raw = frame.copy() right after camera.read()) so detection, pose,
        splash and floor analysis see the real image; drawing still goes
        onto `frame`."""
        t_start = time.time()
        src = clean_frame if clean_frame is not None else frame
        ts = ts if ts is not None else t_start
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        self._frame_idx += 1
        if self._frame_idx == 1:
            warn = self.roi.check_size(w, h)
            if warn:
                print("Component 4:", warn)

        # 0) scene cut / camera moved: an edited video or a bumped camera makes
        #    every object "jump" or "vanish", which would fake throws and sunk
        #    objects. Start fresh instead.
        if self._scene_cut(src):
            self._reset_scene()
            self._scene_start = ts
        if getattr(self, "_scene_start", None) is None:
            self._scene_start = ts

        # 1) garbage model (every Nth frame)
        if self.model is None:
            gdets = self._simulate(w, h) if self._simulating else []
            run_garbage = True
        else:
            run_garbage = (self._frame_idx % max(1, config.GARBAGE_EVERY_N)) == 0 \
                or self._frame_idx == 1
            gdets = self._detect_garbage(src) if run_garbage else self._last_gdets
        # 2) people / animals / hand-held litter
        persons, animals, litter = self._detect_sources(src)
        # 2b) second opinion: the COCO model also knows bottles and cups from
        #     millions of everyday photos. A COCO bottle/cup that is NOT in a
        #     person's hand and that the garbage model missed is added as
        #     garbage (fixes close-up / unusual-angle bottles).
        if not self._simulating and run_garbage and config.COCO_FALLBACK:
            gdets = gdets + self._coco_fallback(litter, persons, gdets, w, h)
        # 2c) open-vocabulary detector: paper, plastic bags, trays, cans, close-ups
        if self.ov_model is not None:
            fresh = (self._frame_idx % max(1, config.OPEN_VOCAB_EVERY)) == 1 \
                or config.OPEN_VOCAB_EVERY <= 1
            if fresh:
                self._ov_cache = self._detect_open_vocab(src)
            ov_garbage, ov_hand = self._split_open_vocab(self._ov_cache, persons, gdets, w, h)
            if run_garbage:
                gdets = gdets + ov_garbage
            if fresh:
                litter = litter + ov_hand
        # 2d) tiny garbage (zoomed tiles) + garbage UNDER the water surface
        uw_boxes = set()
        if run_garbage and not self._simulating:
            self._g_runs += 1
            gdets = gdets + enh.merge_new(gdets, self._small_objects(src, w, h))
            extra_uw, uw_boxes = self._underwater_objects(src, gdets, w, h)
            gdets = gdets + extra_uw
        if not self._simulating:
            self._last_gdets = gdets

        # 3) tracking
        if run_garbage:
            g_active, g_new, g_ended = self.g_tracker.update(gdets, ts, diag)
            for t in g_active:                    # surface / underwater history
                if t.last_seen == ts:
                    t.meta.setdefault("uw", deque(maxlen=8)).append(1 if t.box in uw_boxes else 0)
        else:
            g_active = [t for t in self.g_tracker.tracks if ts - t.last_seen < 0.5]
            g_new, g_ended = [], []
        p_active, _, p_ended = self.p_tracker.update(persons, ts, diag)
        l_active, _, l_ended = self.l_tracker.update(litter, ts, diag)
        self._recent_people.extend(p_ended)
        self._recent_litter.extend(l_ended)

        # 4) continuous pose + splash map
        self.pose.update(src, p_active, ts)
        self.splash.update(src, ts, ignore_boxes=[t.box for t in p_active])

        events = []
        # 5) newly confirmed garbage in the pool -> intent-aware event
        for tr in g_new:
            if tr.label not in GARBAGE_CLASSES:
                continue
            cx, cy = center(tr.box)
            if not self.roi.contains(cx, tr.box[3], w, h) and not self.roi.contains(cx, cy, w, h):
                tr.meta["outside_pool"] = True
                continue
            events.append(self._garbage_event(tr, ts, h, diag, animals))

        # 6) objects that vanished inside the pool -> wait for splash check
        for tr in g_ended:
            if tr.confirmed and tr.label in GARBAGE_CLASSES and not tr.meta.get("outside_pool"):
                self._queue_vanish(tr, "garbage", w, h, diag)
        for tr in l_ended:
            if len(tr.points) >= 3:
                self._queue_vanish(tr, "hand-held " + tr.label, w, h, diag)
        events += self._check_pending_vanish(ts, g_active, diag)
        events += self._check_went_under(g_active)

        # 7) throw motion from the deck with no floating object afterwards
        self._watch_throws(p_active, ts, w, h)
        events += self._check_pending_throws(ts, diag)

        # 8) pool-floor change (objects resting under water)
        if self.floor is not None:
            ignore = [t.box for t in p_active] + [t.box for t in g_active]
            for ev in self.floor.update(src, ts, ignore):
                x1, y1, x2, y2 = ev["box"]
                events.append(self._make_event(
                    "UNDERWATER_OBJECT", "MEDIUM",
                    f"New object resting on the pool floor (stable {ev['age']} s)",
                    location=((x1 + x2) // 2, (y1 + y2) // 2), box=ev["box"]))

        # 9) drawing
        self._draw(frame, g_active, p_active, l_active, ts, w, h)
        for ev in events:
            self.total_events += 1
            self._recent_events.appendleft((ts, ev))
            if ev.get("location") and ev["type"] not in ("GARBAGE_IN_POOL", "SUBMERGED_GARBAGE"):
                self._markers.append((ts + 15, *ev["location"], ev["type"]))
            logged = self.logger.log(ev, frame) if self.logger is not None else ev
            self._feed.appendleft(self._feed_item(ev, logged.get("snapshot")))

        # 10) result for the dashboard (same keys as v1 + new ones)
        pool_garbage = [t for t in g_active if t.label in GARBAGE_CLASSES
                        and t.confirmed and not t.meta.get("outside_pool")]
        underwater = [t for t in pool_garbage if self._is_underwater(t)]
        human_action = self._action_summary(p_active, ts)
        others = [t.label for t in g_active if t.label not in GARBAGE_CLASSES]
        if events:
            top = max(events, key=lambda e: RISK_ORDER.get(e["risk_level"], 0))
            self._last_info = (ts, top)
        info = self._last_info[1] if self._last_info and ts - self._last_info[0] < 15 else None
        intent_info = info.get("intent_info", NO_INTENT_SIGNAL) if info else NO_INTENT_SIGNAL
        risk = info["risk_level"] if info else ("MEDIUM" if pool_garbage else "NORMAL")
        alert_active = bool(pool_garbage) or info is not None

        self._fps.append(time.time() - t_start)
        fps = len(self._fps) / max(sum(self._fps), 1e-6)
        if self.show_hud:
            draw_hud(frame, [
                f"Garbage model: {self.model_status} | Open-vocab: {self.ov_status} | "
                f"People: {self.source_model_status}",
                f"Pose: {self.pose.status} | Intent: {self.intent_analyzer.mode} | "
                f"ROI: {'set' if self.roi.defined else 'whole frame'}",
                f"FPS {fps:.1f} | surface: {len(pool_garbage) - len(underwater)} | "
                f"underwater: {len(underwater)} | people: {len(p_active)} | "
                f"incidents: {self.total_events}",
                f"Human action: {human_action}",
            ], [f"{e['type']}: {e['message'][:70]}" for _, e in list(self._recent_events)[:3]
                if ts - _ < 20])

        result = {
            "objects_detected": len(g_active),
            "object_labels": [t.label for t in pool_garbage],
            "non_garbage_labels": others,
            "alert_status": "ALERT" if alert_active else "CLEAR",
            "total_events": self.total_events,
            "model_status": self.model_status,
            "source": info.get("source", "unknown") if info else "unknown",
            "source_distance": info.get("source_distance") if info else None,
            "intent": intent_info["intent"],
            "intent_score": intent_info["intent_score"],
            "intent_confidence": intent_info["confidence"],
            "intent_reason": intent_info.get("reason", ""),
            "risk_level": risk,
            "signals": intent_info["signals"],
            "events": [{k: v for k, v in e.items() if k != "intent_info"} for e in events],
            "people_count": len(p_active),
            "intent_status": self.pose.status,
            "people_model_status": self.source_model_status,
            "fps": round(fps, 1),
            # ── v2.3: panel feedback ──
            "surface_count": len(pool_garbage) - len(underwater),
            "underwater_count": len(underwater),
            "underwater_labels": [t.label for t in underwater],
            "human_action": human_action,
            "actions": [{"person_id": pid, "action": a, "score": round(sc, 2)}
                        for pid, (a, sc) in self._actions.items()],
            "last_throw_sec_ago": (round(ts - self._throw_history[-1]["t"], 1)
                                   if self._throw_history else None),
            "event_feed": list(self._feed)[:10],
            "event_legend": {k: v[0] for k, v in config.EVENT_INFO.items()},
        }
        alert = None
        if events:
            top = max(events, key=lambda e: RISK_ORDER.get(e["risk_level"], 0))
            alert = f"[{top['risk_level']}] {top['message']}"
        return frame, result, alert

    # ── detection helpers ──────────────────────────────────────
    def _detect_garbage(self, frame):
        """Detections >= conf. Weak ones (>= 0.12) are remembered too: an object
        the normal model sees even weakly is NOT under the water surface."""
        out, weak = [], []
        results = self.model(frame, conf=min(self.conf_threshold, config.WEAK_CONF),
                             iou=config.IOU, imgsz=self.imgsz, verbose=False)
        for r in results:
            for box in r.boxes:
                name = self.model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                c = float(box.conf[0])
                weak.append((x1, y1, x2, y2))
                if c >= self.conf_threshold:
                    out.append(((x1, y1, x2, y2), name, c))
        self._weak_boxes = weak
        return out

    def _scene_cut(self, frame):
        import cv2
        import numpy as np
        small = cv2.resize(frame, (64, 36))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        prev, prev_gray = getattr(self, "_prev_hist", None), getattr(self, "_prev_gray", None)
        self._prev_hist, self._prev_gray = hist, gray
        if prev is None:
            return False
        mad = float(np.mean(np.abs(gray - prev_gray)))     # whole picture changed?
        return (mad > config.SCENE_CUT_MAD
                or cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < config.SCENE_CUT_CORR)

    def _reset_scene(self):
        for tr in (self.g_tracker, self.p_tracker, self.l_tracker):
            tr.tracks = []
        self._recent_people.clear()
        self._recent_litter.clear()
        self._pending_vanish = []
        self._pending_throws = []
        self._last_gdets = []
        self._tile_cache = []
        self._uw_cache = []
        self._action_tags = []
        self.pose.people.clear()
        self.splash.prev = None
        self.splash.history.clear()
        if self.floor is not None:
            self.floor.__init__(self.roi)
        self.scene_cuts = getattr(self, "scene_cuts", 0) + 1

    def _detect_open_vocab(self, frame):
        out = []
        try:
            results = self.ov_model.predict(frame, conf=config.OPEN_VOCAB_CONF,
                                            imgsz=config.OPEN_VOCAB_IMGSZ, verbose=False,
                                            retina_masks=False)
        except Exception:  # noqa: BLE001
            return out
        for r in results:
            for box in r.boxes:
                word = self.ov_model.names[int(box.cls[0])]
                label = config.OPEN_VOCAB_MAP.get(word)
                if label is None:
                    continue
                out.append((tuple(map(int, box.xyxy[0].tolist())), label, float(box.conf[0])))
        return out

    def _split_open_vocab(self, dets, persons, gdets, w, h):
        """Floating, in-pool objects -> garbage; objects at a person -> hand-held
        (used for throw trajectories). Duplicates of the trained model's boxes
        are dropped, so the trained model keeps priority."""
        from component4_garbage.tracker import iou
        garbage, hand = [], []
        for box, label, conf in dets:
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) > 0.25 * w * h:
                continue                                   # whole-scene false box
            cx, cy = center(box)
            dup = False
            for g in gdets:
                gx1, gy1, gx2, gy2 = g[0]
                if iou(box, g[0]) > 0.3 or (gx1 <= cx <= gx2 and gy1 <= cy <= gy2):
                    dup = True
                    break
            if dup:
                continue
            at_person = any(px1 <= cx <= px2 and py1 <= cy <= py2
                            for (px1, py1, px2, py2), _, _ in persons)
            if at_person:
                hand.append((box, label, conf))
            elif self.roi.contains(cx, cy, w, h):
                garbage.append((box, label, conf))
        return garbage, hand

    def _coco_fallback(self, litter, persons, gdets, w, h):
        from component4_garbage.tracker import iou
        extra = []
        for box, label, conf in litter:
            if label not in config.COCO_FALLBACK_LABELS or conf < config.COCO_FALLBACK_CONF:
                continue
            cx, cy = center(box)
            if not self.roi.contains(cx, cy, w, h):
                continue
            in_hand = False
            for pbox, _, _ in persons:
                px1, py1, px2, py2 = pbox
                pad = 0.15 * (px2 - px1)
                if px1 - pad <= cx <= px2 + pad and py1 <= cy <= py2:
                    in_hand = True
                    break
            if in_hand or any(iou(box, g[0]) > 0.3 for g in gdets):
                continue
            if not self._is_resting(box, h):
                continue          # still flying / being carried -> not floating garbage yet
            extra.append((box, config.COCO_FALLBACK_LABELS[label], conf))
        return extra

    def _is_resting(self, box, h):
        """True if a COCO litter track at this box has been seen for >= 1 s and
        barely moves (floating), so a bottle in mid-air is not counted twice."""
        from component4_garbage.tracker import iou
        for tr in self.l_tracker.tracks:
            if iou(tr.box, box) < 0.3 or len(tr.points) < 3:
                continue
            t0, x0, y0 = tr.points[0]
            t1, x1, y1 = tr.points[-1]
            if t1 - t0 < 1.0:
                return False
            recent = [p for p in tr.points if p[0] >= t1 - 1.0]
            ta, xa, ya = recent[0]
            dist = ((x1 - xa) ** 2 + (y1 - ya) ** 2) ** 0.5 / h
            return dist / max(t1 - ta, 1e-3) < 0.08
        return False

    def _detect_sources(self, frame):
        persons, animals, litter = [], [], []
        if not self.source_identification_enabled:
            return persons, animals, litter
        classes = [config.PERSON_CLASS] + list(config.ANIMAL_CLASSES) + list(
            config.COCO_LITTER_CLASSES)
        litter_conf = min(0.25, config.COCO_FALLBACK_CONF)
        results = self.source_model(frame, conf=min(config.SOURCE_CONF, litter_conf),
                                    classes=classes, imgsz=640, verbose=False)
        for r in results:
            for box in r.boxes:
                cid = int(box.cls[0])
                b = tuple(map(int, box.xyxy[0].tolist()))
                c = float(box.conf[0])
                if cid == config.PERSON_CLASS:
                    if c >= config.SOURCE_CONF:
                        persons.append((b, "person", c))
                elif cid in config.ANIMAL_CLASSES:
                    if c >= config.SOURCE_CONF:
                        animals.append((b, config.ANIMAL_CLASSES[cid], c))
                elif c >= litter_conf:
                    litter.append((b, config.COCO_LITTER_CLASSES[cid], c))
        return persons, animals, litter

    # ── events ──────────────────────────────────────────────────
    def _make_event(self, etype, risk, message, **extra):
        ev = {"type": etype, "risk_level": risk, "message": message,
              "component": "C4_garbage", "time_epoch": round(time.time(), 2)}
        ev.update(extra)
        return ev

    def _people_at(self, ts_query):
        """Person tracks (active + recently ended) with their box at ts_query."""
        out = []
        for tr in list(self.p_tracker.tracks) + list(self._recent_people):
            if tr.first_seen - 0.5 <= ts_query <= tr.last_seen + 0.5:
                b = tr.box_at(ts_query)
                if b:
                    out.append((tr, b))
        return out

    def _airborne_path(self, gtrack, h, diag):
        """Object path to judge the launch: a COCO hand-held/flying track that
        ended just before + near where the garbage appeared, then the garbage
        track itself. Returns [] if there is no sign the object was moving."""
        t0, gx, gy = gtrack.points[0]
        best = None
        for lt in list(self._recent_litter) + list(self.l_tracker.tracks):
            if not lt.points:
                continue
            lts, lx, ly = lt.points[-1]
            if t0 - 2.0 <= lts <= t0 + 0.5 and math.hypot(lx - gx, ly - gy) < 0.25 * diag:
                if best is None or lts > best.points[-1][0]:
                    best = lt
        path = []
        if best is not None:
            path = [p for p in best.points if p[0] >= t0 - 1.2]
        path += list(gtrack.points)[:10]
        if len(path) < 2:
            return []
        f = IntentAnalyzer.object_features(path, h)
        moving = f["obj_launch_speed"] > 0.3
        return path if (best is not None or moving) else []

    def _garbage_event(self, tr, ts, h, diag, animals):
        t_appear, ox, oy = tr.points[0]
        if t_appear - self._scene_start < config.ALREADY_PRESENT_SEC:
            # It was already floating when the camera view started (system
            # start, or after a scene cut): nobody can be blamed for it.
            info = dict(NO_INTENT_SIGNAL, intent="already_present",
                        reason="already in the pool when monitoring started")
            tr.alerted = True
            uw = self._is_underwater(tr)
            tr.meta["position"] = "underwater" if uw else "surface"
            return self._make_event(
                "SUBMERGED_GARBAGE" if uw else "GARBAGE_IN_POOL", "MEDIUM",
                f"{tr.label} {'UNDER WATER' if uw else 'in pool'} - already present "
                f"when monitoring started",
                object=tr.label, track_id=tr.id, source="unknown", source_distance=None,
                location=(int(ox), int(oy)), intent_info=info, intent=info["intent"],
                intent_confidence=0, position=tr.meta["position"])
        path = self._airborne_path(tr, h, diag)
        t_origin = path[0][0] if path else t_appear
        px, py = (path[0][1], path[0][2]) if path else (ox, oy)

        # nearest person (at the time the object appeared)
        source, src_dist, person = "unknown", None, None
        for ptr, pbox in self._people_at(t_origin):
            d = _dist_point_box(px, py, pbox)
            if d <= config.MAX_SOURCE_DISTANCE_RATIO * diag and (src_dist is None or d < src_dist):
                source, src_dist, person = "human", round(d, 1), ptr
        if person is None:
            for box, label, _ in animals:
                d = _dist_point_box(px, py, box)
                if d <= config.MAX_SOURCE_DISTANCE_RATIO * diag and (src_dist is None or d < src_dist):
                    source, src_dist = label, round(d, 1)

        intent_info = dict(NO_INTENT_SIGNAL)
        if person is not None:
            entries = self.pose.window(person.id, t_origin - config.LOOKBACK_SEC, t_appear + 0.3)
            intent_info = self.intent_analyzer.classify(entries, path, h)
        elif source in ("bird", "cat", "dog"):
            intent_info = dict(NO_INTENT_SIGNAL, intent="animal", reason=f"{source} nearby")
        else:
            intent_info = dict(NO_INTENT_SIGNAL, intent="environmental",
                               reason="no person or animal nearby (wind / drifted in)")

        # Human action link: a throwing motion seen shortly before, by the
        # nearest person or anyone close enough to have thrown it.
        throw = self._recent_throw(t_appear, px, py, diag, person)
        if throw is not None and intent_info["intent"] != "intentional":
            ago = max(0.0, t_appear - throw["t"])
            reason = intent_info.get("reason", "")
            reason = "" if reason.startswith(("no pose", "no person", "weak")) else reason
            intent_info = dict(intent_info, intent="intentional",
                               confidence=max(int(intent_info.get("confidence", 0)), 70),
                               reason=f"throwing motion seen {ago:.1f} s before"
                                      + (f"; {reason}" if reason else ""))
            if source != "human":
                source, src_dist = "human", None

        if source == "human":
            risk = "HIGH" if intent_info["intent"] == "intentional" else "MEDIUM"
        else:
            risk = "MEDIUM"
        tr.alerted = True
        tr.meta["event_logged"] = True
        uw = self._is_underwater(tr)
        tr.meta["position"] = "underwater" if uw else "surface"
        msg = (f"{tr.label} {'UNDER WATER' if uw else 'in pool'} - source: {source}, "
               f"intent: {intent_info['intent']} ({intent_info.get('reason', '')})")
        return self._make_event("SUBMERGED_GARBAGE" if uw else "GARBAGE_IN_POOL", risk, msg,
                                object=tr.label, track_id=tr.id, source=source,
                                source_distance=src_dist,
                                location=(int(ox), int(oy)), intent_info=intent_info,
                                intent=intent_info["intent"],
                                intent_confidence=intent_info["confidence"],
                                position=tr.meta["position"])

    def _queue_vanish(self, tr, what, w, h, diag):
        t_last, x, y = tr.points[-1]
        margin = 0.04
        if x < w * margin or x > w * (1 - margin) or y < h * margin or y > h * (1 - margin):
            return                                   # left the frame, not sunk
        if not self.roi.contains(x, y, w, h):
            return
        if not self.roi.defined and what != "garbage":
            # Without a pool area, hand-held objects vanish all the time
            # (put in a bag, hidden by the body). Only trust ROI setups.
            return
        if what != "garbage":
            path_len = sum(math.hypot(b[1] - a[1], b[2] - a[2])
                           for a, b in zip(tr.points, list(tr.points)[1:]))
            if path_len < 0.05 * diag:
                return
        # picked up? a person box covering the last position
        for ptr, pbox in self._people_at(t_last):
            if _dist_point_box(x, y, pbox) < 5:
                return
        self._pending_vanish.append({"t": t_last, "x": x, "y": y, "what": what,
                                     "label": tr.label, "track_id": tr.id})

    def _check_pending_vanish(self, ts, g_active, diag):
        out, keep = [], []
        for pv in self._pending_vanish:
            if ts - pv["t"] < config.SPLASH_WINDOW_SEC:
                keep.append(pv)
                continue
            # still floating nearby under a new id? then it did not sink.
            floating = any(math.hypot(center(t.box)[0] - pv["x"], center(t.box)[1] - pv["y"])
                           < 0.08 * diag for t in g_active)
            if floating:
                continue
            strength = self.splash.splash_strength(pv["x"], pv["y"], pv["t"] - 0.5,
                                                   pv["t"] + config.SPLASH_WINDOW_SEC)
            splash = strength >= config.SPLASH_PIXEL_RATIO
            risk = "HIGH" if splash else "MEDIUM"
            msg = (f"{pv['what']} vanished inside the pool"
                   f"{' with a splash' if splash else ''} - likely SUNK, check pool floor")
            out.append(self._make_event("SUNK_OBJECT", risk, msg, object=pv["label"],
                                        location=(int(pv["x"]), int(pv["y"])),
                                        splash_strength=round(strength, 3)))
        self._pending_vanish = keep
        return out

    def _watch_throws(self, p_active, ts, w, h):
        """Human action recognition, every frame, for every visible person:
        standing / arm moving / THROWING (from the pose arm-swing score), or
        swimming (person inside the pool area). Throwing motions are kept for
        THROW_LINK_SEC so garbage that is confirmed later is linked to them."""
        self._actions = {}
        for tr in p_active:
            feet_x, feet_y = (tr.box[0] + tr.box[2]) / 2, tr.box[3]
            swimmer = self.roi.defined and self.roi.contains(feet_x, feet_y, w, h)
            entries = self.pose.window(tr.id, ts - 1.5, ts)
            if swimmer:
                self._actions[tr.id] = ("swimming", 0.0)
                continue          # strokes look like throws, ignore
            if (tr.box[3] - tr.box[1]) < config.MIN_ACTION_BODY_RATIO * h:
                # Too small / far: pose keypoints jitter and look like fast arms.
                self._actions[tr.id] = ("person (too far for arm analysis)", 0.0)
                continue
            if len(entries) < 3:
                self._actions[tr.id] = ("person (arms not visible)" if self.pose.available
                                        else "person (pose off)", 0.0)
                continue
            score = self.intent_analyzer.throw_motion_score(entries)
            peak = IntentAnalyzer.pose_features(entries)["wrist_peak_speed"]
            if score >= config.THROW_ACTION_SCORE and peak < config.THROW_ACTION_MIN_SPEED:
                score = config.THROW_ACTION_SCORE - 0.01     # arm moved, but not a throw
            if score >= config.THROW_ACTION_SCORE:
                action = "THROWING"
            elif score >= 0.35:
                action = "arm moving"
            else:
                action = "standing / still"
            self._actions[tr.id] = (action, score)
            if score < config.THROW_ACTION_SCORE:
                continue
            last = self._throw_history[-1] if self._throw_history else None
            if ts - self._person_throw_cooldown.get(tr.id, -1e9) >= 3:
                self._person_throw_cooldown[tr.id] = ts
                self._throw_history.append({"t": ts, "person": tr.id, "score": score,
                                            "box": tr.box})
                self._action_tags.append((ts + 2.5, tr.box, f"THROWING {score:.0%}"))
            elif last is not None and last["person"] == tr.id:
                last["t"], last["score"] = ts, max(last["score"], score)   # same throw
            if (self.roi.defined and score >= 0.7
                    and ts - self._pending_throw_cooldown.get(tr.id, -1e9) >= 5):
                # needs the pool area to tell deck from water
                self._pending_throw_cooldown[tr.id] = ts
                self._pending_throws.append({"t": ts, "person": tr.id, "score": score,
                                             "box": tr.box})

    def _check_pending_throws(self, ts, diag):
        out, keep = [], []
        for pt in self._pending_throws:
            if ts - pt["t"] < 3.0:
                keep.append(pt)
                continue
            # Did any garbage appear, or a vanish/sunk get queued, after the throw?
            appeared = any(pt["t"] - 0.5 <= t.first_seen <= pt["t"] + 3.0
                           for t in self.g_tracker.tracks if t.confirmed)
            explained = appeared or any(abs(pv["t"] - pt["t"]) < 3.5
                                        for pv in self._pending_vanish)
            if not explained:
                x1, y1, x2, y2 = pt["box"]
                out.append(self._make_event(
                    "THROW_NO_FLOATING_OBJECT", "MEDIUM",
                    "Throwing motion toward the pool but nothing is floating - "
                    "object may have SUNK, check the pool",
                    location=(int((x1 + x2) / 2), int(y1)), throw_score=round(pt["score"], 2)))
        self._pending_throws = keep
        return out

    # ── tiny + underwater garbage (panel feedback) ─────────────
    def _search_region(self, w, h):
        """Bounding box of the pool area (whole frame if not drawn)."""
        if not self.roi.defined:
            return 0, 0, w, h
        import cv2
        x, y, bw, bh = cv2.boundingRect(self.roi.polygon(w, h))
        x, y = max(0, x), max(0, y)
        return x, y, min(w, x + bw), min(h, y + bh)

    def _run_on_crops(self, model, crops, offsets, conf, imgsz, ov=False):
        out = []
        if not crops:
            return out
        try:
            results = model.predict(crops, conf=conf, iou=config.IOU, imgsz=imgsz,
                                    verbose=False)
        except Exception:  # noqa: BLE001
            return out
        for r, (ox, oy) in zip(results, offsets):
            for box in r.boxes:
                name = model.names[int(box.cls[0])]
                if ov:
                    name = config.OPEN_VOCAB_MAP.get(name)
                    if name is None:
                        continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                out.append(((x1 + ox, y1 + oy, x2 + ox, y2 + oy), name, float(box.conf[0])))
        return out

    def _small_objects(self, src, w, h):
        """Tiled detection: each tile of the pool area is enlarged to
        TILE_IMGSZ, so tiny objects become large enough to detect."""
        if config.TILES <= 1 or self.model is None:
            return []
        every = max(1, config.TILE_EVERY)
        if self._g_runs % every == 0 or every == 1:
            x0, y0, x1, y1 = self._search_region(w, h)
            if x1 - x0 < 64 or y1 - y0 < 64:
                return []
            tiles = enh.make_tiles(x0, y0, x1, y1, config.TILES, config.TILE_OVERLAP)
            crops = [src[ty1:ty2, tx1:tx2] for tx1, ty1, tx2, ty2 in tiles]
            dets = self._run_on_crops(self.model, crops, [(t[0], t[1]) for t in tiles],
                                      config.TILE_CONF, config.TILE_IMGSZ)
            keep = []
            for box, label, conf in enh.nms(dets):
                cx, cy = center(box)
                if (box[2] - box[0]) * (box[3] - box[1]) > 0.15 * w * h:
                    continue
                if self.roi.contains(cx, cy, w, h):
                    keep.append((box, label, conf))
            self._tile_cache = keep
        return list(self._tile_cache)

    def _underwater_objects(self, src, base, w, h):
        """Colour-correct the pool area and search again. Returns
        (new detections, set of boxes judged UNDER the water surface)."""
        if not config.UNDERWATER:
            return [], set()
        model = self.model if self.model is not None else self.ov_model
        if model is None:
            return [], set()
        every = max(1, config.UNDERWATER_EVERY)
        if self._g_runs % every == min(1, every - 1) or every == 1:
            x0, y0, x1, y1 = self._search_region(w, h)
            if x1 - x0 < 64 or y1 - y0 < 64:
                return [], set()
            enhanced = enh.enhance_underwater(src[y0:y1, x0:x1])
            dets = self._run_on_crops(model, [enhanced], [(x0, y0)], config.UNDERWATER_CONF,
                                      self.imgsz, ov=model is self.ov_model)
            if (config.UNDERWATER_OPEN_VOCAB and self.ov_model is not None
                    and model is not self.ov_model):
                dets += self._run_on_crops(self.ov_model, [enhanced], [(x0, y0)],
                                           config.OPEN_VOCAB_CONF, config.OPEN_VOCAB_IMGSZ,
                                           ov=True)
            cache = []
            for box, label, conf in enh.nms(dets):
                cx, cy = center(box)
                if (box[2] - box[0]) * (box[3] - box[1]) > 0.15 * w * h:
                    continue
                if not self.roi.contains(cx, cy, w, h):
                    continue
                cache.append(((box, label, conf), enh.box_contrast(src, box),
                              enh.bright_fraction(src, box)))
            self._uw_cache = cache
        new, flagged = [], set()
        base_boxes = [b[0] for b in base]
        weak = list(getattr(self, "_weak_boxes", []))
        for det, contrast, bright in self._uw_cache:
            if enh.is_duplicate(det[0], base_boxes):
                continue                  # the normal search sees it: surface object
            new.append(det)
            base_boxes.append(det[0])
            if (not enh.is_duplicate(det[0], weak)          # invisible without correction
                    and contrast < config.UNDERWATER_CONTRAST  # blends into the water
                    and bright < 0.05):                      # no surface glare/highlights
                flagged.add(det[0])
        return new, flagged

    @staticmethod
    def _is_underwater(tr):
        flags = tr.meta.get("uw")
        if not flags:
            return False
        n = sum(flags)
        return n >= config.UNDERWATER_MIN_FLAGS and n / len(flags) >= 0.5

    def _check_went_under(self, g_active):
        """A garbage object that was alerted as FLOATING and is now only seen
        under the surface -> it sank (slowly)."""
        out = []
        for t in g_active:
            if t.meta.get("position") != "surface" or t.meta.get("went_under"):
                continue
            flags = list(t.meta.get("uw", ()))[-6:]
            if len(flags) == 6 and all(flags):
                t.meta["went_under"] = True
                cx, cy = center(t.box)
                out.append(self._make_event(
                    "SUNK_OBJECT", "MEDIUM",
                    f"{t.label} that was floating has gone UNDER the surface - "
                    f"check the pool floor", object=t.label, track_id=t.id,
                    location=(int(cx), int(cy)), position="underwater",
                    splash_strength=0.0))
        return out

    # ── human action helpers ───────────────────────────────────
    def _recent_throw(self, t_appear, px, py, diag, person):
        best = None
        for th in self._throw_history:
            if not (t_appear - config.THROW_LINK_SEC <= th["t"] <= t_appear + 0.5):
                continue
            near = (person is not None and th["person"] == person.id) or \
                _dist_point_box(px, py, th["box"]) <= config.MAX_SOURCE_DISTANCE_RATIO * diag
            if near and (best is None or th["t"] > best["t"]):
                best = th
        return best

    def _action_summary(self, p_active, ts):
        if not p_active:
            return "no person in view"
        if self._throw_history and ts - self._throw_history[-1]["t"] < 2.5:
            return f"THROWING (person #{self._throw_history[-1]['person']})"
        acts = [a for a, _ in self._actions.values()]
        for a in ("arm moving", "standing / still", "swimming"):
            if a in acts:
                return a
        return acts[0] if acts else "person"

    def _feed_item(self, ev, snapshot=None):
        title, explanation = config.EVENT_INFO.get(ev["type"], (ev["type"], ""))
        intent = ev.get("intent")
        return {
            "time": time.strftime("%H:%M:%S"),
            "type": ev["type"],
            "title": title,
            "explanation": explanation,
            "risk": ev["risk_level"],
            "object": ev.get("object"),
            "position": ev.get("position", "underwater" if ev["type"] in (
                "SUNK_OBJECT", "UNDERWATER_OBJECT", "THROW_NO_FLOATING_OBJECT") else "surface"),
            "source": ev.get("source"),
            "intent": intent,
            "intent_text": config.INTENT_INFO.get(intent, "") if intent else "",
            "reason": (ev.get("intent_info") or {}).get("reason", ""),
            "message": ev["message"],
            "snapshot": snapshot,
        }

    # ── drawing ────────────────────────────────────────────────
    def _draw(self, frame, g_active, p_active, l_active, ts, w, h):
        import cv2
        self.roi.draw(frame)
        for t in g_active:
            is_garbage = t.label in GARBAGE_CLASSES and not t.meta.get("outside_pool")
            tag = "" if t.confirmed else " ?"
            x1, y1, x2, y2 = t.box
            if is_garbage and self._is_underwater(t):
                draw_box(frame, x1, y1, x2, y2,
                         f"#{t.id} UNDERWATER {t.label} {t.conf:.0%}{tag}", UNDERWATER_COLOR)
                continue
            draw_box(frame, x1, y1, x2, y2, f"#{t.id} {t.label} {t.conf:.0%}{tag}",
                     GARBAGE_COLOR if is_garbage else NON_GARBAGE_COLOR)
        for t in (p_active if self.show_hud else ()):
            x1, y1, x2, y2 = t.box
            act = self._actions.get(t.id, ("person", 0))[0]
            draw_box(frame, x1, y1, x2, y2, f"person #{t.id}: {act}", PERSON_COLOR, thickness=1)
            last = self.pose.latest(t.id)
            if last and ts - last["ts"] < 0.5:
                kp = last["kp"]
                for a, b in (("l_sh", "l_el"), ("l_el", "l_wr"), ("r_sh", "r_el"), ("r_el", "r_wr")):
                    if kp[a][2] > 0.3 and kp[b][2] > 0.3:
                        cv2.line(frame, (int(kp[a][0]), int(kp[a][1])),
                                 (int(kp[b][0]), int(kp[b][1])), (0, 255, 0), 2)
        for t in (l_active if self.show_hud else ()):
            x1, y1, x2, y2 = t.box
            draw_box(frame, x1, y1, x2, y2, f"{t.label} (hand/air)", LITTER_COLOR, thickness=1)
            pts = [(int(p[1]), int(p[2])) for p in t.points]
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, LITTER_COLOR, 2)
        # THROWING tag stays on the person for 2.5 s (also on the dashboard)
        self._action_tags = [a for a in self._action_tags if a[0] > ts]
        for _, (x1, y1, x2, y2), text in self._action_tags:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), THROW_COLOR, 2)
            cv2.putText(frame, text, (int(x1), max(15, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, THROW_COLOR, 2)
        self._markers = [m for m in self._markers if m[0] > ts]
        for _, x, y, label in self._markers:
            cv2.circle(frame, (int(x), int(y)), 28, SUNK_COLOR, 3)
            cv2.putText(frame, label, (int(x) - 60, int(y) - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, SUNK_COLOR, 2)

    # ── simulation (no model file) ─────────────────────────────
    def _simulate(self, w, h):
        if self._sim_cooldown > 0:
            self._sim_cooldown -= 1
            return [((w // 2 - 30, h // 2 - 30, w // 2 + 30, h // 2 + 30),
                     random.choice(SIM_OBJECTS), 0.9)]
        if random.random() < 0.01:
            self._sim_cooldown = 20
        return []

    def close(self):
        self.pose.close()
