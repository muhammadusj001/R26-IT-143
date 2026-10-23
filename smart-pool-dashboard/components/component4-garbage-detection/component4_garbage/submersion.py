"""
Heavy / sinking garbage — how we still catch it with a camera above water.

The supervisor's point is correct: a glass bottle, stone, can or phone
sinks within a second or two, so a detector that only looks at the
water SURFACE never gets a clear look at it. We therefore do not rely on
"seeing the floating object". Instead we combine three independent clues:

1. ENTRY EVENT (vanished-in-pool): an object (garbage-model track, or a
   COCO bottle/cup that was in someone's hand / in the air) is tracked
   while it moves, its last position is INSIDE the pool area, and then it
   disappears without leaving the frame and without a person picking it
   up. Floating objects stay visible; an object that vanishes in the water
   has most likely sunk.

2. SPLASH CONFIRMATION: at the moment and place where the object
   vanished, we measure a burst of pixel change on the water (the splash /
   ripple ring). Swimmer areas are masked out. A splash + vanish = strong
   evidence that something went into the water.

3. POOL-FLOOR CHANGE: with clear water and a camera looking down at an
   angle, the pool floor is visible. We learn what the empty floor looks
   like (slow background model) and raise an event when a NEW, small,
   still blob stays on the floor for several seconds (masking swimmers and
   floating detections). This catches objects that sank while no one saw
   them go in, e.g. at night or out of the person detector's view.

Plus a 4th rule lives in detector.py: a strong THROW MOTION by a person on
the pool deck, with no floating garbage appearing afterwards, raises
"possible sunk object — check the pool" for staff.

Limits (say these honestly in the thesis): sun glitter, strong ripples
and murky water reduce clue 3; very deep pools or night need lighting or
an underwater/side-wall camera. The best fix for full depth coverage is a
second camera (underwater IP camera or wall-mounted looking across the
floor) feeding this same module.
"""

import json
import time
from collections import deque

import cv2
import numpy as np

from component4_garbage import config


# ── Pool region of interest ───────────────────────────────────────
class PoolROI:
    def __init__(self, path=None):
        self.path = path or config.POOL_ROI_FILE
        self.norm_points = None          # polygon in 0-1 coords
        self.drawn_size = None           # [w, h] of the frame it was drawn on
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.norm_points = data.get("polygon") or None
                self.drawn_size = data.get("size")
            except Exception:  # noqa: BLE001
                self.norm_points = None
        self._cache = {}

    def check_size(self, w, h):
        """Warn if the live frame's shape differs from the one the pool area
        was drawn on (e.g. drawn at 1280x720 but the dashboard sends 640x480):
        the polygon is stored as fractions, so a different aspect ratio
        shifts it off the water."""
        if not self.defined or not self.drawn_size:
            return None
        dw, dh = self.drawn_size
        if abs(dw / float(dh) - w / float(h)) > 0.05:
            return (f"pool area was drawn on a {dw}x{dh} frame but frames are {w}x{h}. "
                    f"Redraw it: python tools/draw_pool_roi.py --source <cam> --width {w} --height {h}")
        return None

    @property
    def defined(self):
        return bool(self.norm_points) and len(self.norm_points) >= 3

    def polygon(self, w, h):
        key = (w, h)
        if key not in self._cache:
            pts = [(int(x * w), int(y * h)) for x, y in (self.norm_points or [])]
            self._cache[key] = np.array(pts, dtype=np.int32)
        return self._cache[key]

    def contains(self, x, y, w, h):
        if not self.defined:
            return True
        return cv2.pointPolygonTest(self.polygon(w, h), (float(x), float(y)), False) >= 0

    def mask(self, w, h):
        m = np.zeros((h, w), np.uint8)
        if self.defined:
            cv2.fillPoly(m, [self.polygon(w, h)], 255)
        else:
            m[:] = 255
        return m

    def draw(self, frame):
        if self.defined:
            h, w = frame.shape[:2]
            cv2.polylines(frame, [self.polygon(w, h)], True, (255, 200, 0), 2)


# ── Splash detector ──────────────────────────────────────────────
class SplashDetector:
    """Keeps a short history of 'what changed' maps on a small frame."""

    SCALE_W = 320

    def __init__(self, keep_sec=None):
        self.keep_sec = (keep_sec or config.SPLASH_WINDOW_SEC) + 1.5
        self.prev = None
        self.history = deque()           # (ts, change_mask)
        self.scale = 1.0

    def update(self, frame, ts, ignore_boxes=()):
        h, w = frame.shape[:2]
        self.scale = self.SCALE_W / float(w)
        small = cv2.resize(frame, (self.SCALE_W, int(h * self.scale)))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.prev is not None and self.prev.shape == gray.shape:
            diff = cv2.absdiff(gray, self.prev)
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            for x1, y1, x2, y2 in ignore_boxes:
                cv2.rectangle(mask, (int(x1 * self.scale), int(y1 * self.scale)),
                              (int(x2 * self.scale), int(y2 * self.scale)), 0, -1)
            self.history.append((ts, mask))
        self.prev = gray
        while self.history and ts - self.history[0][0] > self.keep_sec:
            self.history.popleft()

    def splash_strength(self, x, y, t0, t1, radius_px=60):
        """Max fraction of changed pixels in a patch around (x, y) between t0..t1."""
        cx, cy = int(x * self.scale), int(y * self.scale)
        r = max(6, int(radius_px * self.scale))
        best = 0.0
        for ts, mask in self.history:
            if t0 <= ts <= t1:
                patch = mask[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
                if patch.size:
                    best = max(best, float(np.count_nonzero(patch)) / patch.size)
        return best


# ── Pool floor change monitor ────────────────────────────────────
class FloorChangeMonitor:
    """Finds new, still blobs on the pool floor that stay for seconds."""

    SCALE_W = 480

    def __init__(self, roi):
        self.roi = roi
        self.bg = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=40,
                                                     detectShadows=False)
        self.candidates = []             # dicts: box, first, last, reported
        self.frame_count = 0
        self.warmup_frames = 60
        self.scale = 1.0

    def update(self, frame, ts, ignore_boxes=()):
        """Returns list of NEW floor-object events [{'box':..., 'age':...}]."""
        self.frame_count += 1
        if self.frame_count % 3:          # every 3rd frame is enough
            return []
        h, w = frame.shape[:2]
        self.scale = self.SCALE_W / float(w)
        sw, sh = self.SCALE_W, int(h * self.scale)
        small = cv2.resize(frame, (sw, sh))
        fg = self.bg.apply(small, learningRate=0.002)
        if self.frame_count < self.warmup_frames:
            return []
        fg = cv2.bitwise_and(fg, self.roi.mask(sw, sh))
        for x1, y1, x2, y2 in ignore_boxes:          # swimmers, floating garbage
            pad = 10
            cv2.rectangle(fg, (int(x1 * self.scale) - pad, int(y1 * self.scale) - pad),
                          (int(x2 * self.scale) + pad, int(y2 * self.scale) + pad), 0, -1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = config.FLOOR_MAX_AREA_RATIO * sw * sh
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if config.FLOOR_MIN_AREA * (self.scale ** 2) * 4 <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(c)
                blobs.append((x, y, x + bw, y + bh))

        new_events = []
        for b in blobs:
            match = None
            for cand in self.candidates:
                if _iou(cand["box"], b) > 0.3:
                    match = cand
                    break
            if match:
                match["box"], match["last"] = b, ts
            else:
                self.candidates.append({"box": b, "first": ts, "last": ts, "reported": False})
        keep = []
        for cand in self.candidates:
            if ts - cand["last"] > 2.0:       # blob gone (was a ripple)
                continue
            if not cand["reported"] and ts - cand["first"] >= config.FLOOR_PERSIST_SEC:
                cand["reported"] = True
                x1, y1, x2, y2 = cand["box"]
                s = 1.0 / self.scale
                new_events.append({"box": (int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s)),
                                   "age": round(ts - cand["first"], 1)})
            keep.append(cand)
        self.candidates = keep
        return new_events


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / float(ua)


def now():
    return time.time()
