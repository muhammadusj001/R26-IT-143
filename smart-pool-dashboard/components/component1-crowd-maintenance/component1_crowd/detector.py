"""
Component 1 — Crowd-Aware Maintenance Scheduling (runtime service).

Uses the team's REAL Component 1 assets:
  - trained single-class YOLOv8 swimmer model, OR a stock COCO model
    (yolo11m.pt / yolov8m.pt) which detects the generic "person" class
  - BatherLoadCalculator (recovered from bytecode, see bather_load.py)
  - MaintenanceScheduler with its original person-hour thresholds

MODEL AUTO-DETECTION:
The detector inspects the loaded model's class names. If it sees the 80
COCO classes it switches to COCO mode and counts ONLY class 0 ("person"),
ignoring chairs, umbrellas, phones etc. If it sees a small custom class
list (e.g. ["swimmer"]) it counts every detection, as before. This means
the same code works with either model with no edits.

COUNT SMOOTHING:
Raw per-frame counts flicker when detections sit near the confidence
threshold. A short rolling median is applied so the dashboard shows a
stable number, while the underlying detections remain unmodified.

POOL REGION (ROI) FILTER:
An optional pool_polygon (relative 0.0-1.0 corner points) restricts
counting to people actually inside the pool water, so deck/poolside
bystanders aren't counted as swimmers. Detection still runs on the
whole frame — people outside the polygon are drawn dimmed but excluded
from swimmer_count / bather load / density. With no polygon configured,
behaviour is unchanged (everyone detected counts, as before).
"""

import fnmatch
import random
import shutil
import time
from collections import deque
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from component1_crowd.drawing import draw_box
from component1_crowd.bather_load import BatherLoadCalculator
from component1_crowd.scheduler import MaintenanceScheduler

SWIMMER_COLOR = (255, 200, 0)
OUT_OF_POOL_COLOR = (140, 140, 140)   # dim grey — detected but outside the pool region
POOL_OUTLINE_COLOR = (238, 211, 34)   # aqua (BGR), matches the dashboard's aqua accent
PERSON_CLASS_ID = 0          # "person" in COCO
SMOOTHING_WINDOW = 5         # frames used for the rolling median

# Standalone defaults (the dashboard overrides these via the constructor)
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.45
DENSITY_MODERATE = 4
DENSITY_HIGH = 8

# Bather load standard: water surface area required per swimmer.
# Based on international pool capacity guidance (shallow ~2.2 m2,
# deep ~2.7 m2 per bather). Configurable per facility.
DEFAULT_POOL_AREA_M2 = 250.0      # e.g. a 25m x 10m pool
DEFAULT_AREA_PER_BATHER_M2 = 2.7
DENSITY_MODERATE_PCT = 0.50       # 50% of capacity
DENSITY_HIGH_PCT = 0.80           # 80% of capacity

# Standard Ultralytics weights that YOLO() can auto-download by name.
STOCK_MODEL_PATTERNS = ("yolo11*.pt", "yolov8*.pt")
# Used only if the configured model is missing and can't be downloaded.
FALLBACK_MODEL_NAME = "best_swimmer_model.pt"


def _is_stock_model(name):
    return any(fnmatch.fnmatch(name.lower(), pat) for pat in STOCK_MODEL_PATTERNS)


class CrowdDetector:
    def __init__(self, model_path=None, conf_threshold=DEFAULT_CONF,
                 iou_threshold=DEFAULT_IOU,
                 density_moderate=DENSITY_MODERATE, density_high=DENSITY_HIGH,
                 imgsz=None, smooth=True,
                 pool_area_m2=DEFAULT_POOL_AREA_M2,
                 area_per_bather_m2=DEFAULT_AREA_PER_BATHER_M2,
                 pool_polygon=None):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Kept for backward compatibility; density_level() now uses
        # capacity percentages instead of these fixed counts.
        self.density_moderate = density_moderate
        self.density_high = density_high
        self.imgsz = imgsz            # e.g. 1280 helps distant swimmers
        self.smooth = smooth

        self.pool_area_m2 = pool_area_m2
        self.area_per_bather_m2 = area_per_bather_m2
        self.max_capacity = int(pool_area_m2 / area_per_bather_m2)

        # List of (x, y) points in RELATIVE coordinates (0.0-1.0), or
        # None to count every detected person (no ROI filtering).
        self.pool_polygon = pool_polygon

        self.model = None
        self.model_status = "not_loaded"
        self.is_coco_model = False    # set during load()

        self.bather_load = BatherLoadCalculator()
        self.scheduler = MaintenanceScheduler()
        self._last_ts = None
        self._sim_count = 3
        self._recent_counts = deque(maxlen=SMOOTHING_WINDOW)

    # ── Loading ──────────────────────────────────────────────
    def load(self):
        if self.model_path is None:
            self.model_status = "simulation"
            return False
        model_path = Path(self.model_path)
        try:
            from ultralytics import YOLO

            if model_path.exists():
                self.model = YOLO(str(model_path))
            elif _is_stock_model(model_path.name):
                self.model = self._download_stock_model(YOLO, model_path)
            else:
                raise FileNotFoundError(f"model file not found: {model_path}")
        except Exception as exc:  # noqa: BLE001
            fallback = model_path.parent / FALLBACK_MODEL_NAME
            print(f"  WARNING: crowd model {model_path.name} unavailable ({exc})")
            if fallback == model_path or not fallback.exists():
                print("  WARNING: no fallback crowd model — running in simulation mode")
                self.model = None
                self.model_status = "simulation"
                return False
            print(f"  WARNING: falling back to {fallback.name}")
            try:
                self.model = YOLO(str(fallback))
                model_path = fallback
            except Exception as fb_exc:  # noqa: BLE001
                self.model = None
                self.model_status = f"error: {fb_exc}"
                return False

        # A COCO model has 80 classes with "person" at index 0.
        names = self.model.names
        self.is_coco_model = (
            len(names) > 10 and str(names.get(0, "")).lower() == "person"
        )
        mode = "COCO/person" if self.is_coco_model else "custom/swimmer"
        self.model_status = "loaded"
        print(f"  Crowd model: {model_path.name} — {mode} mode, "
              f"conf={self.conf_threshold}, imgsz={self.imgsz}")
        return True

    @staticmethod
    def _download_stock_model(YOLO, target):
        """Load a standard Ultralytics model by bare name (auto-downloads
        it on first use), then cache it at `target` for next time."""
        print(f"  Crowd model {target.name} not found — downloading it "
              f"(first run only, needs internet)...")
        # Ultralytics downloads bare names into the current directory.
        cwd_copy = Path.cwd() / target.name
        had_cwd_copy = cwd_copy.exists()
        model = YOLO(target.name)
        source = Path(model.ckpt_path or cwd_copy).resolve()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source != target.resolve():
                if source == cwd_copy.resolve() and not had_cwd_copy:
                    shutil.move(str(source), target)
                else:
                    shutil.copy2(source, target)
            print(f"  Saved crowd model to {target}")
        except OSError as exc:
            print(f"  WARNING: could not save {target.name} to {target.parent} ({exc}); "
                  f"it will be downloaded again next run")
        return model

    # ── Density / maintenance rules (capacity-based) ──────────
    def density_level(self, count: int) -> str:
        occupancy = count / self.max_capacity
        if occupancy >= 1.0:
            return "OVER CAPACITY"
        if occupancy >= DENSITY_HIGH_PCT:
            return "HIGH"
        if occupancy >= DENSITY_MODERATE_PCT:
            return "MODERATE"
        return "LOW"

    # ── Pool region (ROI) filter ──────────────────────────────
    def is_in_pool(self, box, frame_width, frame_height) -> bool:
        """box = (x1, y1, x2, y2) in pixel coordinates. Checks the
        bottom-centre point of the box — where a person meets the
        water/ground — against the pool polygon. Returns True
        unconditionally when no polygon is configured, so behaviour is
        unchanged by default."""
        if self.pool_polygon is None:
            return True

        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2.0
        point = (x_center, float(y2))

        polygon_px = np.array(
            [(px * frame_width, py * frame_height) for px, py in self.pool_polygon],
            dtype=np.float32,
        )
        return cv2.pointPolygonTest(polygon_px, point, False) >= 0

    def _draw_pool_region(self, frame):
        """Translucent fill + thin aqua outline showing the configured
        pool region, so it's visible in the demo. No-op with no polygon."""
        if not self.pool_polygon:
            return
        h, w = frame.shape[:2]
        pts = np.array(
            [(int(px * w), int(py * h)) for px, py in self.pool_polygon],
            dtype=np.int32,
        )
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], POOL_OUTLINE_COLOR)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=POOL_OUTLINE_COLOR, thickness=2)

    # ── Per-frame processing ─────────────────────────────────
    def process(self, frame):
        if self.model is not None:
            raw_count, total_people, people_outside = self._detect(frame)
        else:
            raw_count, total_people, people_outside = self._simulate()

        # Rolling median removes single-frame flicker without hiding
        # genuine changes in occupancy.
        self._recent_counts.append(raw_count)
        count = int(median(self._recent_counts)) if self.smooth else raw_count

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
            recommendation = (f"{top['action']} — {top['priority']} "
                              f"(overdue by {top['overdue_by']} ph)")
        else:
            recommendation = "No maintenance needed"

        density = self.density_level(count)
        occupancy_pct = round((count / self.max_capacity) * 100, 1)
        if density == "OVER CAPACITY":
            crowd_alert = (f"POOL OVER CAPACITY: {count}/{self.max_capacity} "
                           f"bathers ({occupancy_pct}%)")
        elif density == "HIGH":
            crowd_alert = (f"Pool nearing capacity: {count}/{self.max_capacity} "
                           f"bathers ({occupancy_pct}%)")
        else:
            crowd_alert = None

        return frame, {
            "swimmer_count": count,
            "raw_count": raw_count,
            "density_level": density,
            "bather_load": load_hours,
            "peak_hour": self.bather_load.get_peak_hour(),
            "pending_actions": len(recs),
            "recommendations": recs[:3],
            "maintenance_recommendation": recommendation,
            "model_status": self.model_status,
            "max_capacity": self.max_capacity,
            "occupancy_pct": occupancy_pct,
            "crowd_alert": crowd_alert,
            "total_people_detected": total_people,
            "people_outside_pool": people_outside,
        }

    def _detect(self, frame):
        kwargs = {"conf": self.conf_threshold, "iou": self.iou_threshold, "verbose": False}
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        # Ask YOLO itself to return only people when running a COCO model.
        if self.is_coco_model:
            kwargs["classes"] = [PERSON_CLASS_ID]

        self._draw_pool_region(frame)
        frame_h, frame_w = frame.shape[:2]

        results = self.model(frame, **kwargs)
        in_pool_count = 0
        total_people = 0
        people_outside = 0
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # Belt-and-braces: skip non-person boxes even if the
                # classes filter above was not applied.
                if self.is_coco_model and cls_id != PERSON_CLASS_ID:
                    continue
                name = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                total_people += 1

                if self.is_in_pool((x1, y1, x2, y2), frame_w, frame_h):
                    in_pool_count += 1
                    draw_box(frame, x1, y1, x2, y2,
                             f"{name} {float(box.conf[0]):.0%}", SWIMMER_COLOR)
                else:
                    people_outside += 1
                    draw_box(frame, x1, y1, x2, y2,
                             f"{name} {float(box.conf[0]):.0%}", OUT_OF_POOL_COLOR,
                             thickness=1)
        return in_pool_count, total_people, people_outside

    def _simulate(self):
        self._sim_count = max(0, min(12, self._sim_count + random.choice([-1, 0, 0, 1])))
        # Simulation has no frame to filter, so every simulated person
        # is treated as in-pool (matches "no polygon = unchanged" default).
        return self._sim_count, self._sim_count, 0
