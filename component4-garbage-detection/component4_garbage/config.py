"""
Component 4 — central settings (v2, live-camera edition).

Every tunable number lives here so you can tune the system for your
pool and camera without touching the logic files. Each value can also
be overridden with an environment variable of the same name, e.g.

    set C4_CONF=0.35            (Windows CMD)
    $env:C4_CONF="0.35"         (Windows PowerShell)
    export C4_CONF=0.35         (macOS / Linux)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"
EVENTS_DIR = ROOT / "events"


def _env(name, default, cast=float):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return cast(value)
    except ValueError:
        return default


# ── Models ────────────────────────────────────────────────────────
GARBAGE_MODEL_PATH = Path(os.environ.get(
    "C4_GARBAGE_MODEL", MODELS_DIR / "swimming_pool_garbage_yolo.pt"))
# Stock COCO model for people + hand-held / airborne bottles & cups.
# If the file is not in models/, Ultralytics downloads "yolov8n.pt"
# automatically on first run (needs internet once).
SOURCE_MODEL_PATH = Path(os.environ.get("C4_SOURCE_MODEL", MODELS_DIR / "yolov8n.pt"))
POSE_MODEL_PATH = Path(os.environ.get(
    "C4_POSE_MODEL", MODELS_DIR / "pose_landmarker_lite.task"))
# Optional trained intent classifier (tools/train_intent_classifier.py).
INTENT_CLF_PATH = Path(os.environ.get("C4_INTENT_CLF", MODELS_DIR / "intent_clf.joblib"))

# ── Garbage detection ─────────────────────────────────────────────
CONF = _env("C4_CONF", 0.35)            # raised from 0.25: fewer false boxes
IOU = _env("C4_IOU", 0.5)
IMGSZ = _env("C4_IMGSZ", 960, int)      # bigger = small/far objects found better
CONFIRM_FRAMES = _env("C4_CONFIRM_FRAMES", 4, int)    # seen in N frames ...
CONFIRM_WINDOW = _env("C4_CONFIRM_WINDOW", 8, int)    # ... out of the last M
ALERT_COOLDOWN_SEC = _env("C4_ALERT_COOLDOWN", 10.0)  # per object track
# Run the garbage model on every Nth call (floating garbage moves slowly).
# 1 = every call: right for the dashboard, whose stagger already skips frames.
# standalone_demo.py uses 2 on a CPU laptop (--garbage-every).
GARBAGE_EVERY_N = _env("C4_GARBAGE_EVERY", 1, int)

ALL_CLASSES = ["aluminium foil", "ball", "bottle", "juice", "leaf", "thermocol"]
# Preserved from the original app.py: ball and leaf never raise an alert.
GARBAGE_CLASSES = ["aluminium foil", "bottle", "juice", "thermocol",
                   # found by the COCO / open-vocabulary second opinions:
                   "cup", "paper", "plastic bag", "food tray", "can"]

# ── People / source objects (COCO ids) ────────────────────────────
SOURCE_CONF = _env("C4_SOURCE_CONF", 0.40)
PERSON_CLASS = 0
ANIMAL_CLASSES = {14: "bird", 15: "cat", 16: "dog"}
# COCO objects that are usually litter when they leave a hand:
# 39 bottle, 41 cup, 40 wine glass, 46 banana, 47 apple, 67 cell phone
COCO_LITTER_CLASSES = {39: "bottle", 41: "cup", 40: "glass", 46: "banana", 47: "apple",
                       # white foam food boxes / plates are usually seen by COCO
                       # as "frisbee" or "bowl": treat them as a food tray
                       29: "food tray", 45: "food tray"}
# COCO second opinion for floating bottles/cups the garbage model misses.
COCO_FALLBACK = os.environ.get("C4_COCO_FALLBACK", "1") == "1"
COCO_FALLBACK_CONF = _env("C4_COCO_FALLBACK_CONF", 0.30)
COCO_FALLBACK_LABELS = {"bottle": "bottle", "cup": "cup", "food tray": "food tray"}
# Open-vocabulary detector (YOLOE): finds garbage described in WORDS, incl.
# types your trained model has never seen (paper, plastic bag, food tray,
# can). The words are already built into models/pool_garbage_openvocab.pt,
# so no download is needed. Turn off with C4_OPEN_VOCAB=0 if the PC is slow.
OPEN_VOCAB = os.environ.get("C4_OPEN_VOCAB", "1") == "1"
OPEN_VOCAB_MODEL_PATH = Path(os.environ.get("C4_OPEN_VOCAB_MODEL",
                                            MODELS_DIR / "pool_garbage_openvocab.pt"))
OPEN_VOCAB_CONF = _env("C4_OPEN_VOCAB_CONF", 0.25)
OPEN_VOCAB_EVERY = _env("C4_OPEN_VOCAB_EVERY", 3, int)   # run on every Nth call
OPEN_VOCAB_IMGSZ = _env("C4_OPEN_VOCAB_IMGSZ", 640, int)
# word in the model -> label used by the system (None = not garbage, ignored)
OPEN_VOCAB_MAP = {
    "plastic bottle": "bottle", "crumpled paper": "paper", "tissue paper": "paper",
    "plastic bag": "plastic bag", "styrofoam block": "thermocol",
    "foam food tray": "food tray", "aluminium foil": "aluminium foil",
    "drink can": "can", "plastic cup": "cup", "juice box": "juice",
    "leaf": None, "ball": None,
}
MAX_SOURCE_DISTANCE_RATIO = _env("C4_MAX_SOURCE_DIST", 0.45)  # x frame diagonal

# Garbage model fallback order if C4_GARBAGE_MODEL / the default is missing.
GARBAGE_MODEL_FALLBACKS = [MODELS_DIR / "swimming_pool_garbage_yolo.pt",
                           MODELS_DIR / "swimming_pool_garbage_yolo_v2.pt"]

# ── Tiny garbage: tiled (zoomed) detection over the pool area ─────
# The pool area is cut into TILES x TILES overlapping pieces; each piece is
# enlarged to TILE_IMGSZ before detection, so very small objects (bottle caps,
# wrappers, far-away pieces) become big enough for the model. 0 = off.
TILES = _env("C4_TILES", 2, int)
TILE_EVERY = _env("C4_TILE_EVERY", 4, int)      # refresh tiles on every Nth garbage run
TILE_OVERLAP = _env("C4_TILE_OVERLAP", 0.2)
TILE_IMGSZ = _env("C4_TILE_IMGSZ", 640, int)
TILE_CONF = _env("C4_TILE_CONF", 0.35)

# ── Garbage UNDER the water surface ───────────────────────────────
# Water absorbs red light and blurs/lowers contrast, so submerged garbage
# looks blue-grey and the normal model misses it. The pool area is colour-
# corrected (grey-world white balance + red-channel compensation + CLAHE
# contrast) and searched again. An object found ONLY after enhancement and
# with low contrast against the water around it is marked "underwater".
UNDERWATER = os.environ.get("C4_UNDERWATER", "1") == "1"
UNDERWATER_EVERY = _env("C4_UNDERWATER_EVERY", 4, int)
UNDERWATER_CONF = _env("C4_UNDERWATER_CONF", 0.30)
UNDERWATER_OPEN_VOCAB = os.environ.get("C4_UNDERWATER_OPEN_VOCAB", "0") == "1"
UNDERWATER_CONTRAST = _env("C4_UNDERWATER_CONTRAST", 18.0)   # Lab distance object vs water
UNDERWATER_MIN_FLAGS = _env("C4_UNDERWATER_MIN_FLAGS", 3, int)
WEAK_CONF = _env("C4_WEAK_CONF", 0.12)    # normal model "sees it weakly" = surface object

# ── Intent (throw vs accidental) ──────────────────────────────────
POSE_HISTORY_SEC = _env("C4_POSE_HISTORY", 6.0)   # how far back we can look
# Window before the object appeared. Garbage is often only CONFIRMED 1-3 s
# after it lands (it has to settle and be seen in several frames), so the
# arm swing can be several seconds earlier.
LOOKBACK_SEC = _env("C4_LOOKBACK", 4.0)
# Human action recognition: the arm-swing detector runs on every person all
# the time. Score >= THROW_ACTION_SCORE = "throwing motion". Garbage that
# appears within THROW_LINK_SEC after a nearby throwing motion is judged
# intentional even if the pose look-back alone was unclear.
THROW_ACTION_SCORE = _env("C4_THROW_ACTION_SCORE", 0.7)
THROW_LINK_SEC = _env("C4_THROW_LINK_SEC", 6.0)
# A person must be at least this tall (fraction of frame height) for arm
# analysis: on tiny/far people the pose points jitter and look like throws.
MIN_ACTION_BODY_RATIO = _env("C4_MIN_ACTION_BODY", 0.25)
THROW_ACTION_MIN_SPEED = _env("C4_THROW_ACTION_MIN_SPEED", 2.0)  # body-heights / s
MAX_POSE_PEOPLE = _env("C4_MAX_POSE_PEOPLE", 3, int)
# Thresholds are in body-heights per second (scale + FPS independent).
WRIST_SPEED_THROW = _env("C4_WRIST_SPEED_THROW", 1.6)
WRIST_SPEED_DROP = _env("C4_WRIST_SPEED_DROP", 0.6)
OBJECT_LAUNCH_SPEED = _env("C4_OBJECT_LAUNCH_SPEED", 0.9)   # frame-heights / s
INTENT_SCORE_THRESHOLD = _env("C4_INTENT_THRESHOLD", 0.55)

# Garbage first seen within this many seconds of the view starting (start-up
# or scene cut) was already there -> no throw/drop judgement.
ALREADY_PRESENT_SEC = _env("C4_ALREADY_PRESENT", 2.0)

# ── Sinking (heavy) garbage ───────────────────────────────────────
LOST_TRACK_SEC = _env("C4_LOST_TRACK_SEC", 1.5)     # object vanished inside pool
SPLASH_PIXEL_RATIO = _env("C4_SPLASH_RATIO", 0.035) # changed pixels in patch
SPLASH_WINDOW_SEC = _env("C4_SPLASH_WINDOW", 3.0)
FLOOR_CHANGE_ENABLED = os.environ.get("C4_FLOOR_CHANGE", "1") == "1"
FLOOR_PERSIST_SEC = _env("C4_FLOOR_PERSIST", 6.0)   # blob must stay this long
FLOOR_MIN_AREA = _env("C4_FLOOR_MIN_AREA", 120, int)
FLOOR_MAX_AREA_RATIO = _env("C4_FLOOR_MAX_AREA", 0.02)

# Scene cut / camera moved: colour histogram correlation with the previous
# frame below this value -> trackers restart (no fake throws / sunk objects).
SCENE_CUT_CORR = _env("C4_SCENE_CUT_CORR", 0.6)
# ...or the average brightness change between frames above this (0-255).
# Normal live video: ~5-20. A cut / camera knocked: 40-50+.
SCENE_CUT_MAD = _env("C4_SCENE_CUT_MAD", 35.0)

# ── Pool region (drawn with tools/draw_pool_roi.py) ───────────────
POOL_ROI_FILE = CONFIG_DIR / "pool_roi.json"

# ── Plain-language event names for the dashboard (panel feedback) ─
EVENT_INFO = {
    "GARBAGE_IN_POOL": (
        "Garbage on the water surface",
        "A garbage object was confirmed floating inside the pool area."),
    "SUBMERGED_GARBAGE": (
        "Garbage under the water",
        "A garbage object was found below the water surface (seen after "
        "underwater colour correction). It will not float away - remove it from the floor."),
    "SUNK_OBJECT": (
        "Object sank",
        "An object disappeared inside the pool (often with a splash), or a floating "
        "object went under the surface. It is probably on the pool floor."),
    "THROW_NO_FLOATING_OBJECT": (
        "Throw seen, nothing floating",
        "A person made a throwing movement toward the pool but nothing appeared on "
        "the surface. A heavy object may have sunk."),
    "UNDERWATER_OBJECT": (
        "New object on the pool floor",
        "Something new has stayed still on the pool floor for several seconds "
        "(pool-floor change check)."),
}
INTENT_INFO = {
    "intentional": "Thrown on purpose (fast arm swing / object launched)",
    "accidental": "Dropped by accident (arm almost still / object fell down)",
    "already_present": "Already in the pool when monitoring started",
    "environmental": "No person nearby (wind / drifted in)",
    "animal": "An animal was nearby",
    "unknown": "Person nearby but their arms were not clearly visible",
}

# ── Display ───────────────────────────────────────────────────────
# Status panel + people/pose overlays on the video. Off by default so the
# integrated dashboard frame stays clean; standalone_demo.py turns it on.
SHOW_HUD = os.environ.get("C4_SHOW_HUD", "0") == "1"
# Person boxes with the live action (THROWING / arm moving / standing) and arm
# lines, also on the dashboard video. C4_SHOW_PEOPLE=0 hides them.
SHOW_PEOPLE = os.environ.get("C4_SHOW_PEOPLE", "1") == "1"
# Where the YOLO models run: auto = Apple GPU (mps) on M1/M2/M3 Macs, NVIDIA GPU
# (cuda) if present, else CPU. Force with C4_DEVICE=cpu if the GPU path misbehaves.
DEVICE = os.environ.get("C4_DEVICE", "auto")

# ── Event logging ─────────────────────────────────────────────────
SAVE_SNAPSHOTS = os.environ.get("C4_SAVE_SNAPSHOTS", "1") == "1"
MONGO_URI = os.environ.get("MONGO_URI", "")          # optional
MONGO_DB = os.environ.get("MONGO_DB", "pool_monitor")
