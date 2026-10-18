"""
Central configuration for the Smart Pool Monitoring System.

All machine-specific values (paths, camera source, serial port) live here
and can be overridden with environment variables. This replaces the
hardcoded Windows paths, phone IPs and serial ports that previously
existed inside each component's scripts.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Project paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
# Component repos are mounted as git submodules under components/
COMPONENTS_DIR = BASE_DIR / "components"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

# ── Model paths (per module) ─────────────────────────────────
DROWNING_MODEL_PATH = Path(
    os.getenv("DROWNING_MODEL_PATH",
              COMPONENTS_DIR / "component3-drowning-detection" / "models" / "best.pt")
)
CROWD_MODEL_PATH = Path(
    os.getenv(
        "CROWD_MODEL_PATH",
        COMPONENTS_DIR
        / "component1-crowd-maintenance"
        / "models"
        / "best_swimmer_model.pt"
    )
)
GARBAGE_MODEL_PATH = Path(
    os.getenv("GARBAGE_MODEL_PATH",
              COMPONENTS_DIR / "component4-garbage-detection" / "models" / "swimming_pool_garbage_yolo.pt")
)
WATER_MODEL_DIR = Path(
    os.getenv("WATER_MODEL_DIR",
              COMPONENTS_DIR / "component2-water-quality" / "models")
)

# ── Camera source (shared by crowd / drowning / garbage) ─────
# Accepts:
#   an integer index ("0", "1", "3") -> webcam / DroidCam Client / OBS Virtual Camera
#   "droidcam:<phone-ip>" -> DroidCam app over Wi-Fi (http://<phone-ip>:4747/video)
#   a stream URL ("http://<phone-ip>:8080/video") -> IP Webcam app
#   a video file path -> loops, for demos
#   "simulate" -> synthetic frames (no camera needed, demo mode)
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "simulate")
# Seconds before the first reconnect attempt (doubles up to 10 s)
CAMERA_RECONNECT_DELAY = float(os.getenv("CAMERA_RECONNECT_DELAY", "2"))
# Open/read timeout for URL streams (DroidCam over Wi-Fi)
CAMERA_OPEN_TIMEOUT_MS = int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "5000"))
# Resize every frame to FRAME_WIDTH x FRAME_HEIGHT
CAMERA_RESIZE = os.getenv("CAMERA_RESIZE", "1") == "1"

# ── Water quality sensor source ──────────────────────────────
# Serial port of the Arduino (e.g. "COM3", "/dev/cu.usbmodem21401"),
# "auto" (default) to probe for a plugged-in Arduino with zero config —
# if none is found, the card stays "OFFLINE" with no data rather than
# inventing readings, and keeps probing so plugging it in later is
# picked up without a restart — or "simulate" to explicitly force
# simulated demo data (never automatic; must be requested).
SENSOR_SOURCE = os.getenv("SENSOR_SOURCE", "auto")
SENSOR_BAUD_RATE = int(os.getenv("SENSOR_BAUD_RATE", "9600"))
SENSOR_INTERVAL_SECONDS = float(os.getenv("SENSOR_INTERVAL_SECONDS", "2.0"))

# ── Detection thresholds (preserved from original components) ─
# Component 3 used 0.5 in the live script and 0.35 in its dashboard;
# 0.5 is now the default (40-55% Drowning boxes were mostly false alarms).
DROWNING_CONF_THRESHOLD = float(os.getenv("DROWNING_CONF_THRESHOLD", "0.5"))
# Expressed in camera frames; app.py converts it to detector runs.
DROWNING_CONSEC_FRAMES = int(os.getenv("DROWNING_CONSEC_FRAMES", "10"))
# best.pt (YOLOv8s) was trained at 640, so drowning gets its own imgsz
# instead of the shared DETECTION_IMGSZ.
DROWNING_IMGSZ = int(os.getenv("DROWNING_IMGSZ", "640"))

CROWD_CONF_THRESHOLD = float(os.getenv("CROWD_CONF_THRESHOLD", "0.4"))
GARBAGE_CONF_THRESHOLD = float(os.getenv("GARBAGE_CONF_THRESHOLD", "0.4"))

# ── Crowd density / maintenance thresholds (Component 1 design) ─
CROWD_DENSITY_MODERATE = int(os.getenv("CROWD_DENSITY_MODERATE", "4"))
CROWD_DENSITY_HIGH = int(os.getenv("CROWD_DENSITY_HIGH", "8"))
# Bather-load (swimmer-minutes) above which maintenance is recommended.
MAINTENANCE_LOAD_THRESHOLD = float(os.getenv("MAINTENANCE_LOAD_THRESHOLD", "600"))
# Pool capacity standard (water surface area per bather).
POOL_AREA_M2 = float(os.getenv("POOL_AREA_M2", "250"))
AREA_PER_BATHER_M2 = float(os.getenv("AREA_PER_BATHER_M2", "2.7"))


def _parse_pool_polygon(raw: str):
    """Space-separated "x,y" pairs in relative 0.0-1.0 coordinates, e.g.
    "0.2,0.3 0.85,0.3 0.9,0.95 0.1,0.95" -> [(0.2,0.3), (0.85,0.3), ...].
    Returns None (no ROI filtering) if the env var is unset/empty."""
    raw = (raw or "").strip()
    if not raw:
        return None
    points = []
    for pair in raw.split():
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    return points or None


# Pool region (ROI) that swimmer detections are counted within — see
# CrowdDetector.is_in_pool(). None means count every detected person
# (no ROI filtering), matching the previous behaviour.
POOL_POLYGON = _parse_pool_polygon(os.getenv("POOL_POLYGON", ""))

# ── Accounts & subscription plans ────────────────────────────
# Each logged-in user's plan ("basic"/"pro"/"plus", see core/plans.py)
# decides which features they get. DEMO_MODE=1 skips the login
# requirement and falls back to ACTIVE_PLAN (CLI / offline demo).
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
ACTIVE_PLAN = os.getenv("ACTIVE_PLAN", "plus")
# ── Billing: PayHere recurring subscriptions (Pro / Plus, USD monthly) ─
# Real values only in .env. Without merchant id + secret + PUBLIC_BASE_URL
# the upgrade buttons say "Billing not configured — ask admin" (admins
# can still set plans by hand on /admin/users).
PAYHERE_MERCHANT_ID = os.getenv("PAYHERE_MERCHANT_ID", "").strip()
PAYHERE_MERCHANT_SECRET = os.getenv("PAYHERE_MERCHANT_SECRET", "").strip()
PAYHERE_SANDBOX = os.getenv("PAYHERE_SANDBOX", "1") == "1"
PAYHERE_CHECKOUT_URL = (
    "https://sandbox.payhere.lk/pay/checkout" if PAYHERE_SANDBOX
    else "https://www.payhere.lk/pay/checkout"
)
# Public https base URL PayHere can reach for notify_url (e.g. ngrok)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
BILLING_CONFIGURED = bool(PAYHERE_MERCHANT_ID and PAYHERE_MERCHANT_SECRET and PUBLIC_BASE_URL)
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "backend" / "data"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "app.db"))

# ── Server ───────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
# Signs the login session cookie — set a long random value in .env.
# Without one, a random key is generated per run (everyone is logged
# out on restart).
SECRET_KEY = os.getenv("SECRET_KEY", "")
SECRET_KEY_GENERATED = not SECRET_KEY
if SECRET_KEY_GENERATED:
    import secrets

    SECRET_KEY = secrets.token_hex(32)

# Frame processing
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "60"))
MAX_ALERT_HISTORY = int(os.getenv("MAX_ALERT_HISTORY", "10"))
# Inference resolution passed to all 3 vision detectors (crowd, drowning,
# garbage). Smaller = faster YOLO inference at some accuracy cost.
DETECTION_IMGSZ = int(os.getenv("DETECTION_IMGSZ", "480"))
