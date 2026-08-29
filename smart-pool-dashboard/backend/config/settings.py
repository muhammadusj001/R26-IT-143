"""
Central configuration for the Smart Pool Monitoring System.

All machine-specific values (paths, camera source, serial port) live here
and can be overridden with environment variables. This replaces the
hardcoded Windows paths, phone IPs and serial ports that previously
existed inside each component's scripts.
"""

import os
from pathlib import Path

# ── Project paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
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
    os.getenv("CROWD_MODEL_PATH",
              COMPONENTS_DIR / "component1-crowd-maintenance" / "models" / "best_swimmer_model.pt")
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
#   an integer index ("0", "1")  -> local/Iriun webcam
#   a stream URL ("http://<phone-ip>:8080/video") -> IP Webcam app
#   "simulate" -> synthetic frames (no camera needed, demo mode)
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "simulate")

# ── Water quality sensor source ──────────────────────────────
# Serial port of the Arduino (e.g. "COM3", "/dev/cu.usbmodem21401")
# or "simulate" to generate realistic sensor values without hardware.
SENSOR_SOURCE = os.getenv("SENSOR_SOURCE", "simulate")
SENSOR_BAUD_RATE = int(os.getenv("SENSOR_BAUD_RATE", "9600"))
SENSOR_INTERVAL_SECONDS = float(os.getenv("SENSOR_INTERVAL_SECONDS", "2.0"))

# ── Detection thresholds (preserved from original components) ─
# Component 3 used 0.5 in the live script and 0.35 in its dashboard;
# the dashboard value is kept as the runtime default and is now tunable.
DROWNING_CONF_THRESHOLD = float(os.getenv("DROWNING_CONF_THRESHOLD", "0.35"))
DROWNING_CONSEC_FRAMES = int(os.getenv("DROWNING_CONSEC_FRAMES", "3"))

CROWD_CONF_THRESHOLD = float(os.getenv("CROWD_CONF_THRESHOLD", "0.4"))
GARBAGE_CONF_THRESHOLD = float(os.getenv("GARBAGE_CONF_THRESHOLD", "0.4"))

# ── Crowd density / maintenance thresholds (Component 1 design) ─
CROWD_DENSITY_MODERATE = int(os.getenv("CROWD_DENSITY_MODERATE", "4"))
CROWD_DENSITY_HIGH = int(os.getenv("CROWD_DENSITY_HIGH", "8"))
# Bather-load (swimmer-minutes) above which maintenance is recommended.
MAINTENANCE_LOAD_THRESHOLD = float(os.getenv("MAINTENANCE_LOAD_THRESHOLD", "600"))

# ── Server ───────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
SECRET_KEY = os.getenv("SECRET_KEY", "smart_pool_r26_it_143")

# Frame processing
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
MAX_ALERT_HISTORY = int(os.getenv("MAX_ALERT_HISTORY", "10"))
