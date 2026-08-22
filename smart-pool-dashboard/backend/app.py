"""
R26-IT-143 — AI-Based Smart Swimming Pool Monitoring System
Unified backend: ONE dashboard, FOUR independent AI modules.

Pipeline per camera frame (shared stream):
    frame -> CrowdDetector -> DrowningDetector -> GarbageDetector
Each module draws its own overlays on the same frame and writes its own
results into SystemState. WaterQualityMonitor runs independently on its
own thread. The DecisionEngine fuses all module outputs into
system-level intelligence. Flask-SocketIO (the stack Component 3
already used) pushes everything to the single dashboard in real time.
"""

import base64
import sys
import threading
import time
from pathlib import Path

# Mount the four component repos (git submodules) onto the import path
_BASE = Path(__file__).resolve().parents[1]
for _repo in ("component1-crowd-maintenance", "component2-water-quality",
              "component3-drowning-detection", "component4-garbage-detection"):
    sys.path.insert(0, str(_BASE / "components" / _repo))

import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from config import settings
from core.camera import CameraManager
from core.state import state
from core.decision_engine import decision_engine
from component1_crowd.detector import CrowdDetector
from component3_drowning.detector import DrowningDetector
from component4_garbage.detector import GarbageDetector
from core.water_monitor import WaterQualityMonitor

app = Flask(
    __name__,
    template_folder=str(settings.BASE_DIR / "frontend-vanilla" / "templates"),
    static_folder=str(settings.BASE_DIR / "frontend-vanilla" / "static"),
)
app.config["SECRET_KEY"] = settings.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

camera = CameraManager()
crowd = CrowdDetector(model_path=settings.CROWD_MODEL_PATH,
                      conf_threshold=settings.CROWD_CONF_THRESHOLD,
                      density_moderate=settings.CROWD_DENSITY_MODERATE,
                      density_high=settings.CROWD_DENSITY_HIGH,
                      pool_area_m2=settings.POOL_AREA_M2,
                      area_per_bather_m2=settings.AREA_PER_BATHER_M2)
drowning = DrowningDetector(model_path=settings.DROWNING_MODEL_PATH,
                            conf_threshold=settings.DROWNING_CONF_THRESHOLD,
                            consec_frames=settings.DROWNING_CONSEC_FRAMES)
garbage = GarbageDetector(model_path=settings.GARBAGE_MODEL_PATH,
                          conf_threshold=settings.GARBAGE_CONF_THRESHOLD)
water = WaterQualityMonitor(
    on_update=lambda: socketio.emit("state_update", state.snapshot())
)

_vision_thread = None
_last_crowd_density = None


def vision_loop():
    """Shared camera loop: one frame feeds all three vision modules."""
    global _last_crowd_density
    if not camera.open():
        state.update_root({"running": False, "camera_connected": False})
        socketio.emit("camera_error", {"message": f"Cannot open camera: {camera.source}"})
        return

    state.update_root({"running": True, "camera_connected": True})
    start = time.time()
    frame_num = 0

    while state.data["running"]:
        ok, frame = camera.read()
        if not ok:
            state.update_root({"camera_connected": False})
            socketio.emit("camera_error", {"message": "Camera lost — reconnecting"})
            if not camera.reconnect():
                continue
            state.update_root({"camera_connected": True})
            continue

        frame_num += 1
        # Frame skipping preserved from Component 3 (FPS optimisation)
        if frame_num % settings.PROCESS_EVERY_N_FRAMES != 0:
            continue

        # ── Independent modules, same frame ──────────────────
        frame, crowd_result = crowd.process(frame)
        frame, drown_result, drown_alert = drowning.process(frame)
        frame, garbage_result, garbage_alert = garbage.process(frame)

        state.update_module("crowd", crowd_result)
        state.update_module("drowning", drown_result)
        state.update_module("garbage", garbage_result)

        crowd_density = crowd_result.get("density_level")
        crowd_alert = crowd_result.get("crowd_alert")
        if crowd_alert and crowd_density != _last_crowd_density:
            state.add_alert(
                "crowd",
                crowd_alert,
                "danger" if crowd_density == "OVER CAPACITY" else "warning",
            )
        _last_crowd_density = crowd_density

        if drown_alert:
            state.add_alert(
                "drowning",
                f"DROWNING DETECTED — {drown_alert['count']} person(s)",
                "danger",
            )
            socketio.emit("drowning_alert", drown_alert)
        if garbage_alert:
            state.add_alert("garbage", garbage_alert, "warning")

        # ── Cross-module intelligence (Decision Engine only) ─
        state.update_module("decision", decision_engine.evaluate(state.snapshot()))

        # ── Session stats ────────────────────────────────────
        elapsed = time.time() - start
        fps = frame_num / elapsed if elapsed > 0 else 0
        state.update_root({
            "frame_num": frame_num,
            "fps": round(fps, 1),
            "session_time": f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
        })

        # ── Push frame + full state to dashboard ─────────────
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
        )
        socketio.emit("frame_update", {
            "frame": base64.b64encode(buffer).decode("utf-8"),
            "state": state.snapshot(),
        })
        time.sleep(0.02)

    camera.release()


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return state.snapshot()


# ── Socket events ────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("state_update", state.snapshot())


@socketio.on("start_detection")
def start_detection():
    global _vision_thread
    if not state.data["running"]:
        _vision_thread = threading.Thread(target=vision_loop, daemon=True)
        _vision_thread.start()
        emit("detection_started", {"message": "Detection started"})


@socketio.on("stop_detection")
def stop_detection():
    state.update_root({"running": False})
    emit("detection_stopped", {"message": "Detection stopped"})


def main():
    print("=" * 60)
    print("  R26-IT-143 — SMART POOL MONITORING (UNIFIED)")
    print("=" * 60)
    print(f"  Camera source : {settings.CAMERA_SOURCE}")
    print(f"  Sensor source : {settings.SENSOR_SOURCE}")
    crowd.load()
    drowning.load()
    garbage.load()
    print(f"  Crowd model   : {crowd.model_status}")
    print(f"  Drowning model: {drowning.model_status}")
    print(f"  Garbage model : {garbage.model_status}")
    water.start()
    print(f"  Water model   : {water.predictor.model_status}")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:{settings.PORT}")
    socketio.run(
        app, host=settings.HOST, port=settings.PORT,
        debug=False, allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
