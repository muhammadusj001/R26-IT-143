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
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

# Mount the four component repos (git submodules) onto the import path.
# Folder names may carry a suffix (e.g. "component4-garbage-detection V2.2")
# or be nested one level after unzipping; find the folder that really
# contains the Python package. For Component 4, prefer the v2 copy
# (the one with component4_garbage/submersion.py).
_BASE = Path(__file__).resolve().parents[1]
_COMPONENTS = _BASE / "components"
_REPOS = {
    "component1-crowd-maintenance": ("component1_crowd", None),
    "component2-water-quality": (None, None),
    "component3-drowning-detection": ("component3_drowning", None),
    "component4-garbage-detection": ("component4_garbage", "submersion.py"),
}


def _find_repo(name, package, prefer_file):
    candidates = []
    for p in sorted(_COMPONENTS.glob(name + "*")):
        if p.is_dir():
            candidates += [p] + [c for c in p.iterdir() if c.is_dir()]
    if package:
        with_pkg = [c for c in candidates if (c / package / "__init__.py").exists()
                    or (c / package).is_dir()]
        if prefer_file:
            best = [c for c in with_pkg if (c / package / prefer_file).exists()]
            if best:
                return best[0]
        if with_pkg:
            return with_pkg[0]
    return _COMPONENTS / name


for _repo, (_pkg, _prefer) in _REPOS.items():
    _path = _find_repo(_repo, _pkg, _prefer)
    sys.path.insert(0, str(_path))
    if _repo.startswith("component4"):
        print(f"  Component 4 folder: {_path}")

import cv2
from flask import Flask, render_template, send_file, jsonify, after_this_request
from flask_socketio import SocketIO, emit

from config import settings
from core.camera import CameraManager
from core.state import state
from core.decision_engine import decision_engine
from core.session_tracker import SessionTracker
from core.report_generator import generate_session_report
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

# The vision loop below staggers crowd/drowning/garbage across processed
# frames as a CPU optimisation — see vision_loop(). Schedule over a
# 6-processed-frame cycle (stage = processed_num % 6):
#   0 crowd | 1 drowning | 2 drowning | 3 crowd | 4 drowning | 5 garbage
# i.e. crowd 2/6 (unchanged, 1 in 3), drowning 3/6, garbage 1/6.
_STAGGER_CYCLE = 6
_CROWD_STAGES = (0, 3)
_DROWNING_STAGES = (1, 2, 4)
_GARBAGE_STAGES = (5,)
# C4_FULL_RATE=1: Component 4 runs on EVERY processed frame (throw / drop
# detection needs to see the arm move several times per second); crowd and
# drowning keep their own stages above.
_C4_FULL_RATE = os.getenv("C4_FULL_RATE", "0") == "1"
# DROWNING_CONSEC_FRAMES is expressed in camera frames. Drowning runs
# once every PROCESS_EVERY_N_FRAMES * cycle / runs camera frames, so
# convert to "detector runs" covering about the same real time (min 5,
# which is also the detector's minimum smoothing window).
_CAMERA_FRAMES_PER_DROWNING_RUN = (
    settings.PROCESS_EVERY_N_FRAMES * _STAGGER_CYCLE / len(_DROWNING_STAGES)
)
_drowning_consec_frames = max(
    5, round(settings.DROWNING_CONSEC_FRAMES / _CAMERA_FRAMES_PER_DROWNING_RUN)
)

camera = CameraManager()
crowd = CrowdDetector(model_path=settings.CROWD_MODEL_PATH,
                      conf_threshold=settings.CROWD_CONF_THRESHOLD,
                      density_moderate=settings.CROWD_DENSITY_MODERATE,
                      density_high=settings.CROWD_DENSITY_HIGH,
                      pool_area_m2=settings.POOL_AREA_M2,
                      area_per_bather_m2=settings.AREA_PER_BATHER_M2,
                      imgsz=settings.DETECTION_IMGSZ,
                      pool_polygon=settings.POOL_POLYGON)
drowning = DrowningDetector(model_path=settings.DROWNING_MODEL_PATH,
                            conf_threshold=settings.DROWNING_CONF_THRESHOLD,
                            consec_frames=_drowning_consec_frames,
                            imgsz=settings.DROWNING_IMGSZ)
garbage = GarbageDetector(model_path=settings.GARBAGE_MODEL_PATH,
                          conf_threshold=settings.GARBAGE_CONF_THRESHOLD,
                          imgsz=settings.DETECTION_IMGSZ)
session = SessionTracker()
water = WaterQualityMonitor(
    on_update=lambda: socketio.emit("state_update", state.snapshot()),
    session_tracker=session,
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

    state.update_root({"running": True, "camera_connected": True, "camera": camera.status()})
    start = time.time()
    frame_num = 0
    processed_num = 0

    while state.data["running"]:
        ok, frame = camera.read()
        if not ok:
            # Reconnection retries forever with backoff; only notify the
            # dashboard once per outage instead of on every attempt.
            if state.data["camera_connected"]:
                state.update_root({"camera_connected": False, "camera": camera.status()})
                socketio.emit("camera_error", {"message": "Camera lost — reconnecting"})
            if not camera.reconnect():
                continue
            state.update_root({"camera_connected": True, "camera": camera.status()})
            continue
        if frame is None:
            # OpenCV can report a transient read failure with ok=True.
            # Do not send an empty frame into the detectors or imencode.
            continue

        frame_num += 1
        # Frame skipping preserved from Component 3 (FPS optimisation)
        if frame_num % settings.PROCESS_EVERY_N_FRAMES != 0:
            continue

        processed_num += 1
        # Clean copy (no boxes from Components 1/3) for Component 4's
        # splash / pool-floor analysis.
        raw = frame.copy()

        # ── CPU optimisation: stagger the 3 vision models ─────
        # Running crowd + drowning + garbage detection on every single
        # processed frame is the main FPS cost. Instead, each module
        # gets its own turns in a 6-frame cycle (see _STAGGER_CYCLE):
        # crowd on stages 0/3, drowning on 1/2/4, garbage on 5 — each
        # module still runs completely independently, just less often.
        # On the frames where a module doesn't run, its previous result
        # simply isn't touched, so state.snapshot() keeps returning its
        # last known values and the dashboard card never goes blank.
        # Drowning boxes are redrawn on its off-frames (draw_last) so the
        # overlay doesn't flicker.
        stage = processed_num % _STAGGER_CYCLE
        drown_alert = None
        garbage_alert = None

        if stage in _CROWD_STAGES:
            frame, crowd_result = crowd.process(frame)
            state.update_module("crowd", crowd_result)
            session.record_crowd(
                crowd_result.get("swimmer_count", 0),
                crowd_result.get("density_level"),
                crowd_result.get("bather_load", 0.0),
                crowd_result.get("recommendations"),
            )

            crowd_density = crowd_result.get("density_level")
            crowd_alert = crowd_result.get("crowd_alert")
            if crowd_alert and crowd_density != _last_crowd_density:
                crowd_severity = "danger" if crowd_density == "OVER CAPACITY" else "warning"
                state.add_alert("crowd", crowd_alert, crowd_severity)
                session.record_alert("crowd", crowd_alert, crowd_severity)
            _last_crowd_density = crowd_density
        elif stage in _DROWNING_STAGES:
            frame, drown_result, drown_alert = drowning.process(frame)
            state.update_module("drowning", drown_result)

        if stage in _GARBAGE_STAGES or _C4_FULL_RATE:
            frame, garbage_result, garbage_alert = garbage.process(frame, clean_frame=raw)
            state.update_module("garbage", garbage_result)

        if stage not in _DROWNING_STAGES:
            frame = drowning.draw_last(frame)

        if drown_alert:
            drown_message = f"DROWNING DETECTED — {drown_alert['count']} person(s)"
            state.add_alert("drowning", drown_message, "danger")
            session.record_alert("drowning", drown_message, "danger")
            socketio.emit("drowning_alert", drown_alert)
        if garbage_alert:
            garbage_level = "danger" if garbage_alert.startswith("[HIGH]") else "warning"
            state.add_alert("garbage", garbage_alert, garbage_level)
            session.record_alert("garbage", garbage_alert, garbage_level)

        # ── Cross-module intelligence (Decision Engine only) ─
        state.update_module("decision", decision_engine.evaluate(state.snapshot()))

        # ── Session stats ────────────────────────────────────
        elapsed = time.time() - start
        fps = frame_num / elapsed if elapsed > 0 else 0
        state.update_root({
            "frame_num": frame_num,
            "fps": round(fps, 1),
            "session_time": f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
            "camera": camera.status(),
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


@app.route("/api/garbage/snapshot/<path:name>")
def garbage_snapshot(name):
    """Snapshot photo saved by Component 4 for an incident (event_feed[].snapshot)."""
    from flask import send_from_directory
    folder = getattr(getattr(garbage, "logger", None), "dir", None)
    if folder is None:
        return jsonify({"error": "no snapshots"}), 404
    return send_from_directory(str(folder), name)


@app.route("/api/garbage/events")
def garbage_events():
    """Last Component 4 incidents in plain language (for the dashboard list)."""
    snap = state.snapshot().get("garbage", {})
    return jsonify({"event_feed": snap.get("event_feed", []),
                    "event_legend": snap.get("event_legend", {})})


@app.route("/api/session/summary")
def session_summary():
    return session.get_summary()


@app.route("/api/report")
def download_report():
    try:
        snap = state.snapshot()
        summary = session.get_summary()
        summary["frames_processed"] = snap.get("frame_num", 0)
        summary["model_status"] = {
            "crowd": snap.get("crowd", {}).get("model_status"),
            "drowning": snap.get("drowning", {}).get("model_status"),
            "garbage": snap.get("garbage", {}).get("model_status"),
            "water_quality": snap.get("water_quality", {}).get("model_status"),
        }

        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        generate_session_report(summary, tmp_path)

        @after_this_request
        def _cleanup(response):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return response

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            tmp_path,
            as_attachment=True,
            download_name=f"pool_session_report_{timestamp}.pdf",
            mimetype="application/pdf",
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to generate report: {exc}"}), 500


# ── Socket events ────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("state_update", state.snapshot())


@socketio.on("start_detection")
def start_detection():
    global _vision_thread
    if not state.data["running"]:
        session.start()
        _vision_thread = threading.Thread(target=vision_loop, daemon=True)
        _vision_thread.start()
        emit("detection_started", {"message": "Detection started"})


@socketio.on("stop_detection")
def stop_detection():
    state.update_root({"running": False})
    session.stop()
    emit("detection_stopped", {"message": "Detection stopped"})


@socketio.on("start_water_test")
def start_water_test():
    water.start_test()
    # Broadcast (not the request-scoped `emit`) so every connected
    # dashboard reflects the test starting, matching how regular water
    # readings are pushed via water.on_update.
    socketio.emit("state_update", state.snapshot())


def main():
    print("=" * 60)
    print("  R26-IT-143 — SMART POOL MONITORING (UNIFIED)")
    print("=" * 60)
    print(f"  Camera source : {settings.CAMERA_SOURCE} -> {camera.source} ({camera.type})")
    print(f"  Sensor source : {settings.SENSOR_SOURCE}")
    crowd.load()
    drowning.load()
    garbage.load()
    print(f"  Crowd model   : {crowd.model_status}")
    print(f"  Drowning model: {drowning.model_status}")
    print(f"  Garbage model : {garbage.model_status}")
    if hasattr(garbage, "status_summary"):
        print(f"  Component 4   : {garbage.status_summary()}")
        print(f"  C4 full rate  : {'ON (best for throw/drop)' if _C4_FULL_RATE else 'off (set C4_FULL_RATE=1 for throw/drop demo)'}")
        import atexit
        atexit.register(garbage.close)
    water.start()
    print(f"  Water model   : {water.predictor.model_status}")
    print("=" * 60)
    dash_url = f"http://localhost:{settings.PORT}"
    print(f"  Dashboard: {dash_url}")

    def auto_open():
        time.sleep(1.5)
        import webbrowser
        print(f"\n[+] Opening integrated dashboard at {dash_url} ...\n")
        webbrowser.open(dash_url)

    threading.Thread(target=auto_open, daemon=True).start()

    socketio.run(
        app, host=settings.HOST, port=settings.PORT,
        debug=False, allow_unsafe_werkzeug=True,
    )



if __name__ == "__main__":
    main()
