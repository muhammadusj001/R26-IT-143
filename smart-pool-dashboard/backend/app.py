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

# Mount the four component repos (git submodules) onto the import path
_BASE = Path(__file__).resolve().parents[1]
for _repo in ("component1-crowd-maintenance", "component2-water-quality",
              "component3-drowning-detection", "component4-garbage-detection"):
    sys.path.insert(0, str(_BASE / "components" / _repo))

import cv2
from flask import Flask, render_template, send_file, jsonify, after_this_request
from flask_socketio import SocketIO, emit

from config import settings
from core.camera import CameraManager
from core.state import state
from core.decision_engine import decision_engine
from core.plans import has_feature, plan_payload, required_plan, resolve_plan_key
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

# ── Subscription plan (feature gating is enforced here, not in the UI) ─
ACTIVE_PLAN = resolve_plan_key(settings.ACTIVE_PLAN)
state.update_root({"plan": plan_payload(ACTIVE_PLAN)})

# The vision loop below staggers crowd/drowning/garbage across processed
# frames as a CPU optimisation — see vision_loop(). With every module
# unlocked the cycle is 6 processed frames (stage = processed_num % 6):
#   0 crowd | 1 drowning | 2 drowning | 3 crowd | 4 drowning | 5 garbage
# i.e. crowd 2/6 (1 in 3), drowning 3/6, garbage 1/6. Modules locked by
# the active plan are removed from the cycle, so only enabled modules
# take turns:
#   pro   (no garbage)           -> crowd, drowning, drowning, crowd, drowning
#   basic (no drowning/garbage)  -> crowd on every processed frame
_FULL_SCHEDULE = ("crowd", "drowning", "drowning", "crowd", "drowning", "garbage")
_SCHEDULE = tuple(m for m in _FULL_SCHEDULE if has_feature(ACTIVE_PLAN, m))
_DROWNING_ENABLED = "drowning" in _SCHEDULE
# DROWNING_CONSEC_FRAMES is expressed in camera frames. Drowning runs
# once every PROCESS_EVERY_N_FRAMES * cycle / runs camera frames, so
# convert to "detector runs" covering about the same real time (min 5,
# which is also the detector's minimum smoothing window).
_drowning_consec_frames = 5
if _DROWNING_ENABLED:
    _CAMERA_FRAMES_PER_DROWNING_RUN = (
        settings.PROCESS_EVERY_N_FRAMES * len(_SCHEDULE) / _SCHEDULE.count("drowning")
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

        # ── CPU optimisation: stagger the 3 vision models ─────
        # Running crowd + drowning + garbage detection on every single
        # processed frame is the main FPS cost. Instead, each enabled
        # module gets its own turns in the cycle (see _SCHEDULE); modules
        # locked by the plan never run — each module still runs
        # completely independently, just less often.
        # On the frames where a module doesn't run, its previous result
        # simply isn't touched, so state.snapshot() keeps returning its
        # last known values and the dashboard card never goes blank.
        # Drowning boxes are redrawn on its off-frames (draw_last) so the
        # overlay doesn't flicker.
        module = _SCHEDULE[processed_num % len(_SCHEDULE)] if _SCHEDULE else None
        drown_alert = None
        garbage_alert = None

        if module == "crowd":
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
        elif module == "drowning":
            frame, drown_result, drown_alert = drowning.process(frame)
            state.update_module("drowning", drown_result)
        elif module == "garbage":
            frame, garbage_result, garbage_alert = garbage.process(frame)
            state.update_module("garbage", garbage_result)

        if _DROWNING_ENABLED and module != "drowning":
            frame = drowning.draw_last(frame)

        if drown_alert:
            drown_message = f"DROWNING DETECTED — {drown_alert['count']} person(s)"
            state.add_alert("drowning", drown_message, "danger")
            session.record_alert("drowning", drown_message, "danger")
            socketio.emit("drowning_alert", drown_alert)
        if garbage_alert:
            state.add_alert("garbage", garbage_alert, "warning")
            session.record_alert("garbage", garbage_alert, "warning")

        # ── Cross-module intelligence (Decision Engine only) ─
        if has_feature(ACTIVE_PLAN, "decision_engine"):
            decision = decision_engine.evaluate(state.snapshot())
        else:
            decision = decision_engine.neutral()
        state.update_module("decision", decision)

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


@app.route("/api/session/summary")
def session_summary():
    return session.get_summary()


@app.route("/api/report")
def download_report():
    if not has_feature(ACTIVE_PLAN, "reports"):
        return jsonify({"error": "upgrade_required",
                        "required_plan": required_plan("reports")}), 403
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
    print(f"  Plan          : {ACTIVE_PLAN} ({', '.join(plan_payload(ACTIVE_PLAN)['features'])})")
    # Models for features locked by the plan are never run, so skip loading them
    for name, detector in (("Crowd", crowd), ("Drowning", drowning), ("Garbage", garbage)):
        if has_feature(ACTIVE_PLAN, name.lower()):
            detector.load()
            print(f"  {name + ' model':<14}: {detector.model_status}")
        else:
            print(f"  {name + ' model':<14}: locked by plan")
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
