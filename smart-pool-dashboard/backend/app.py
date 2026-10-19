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

Accounts: every viewer logs in and has their own plan (core/plans.py).
The vision loop runs the modules unlocked for the user who pressed
Start; everything sent to a viewer (state, alerts, video overlays,
reports) is filtered for that viewer's own plan on the server.
"""

import base64
import logging
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
from flask import Flask, render_template, send_file, jsonify, after_this_request, redirect, request
from flask_socketio import SocketIO, emit, ConnectionRefusedError
from werkzeug.middleware.proxy_fix import ProxyFix

from config import settings
from core import auth, billing, db
from core.auth import current_plan, current_user, login_required
from core.camera import CameraManager
from core.state import state
from core.decision_engine import decision_engine
from core.gating import filter_state, filter_summary
from core.plans import has_feature, required_plan, resolve_plan_key
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
# Behind ngrok / VS Code tunnels TLS ends at the proxy; honour its
# X-Forwarded-Proto so request.host_url keeps "https" (billing return URLs)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)
app.config.update(
    SECRET_KEY=settings.SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
app.register_blueprint(auth.bp)
app.register_blueprint(billing.bp)


@app.before_request
def _redirect_127_to_localhost():
    """127.0.0.1 and localhost get separate login cookies, so send local
    visitors to one canonical host: http://127.0.0.1:5000/x?y -> http://localhost:5000/x?y.
    Only the exact 127.0.0.1 host is affected (ngrok, VS Code tunnels and LAN
    IPs pass through), and only GET/HEAD: a 302 would turn a POST into a GET
    and drop its body. Socket.IO traffic is left alone."""
    hostname, sep, port = request.host.partition(":")
    if hostname != "127.0.0.1" or request.method not in ("GET", "HEAD") \
            or request.path.startswith(("/socket.io", "/billing/notify")):
        return None
    target = f"{request.scheme}://localhost{sep}{port}{request.path}"
    if request.query_string:
        target += "?" + request.query_string.decode("latin-1")
    return redirect(target, code=302)
db.init_db()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# The vision loop below staggers crowd/drowning/garbage across processed
# frames as a CPU optimisation — see vision_loop(). With every module
# unlocked the cycle is 6 processed frames (stage = processed_num % 6):
#   0 crowd | 1 drowning | 2 drowning | 3 crowd | 4 drowning | 5 garbage
# i.e. crowd 2/6 (1 in 3), drowning 3/6, garbage 1/6. Modules locked by
# the plan of the user who pressed Start are removed from the cycle, so
# only enabled modules take turns:
#   pro   (no garbage)           -> crowd, drowning, drowning, crowd, drowning
#   basic (no drowning/garbage)  -> crowd on every processed frame
_FULL_SCHEDULE = ("crowd", "drowning", "drowning", "crowd", "drowning", "garbage")
# C4_FULL_RATE=1: Component 4 runs on EVERY processed frame (throw / drop
# detection needs to see the arm move several times per second); crowd and
# drowning keep their own turns above. Only when the plan includes garbage.
_C4_FULL_RATE = os.getenv("C4_FULL_RATE", "0") == "1"
# DROWNING_CONSEC_FRAMES is expressed in camera frames. Drowning runs
# once every PROCESS_EVERY_N_FRAMES * cycle / runs camera frames, so
# convert to "detector runs" covering about the same real time (min 5,
# which is also the detector's minimum smoothing window). Computed for
# the full schedule; with garbage locked drowning runs slightly more often.
_CAMERA_FRAMES_PER_DROWNING_RUN = (
    settings.PROCESS_EVERY_N_FRAMES * len(_FULL_SCHEDULE) / _FULL_SCHEDULE.count("drowning")
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


# ── Connected viewers (one entry per Socket.IO connection) ───
_viewers = {}  # sid -> {"user_id": int | None, "plan": str}
_viewers_lock = threading.Lock()


def _viewer_groups():
    """{plan_key: [sid, ...]} for everyone currently connected."""
    groups = {}
    with _viewers_lock:
        for sid, viewer in _viewers.items():
            groups.setdefault(viewer["plan"], []).append(sid)
    return groups


def _overlay_tier(plan_key):
    """Which video overlays a viewer may see. Plans are nested
    (basic ⊂ pro ⊂ plus), so three frame variants cover every viewer."""
    if has_feature(plan_key, "garbage"):
        return "plus"
    if has_feature(plan_key, "drowning"):
        return "pro"
    return "basic"


def broadcast_state():
    """Send every viewer the shared state filtered for their own plan."""
    snap = state.snapshot()
    for plan_key, sids in _viewer_groups().items():
        payload = filter_state(snap, plan_key)
        for sid in sids:
            socketio.emit("state_update", payload, to=sid)


def _on_plan_change(user_id):
    """Admin changed a plan: re-gate that user's open dashboards now."""
    plan_key = auth.plan_for(db.get_user(user_id))
    with _viewers_lock:
        for viewer in _viewers.values():
            if viewer["user_id"] == user_id:
                viewer["plan"] = plan_key
    broadcast_state()


auth.plan_change_listeners.append(_on_plan_change)

water = WaterQualityMonitor(on_update=broadcast_state, session_tracker=session)

_vision_thread = None
_last_crowd_density = None


def vision_loop(plan_key=None):
    """Shared camera loop: one frame feeds all three vision modules.

    plan_key is the plan of the user who pressed Start; without one
    (CLI / system_check / DEMO_MODE) ACTIVE_PLAN is used."""
    global _last_crowd_density
    plan_key = resolve_plan_key(plan_key or settings.ACTIVE_PLAN)
    schedule = tuple(m for m in _FULL_SCHEDULE if has_feature(plan_key, m))
    drowning_enabled = "drowning" in schedule
    garbage_enabled = "garbage" in schedule
    decision_enabled = has_feature(plan_key, "decision_engine")

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
        # splash / pool-floor analysis; also the overlay-free view.
        clean = frame.copy()

        # ── CPU optimisation: stagger the 3 vision models ─────
        # Running crowd + drowning + garbage detection on every single
        # processed frame is the main FPS cost. Instead, each enabled
        # module gets its own turns in the cycle (see _FULL_SCHEDULE);
        # modules locked by the plan never run — each module still runs
        # completely independently, just less often.
        # On the frames where a module doesn't run, its previous result
        # simply isn't touched, so state.snapshot() keeps returning its
        # last known values and the dashboard card never goes blank.
        # Drowning boxes are redrawn on its off-frames (draw_last) so the
        # overlay doesn't flicker.
        module = schedule[processed_num % len(schedule)] if schedule else None
        drown_alert = None
        garbage_alert = None

        # Viewers on lower plans get a frame without locked overlays.
        groups = _viewer_groups()
        tiers = {_overlay_tier(p) for p in groups}
        run_garbage = garbage_enabled and (module == "garbage" or _C4_FULL_RATE)

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

        # Crowd overlays only (crowd frames) or no overlays (other frames)
        crowd_only = None
        if "basic" in tiers:
            crowd_only = frame.copy() if module == "crowd" else clean

        # Pro viewers must not see garbage boxes: keep the frame as it was
        # before Component 4 draws on it.
        no_garbage = None
        if run_garbage:
            if "pro" in tiers:
                no_garbage = frame.copy()
            frame, garbage_result, garbage_alert = garbage.process(frame, clean_frame=clean)
            state.update_module("garbage", garbage_result)

        if drowning_enabled and module != "drowning":
            frame = drowning.draw_last(frame)
            if no_garbage is not None:
                no_garbage = drowning.draw_last(no_garbage)

        views = {"plus": frame, "basic": crowd_only}
        if "pro" in tiers:
            views["pro"] = no_garbage if no_garbage is not None else frame

        if drown_alert:
            drown_message = f"DROWNING DETECTED — {drown_alert['count']} person(s)"
            state.add_alert("drowning", drown_message, "danger")
            session.record_alert("drowning", drown_message, "danger")
            for viewer_plan, sids in groups.items():
                if has_feature(viewer_plan, "drowning"):
                    for sid in sids:
                        socketio.emit("drowning_alert", drown_alert, to=sid)
        if garbage_alert:
            garbage_level = "danger" if garbage_alert.startswith("[HIGH]") else "warning"
            state.add_alert("garbage", garbage_alert, garbage_level)
            session.record_alert("garbage", garbage_alert, garbage_level)

        # ── Cross-module intelligence (Decision Engine only) ─
        if decision_enabled:
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

        # ── Push frame + state to each viewer, filtered by plan ─
        encoded = {}
        for tier in tiers:
            _, buffer = cv2.imencode(
                ".jpg", views[tier], [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
            )
            encoded[tier] = base64.b64encode(buffer).decode("utf-8")
        snap = state.snapshot()
        for viewer_plan, sids in groups.items():
            payload = {"frame": encoded[_overlay_tier(viewer_plan)],
                       "state": filter_state(snap, viewer_plan)}
            for sid in sids:
                socketio.emit("frame_update", payload, to=sid)
        time.sleep(0.02)

    camera.release()


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/api/status")
@login_required
def status():
    return filter_state(state.snapshot(), current_plan())


def _garbage_locked():
    """403 response when the viewer's plan doesn't include garbage, else None."""
    if has_feature(current_plan(), "garbage"):
        return None
    return jsonify({"error": "upgrade_required", "required_plan": required_plan("garbage")}), 403


@app.route("/api/garbage/snapshot/<path:name>")
@login_required
def garbage_snapshot(name):
    """Snapshot photo saved by Component 4 for an incident (event_feed[].snapshot)."""
    from flask import send_from_directory
    locked = _garbage_locked()
    if locked:
        return locked
    folder = getattr(getattr(garbage, "logger", None), "dir", None)
    if folder is None:
        return jsonify({"error": "no snapshots"}), 404
    return send_from_directory(str(folder), name)


@app.route("/api/garbage/events")
@login_required
def garbage_events():
    """Last Component 4 incidents in plain language (for the dashboard list)."""
    locked = _garbage_locked()
    if locked:
        return locked
    snap = state.snapshot().get("garbage", {})
    return jsonify({"event_feed": snap.get("event_feed", []),
                    "event_legend": snap.get("event_legend", {})})


@app.route("/api/session/summary")
@login_required
def session_summary():
    return filter_summary(session.get_summary(), current_plan())


@app.route("/api/report")
@login_required
def download_report():
    plan_key = current_plan()
    if not has_feature(plan_key, "reports"):
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
        summary = filter_summary(summary, plan_key)

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
def on_connect(auth_data=None):
    # Same login session cookie as the HTTP routes
    plan_key = current_plan()
    if plan_key is None:
        raise ConnectionRefusedError("login_required")
    user = current_user()
    with _viewers_lock:
        _viewers[request.sid] = {"user_id": user["id"] if user else None, "plan": plan_key}
    emit("state_update", filter_state(state.snapshot(), plan_key))


@socketio.on("disconnect")
def on_disconnect(*_args):
    with _viewers_lock:
        _viewers.pop(request.sid, None)


@socketio.on("start_detection")
def start_detection():
    global _vision_thread
    if not state.data["running"]:
        with _viewers_lock:
            plan_key = _viewers.get(request.sid, {}).get("plan")
        session.start()
        _vision_thread = threading.Thread(target=vision_loop, args=(plan_key,), daemon=True)
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
    # Broadcast so every connected dashboard reflects the test starting,
    # matching how regular water readings are pushed via water.on_update.
    broadcast_state()


def main():
    print("=" * 60)
    print("  R26-IT-143 — SMART POOL MONITORING (UNIFIED)")
    print("=" * 60)
    print(f"  Camera source : {settings.CAMERA_SOURCE} -> {camera.source} ({camera.type})")
    print(f"  Sensor source : {settings.SENSOR_SOURCE}")
    if settings.DEMO_MODE:
        print(f"  Accounts      : DEMO_MODE — no login required, plan '{settings.ACTIVE_PLAN}'")
    else:
        print(f"  Accounts      : login required ({len(db.list_users())} users in {settings.DB_PATH.name})")
        if not any(u["is_admin"] for u in db.list_users()):
            print("  No admin yet  : run 'python manage.py create-admin'")
    if settings.SECRET_KEY_GENERATED:
        print("  WARNING: SECRET_KEY not set in .env — using a random key; "
              "everyone is logged out on restart")
    # Each user has their own plan, so every model may be needed
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
