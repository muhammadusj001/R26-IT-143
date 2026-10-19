"""
Component 4 alone behind a tiny web server (live stream + JSON), for testing
the same way the dashboard consumes it. The REAL team dashboard
(smart-pool-dashboard/backend/app.py) already has its own shared camera loop;
how to plug v2 into it is in INTEGRATION.md, not here.

    python integration_example.py --source 1
then open http://127.0.0.1:5004  (live annotated stream + JSON at /status)
"""

import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify

from component4_garbage.camera import LiveCamera
from component4_garbage.detector import GarbageDetector

# ---------- loop (the dashboard does the same inside its vision_loop) --
_latest = {"jpg": None, "result": {}, "alerts": []}


def run_component4(source, emit=None):
    """Background loop. `emit` = socketio.emit in the integrated backend."""
    det = GarbageDetector(show_hud=True)   # uses models/ + config/pool_roi.json
    det.load()
    cam = LiveCamera(source)
    try:
        while True:
            ok, frame, _ = cam.read()
            if not ok:
                time.sleep(0.5)
                continue
            frame, result, alert = det.process(frame)
            ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok_jpg:
                _latest["jpg"] = buf.tobytes()
            _latest["result"] = result
            if emit is not None:
                emit("c4_status", result)                 # every frame: small JSON
                for ev in result["events"]:               # only NEW events
                    emit("c4_event", ev)
            if alert:
                _latest["alerts"] = ([alert] + _latest["alerts"])[:20]
    finally:
        cam.release()
        det.close()
# ------------------------------------------------------------------------


app = Flask(__name__)


@app.route("/")
def index():
    return ('<h3>Component 4 live</h3><img src="/video" style="max-width:100%">'
            '<p><a href="/status">/status (JSON)</a></p>')


@app.route("/video")
def video():
    def gen():
        while True:
            if _latest["jpg"]:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _latest["jpg"] + b"\r\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify({"result": _latest["result"], "recent_alerts": _latest["alerts"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--port", type=int, default=5004)
    a = ap.parse_args()
    threading.Thread(target=run_component4, args=(a.source,), daemon=True).start()
    app.run(host="0.0.0.0", port=a.port, threaded=True)
