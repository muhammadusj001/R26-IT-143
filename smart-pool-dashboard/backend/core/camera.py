"""
Shared camera manager.

ONE stream is captured here and the same frame is handed to the crowd,
drowning and garbage modules. This replaces the previous design where
each component opened its own camera.

Supports (see normalise_source):
  - integer index ("0", "1", "3") -> webcam / DroidCam Client / OBS Virtual Camera
  - "droidcam:IP" or "droidcam:IP:PORT" -> DroidCam app over Wi-Fi (port 4747)
  - http/https/rtsp stream URL (e.g. "http://192.168.1.5:4747/video")
  - a video file path (loops when it ends, for demos)
  - "simulate": generated frames so the dashboard runs with no hardware

WINDOWS FIX: OpenCV defaults to the MSMF backend on Windows, which fails
to grab frames from virtual cameras (Iriun, OBS, DroidCam) — the symptom
is "CvCapture_MSMF::grabFrame ... can't grab frame" spam. We therefore
try DirectShow (CAP_DSHOW) first for integer indices, then fall back to
MSMF and finally CAP_ANY. Stream URLs try FFMPEG first, then CAP_ANY.

URL streams are read by a background thread that keeps ONLY the newest
frame, so the dashboard shows live video with no growing delay even
though the AI loop runs slower than the stream. If a stream or device
drops it is reconnected forever with backoff (CAMERA_RECONNECT_DELAY,
doubling up to MAX_RECONNECT_DELAY) instead of giving up.
"""

import platform
import re
import threading
import time
from collections import deque

import cv2
import numpy as np

from config import settings

IS_WINDOWS = platform.system() == "Windows"
MAX_RECONNECT_DELAY = 10.0
DROIDCAM_PORT = 4747
URL_FAILED_READS_LIMIT = 3     # consecutive failed reads before a URL counts as dropped
DEVICE_FAILED_READS_LIMIT = 10  # tolerate occasional dropped frames on devices
NEW_FRAME_WAIT_S = 0.05        # read() waits this long for a new URL frame


def normalise_source(src):
    """Turn a CAMERA_SOURCE value into something cv2.VideoCapture accepts.

    "droidcam:172.28.11.95[:4747]" -> "http://172.28.11.95:4747/video";
    "simulate", digit indices, full URLs and file paths are returned as-is.
    """
    src = str(src).strip()
    match = re.fullmatch(r"droidcam:([^:/\s]+)(?::(\d+))?", src, re.IGNORECASE)
    if match:
        host, port = match.group(1), match.group(2) or DROIDCAM_PORT
        return f"http://{host}:{port}/video"
    return src


def source_type(src):
    if src.lower() == "simulate":
        return "simulate"
    if src.isdigit():
        return "device"
    if re.match(r"(https?|rtsp)://", src, re.IGNORECASE):
        return "url"
    return "file"


class CameraManager:
    def __init__(self, source=None):
        self.source = normalise_source(source if source is not None else settings.CAMERA_SOURCE)
        self.type = source_type(self.source)
        self.simulated = self.type == "simulate"
        self.cap = None
        self.connected = False
        self.backend_used = None
        self.native_size = None  # (w, h) of frames before CAMERA_RESIZE
        self._sim_tick = 0
        self._failed_reads = 0
        self._reconnect_delay = settings.CAMERA_RECONNECT_DELAY
        self._frame_times = deque(maxlen=30)

        # URL reader thread state
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._latest_seq = 0
        self._read_seq = 0

    # ── Connection ───────────────────────────────────────────
    def _candidate_backends(self):
        """Backends to try, in order, for the current source type."""
        if self.type == "device" and IS_WINDOWS:
            # DirectShow first: required for Iriun / OBS / DroidCam Client
            return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
        if self.type == "url":
            return [("FFMPEG", cv2.CAP_FFMPEG), ("ANY", cv2.CAP_ANY)]
        return [("ANY", cv2.CAP_ANY)]

    def _new_capture(self, cam, backend):
        if self.type != "url":
            return cv2.VideoCapture(cam, backend)
        timeout = settings.CAMERA_OPEN_TIMEOUT_MS
        params = []
        for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
            if hasattr(cv2, prop):
                params += [getattr(cv2, prop), timeout]
        try:
            return cv2.VideoCapture(cam, backend, params)
        except (TypeError, cv2.error):
            # Older OpenCV builds without the params overload
            return cv2.VideoCapture(cam, backend)

    def _connect(self, quiet=False):
        """Try each backend until one opens AND delivers a real frame.
        Returns (cap, backend_name, first_frame) or (None, None, None)."""
        cam = int(self.source) if self.type == "device" else self.source

        for name, backend in self._candidate_backends():
            cap = self._new_capture(cam, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.type == "device":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

            # Opening is not proof of life: virtual cameras can open and
            # then deliver nothing. Require one real frame before accepting.
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, name, frame

            cap.release()
            if not quiet:
                print(f"  Camera source={self.source} opened on {name} but gave no frames — trying next backend")
        return None, None, None

    def open(self):
        self._stop.clear()  # release() sets it; the app reopens on every Start
        if self.simulated:
            self.connected = True
            self.backend_used = "simulate"
            return True

        cap, name, frame = self._connect()
        if cap is None:
            self.connected = False
            print(
                f"  ERROR: could not get frames from camera source '{self.source}'.\n"
                f"         Run 'python camera_test.py' to find a working index."
            )
            return False

        self.backend_used = name
        self.connected = True
        self._failed_reads = 0
        self._reconnect_delay = settings.CAMERA_RECONNECT_DELAY
        print(f"  Camera opened: source={self.source} type={self.type} backend={name}")

        if self.type == "url":
            self._store_latest(frame)
            self._thread = threading.Thread(
                target=self._url_reader, args=(cap,), name="camera-reader", daemon=True
            )
            self._thread.start()
        else:
            self.cap = cap
        return True

    def reconnect(self):
        """One reconnect attempt with backoff. The caller keeps calling it
        while the camera is down, so reconnection never gives up."""
        if self.simulated:
            return True
        if self.type == "url":
            # The reader thread reconnects on its own; just wait a little.
            self._stop.wait(min(self._reconnect_delay, 1.0))
            return self.connected

        if self.cap:
            self.cap.release()
            self.cap = None
        delay = self._reconnect_delay
        print(f"  Camera '{self.source}' down — reconnecting in {delay:g}s")
        if self._stop.wait(delay):
            return False
        cap, name, _ = self._connect(quiet=True)
        if cap is None:
            self._reconnect_delay = min(delay * 2, MAX_RECONNECT_DELAY)
            return False
        self.cap, self.backend_used = cap, name
        self.connected = True
        self._failed_reads = 0
        self._reconnect_delay = settings.CAMERA_RECONNECT_DELAY
        print(f"  Camera reconnected: source={self.source} backend={name}")
        return True

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=settings.CAMERA_OPEN_TIMEOUT_MS / 1000 + 2)
            self._thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False

    # ── URL reader thread ────────────────────────────────────
    def _store_latest(self, frame):
        with self._lock:
            self._latest = frame
            self._latest_seq += 1
        self._tick(frame)
        self._new_frame.set()

    def _url_reader(self, cap):
        """Continuously grab frames, keep only the newest, reconnect forever."""
        failed = 0
        delay = settings.CAMERA_RECONNECT_DELAY
        try:
            while not self._stop.is_set():
                if cap is None:
                    print(f"  Camera stream '{self.source}' down — reconnecting in {delay:g}s")
                    if self._stop.wait(delay):
                        break
                    cap, name, frame = self._connect(quiet=True)
                    if cap is None:
                        delay = min(delay * 2, MAX_RECONNECT_DELAY)
                        continue
                    print(f"  Camera stream reconnected: backend={name}")
                    self.backend_used = name
                    self.connected = True
                    delay = settings.CAMERA_RECONNECT_DELAY
                    failed = 0
                    self._store_latest(frame)
                    continue

                ok, frame = cap.read()
                if ok and frame is not None:
                    failed = 0
                    self.connected = True
                    self._store_latest(frame)
                    continue

                failed += 1
                if failed >= URL_FAILED_READS_LIMIT:
                    self.connected = False
                    cap.release()
                    cap = None
        except Exception as exc:  # noqa: BLE001 — never let the reader kill the app
            print(f"  Camera reader error: {exc}")
            self.connected = False
        finally:
            if cap is not None:
                cap.release()

    # ── Frames ───────────────────────────────────────────────
    def read(self):
        """Return (ok, frame). Never raises; reports failure via ok.
        (True, None) means "no new frame yet" — the caller just skips."""
        if self.simulated:
            frame = self._simulated_frame()
            self._tick(frame)
            return True, frame
        if self.type == "url":
            return self._read_url()
        if not self.cap:
            return False, None

        ok, frame = self.cap.read()
        if (not ok or frame is None) and self.type == "file":
            # Loop the demo video when it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok and frame is not None:
            self._failed_reads = 0
            self.connected = True
            self._tick(frame)
            return True, self._resize(frame)

        # Tolerate occasional dropped frames; only report a real outage.
        self._failed_reads += 1
        if self._failed_reads < DEVICE_FAILED_READS_LIMIT:
            return True, None  # transient: caller skips this frame
        self.connected = False
        return False, None

    def _read_url(self):
        if not self.connected:
            return False, None
        # Wait briefly for a fresh frame so the caller doesn't busy-spin
        if self._read_seq == self._latest_seq:
            self._new_frame.wait(NEW_FRAME_WAIT_S)
        with self._lock:
            self._new_frame.clear()
            if self._latest is None or self._read_seq == self._latest_seq:
                return True, None
            self._read_seq = self._latest_seq
            frame = self._latest
        return True, self._resize(frame)

    def _resize(self, frame):
        if not settings.CAMERA_RESIZE:
            return frame
        size = (settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
        if (frame.shape[1], frame.shape[0]) == size:
            return frame
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def _tick(self, frame):
        self.native_size = (frame.shape[1], frame.shape[0])
        self._frame_times.append(time.time())

    def status(self):
        times = list(self._frame_times)
        fps = 0.0
        if len(times) >= 2 and times[-1] > times[0]:
            fps = (len(times) - 1) / (times[-1] - times[0])
        return {
            "source": self.source,
            "type": self.type,
            "backend": self.backend_used,
            "connected": self.connected,
            "fps": round(fps, 1),
        }

    def _simulated_frame(self):
        """Generate a synthetic pool-like frame for hardware-free demos."""
        self._sim_tick += 1
        w, h = settings.FRAME_WIDTH, settings.FRAME_HEIGHT
        frame = np.full((h, w, 3), (150, 105, 20), dtype=np.uint8)  # pool blue (BGR)
        for y in range(0, h, 24):
            offset = int(10 * np.sin(self._sim_tick / 10 + y / 30))
            cv2.line(frame, (0, y + offset), (w, y - offset), (170, 125, 40), 2)
        cv2.putText(
            frame, "SIMULATED CAMERA FEED", (w // 2 - 160, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, "Set CAMERA_SOURCE env var to use a real camera",
            (w // 2 - 220, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
        )
        return frame
