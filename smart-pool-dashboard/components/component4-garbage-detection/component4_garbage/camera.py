"""
Live camera input for Component 4 (Iriun Webcam, laptop webcam, CCTV/RTSP,
IP Webcam app, or a test video file).

How Iriun works: the Iriun app on your phone streams to the Iriun desktop
program over Wi-Fi/USB, and the desktop program installs a VIRTUAL WEBCAM.
OpenCV sees it like any normal webcam, just with an index number:
  index 0 -> usually the laptop's built-in camera
  index 1 -> usually Iriun (2 on some PCs)
Run  `python tools/list_cameras.py`  to see which index is Iriun.

`source` can be:
  0, 1, 2 ...                      webcam index (Iriun = usually 1)
  "iriun"                          auto-pick the first NON-built-in camera
  "http://192.168.1.5:8080/video"  IP Webcam (Android) / DroidCam URL
  "rtsp://user:pass@ip:554/..."    CCTV / NVR stream
  "test_clip.mp4"                  a recorded video (for repeatable tests)
"""

import platform
import threading
import time

import cv2


def _backend():
    # DirectShow opens USB/virtual cams (Iriun) much faster on Windows.
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def list_cameras(max_index=6):
    """Returns [(index, width, height)] for every camera that opens."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, _backend())
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append((idx, frame.shape[1], frame.shape[0]))
        cap.release()
    return found


def parse_source(source):
    """'1' -> 1, 'iriun' -> best guess index, URLs/paths unchanged."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    if s.lower() == "iriun":
        cams = list_cameras()
        # Prefer index >= 1 (index 0 is normally the built-in webcam).
        external = [c for c in cams if c[0] >= 1]
        if external:
            return external[0][0]
        if cams:
            return cams[0][0]
        raise RuntimeError("No camera found. Is the Iriun desktop app running?")
    return s


class LiveCamera:
    """Background-thread reader that always keeps only the NEWEST frame.

    Without this, a slow model makes OpenCV queue old frames and the
    "live" view drifts seconds behind reality, which breaks throw timing.
    For video files we read in order instead (no frame dropping)."""

    def __init__(self, source=0, width=1280, height=720):
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and not self.source.startswith(
            ("http://", "https://", "rtsp://", "rtmp://"))
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, _backend())
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            self.cap = cv2.VideoCapture(self.source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source {self.source!r}. "
                "Check the index with tools/list_cameras.py, or that the "
                "phone and PC are on the same Wi-Fi (for URL sources).")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame = None
        self._ts = 0.0
        self._ok = True
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None
        if not self.is_file:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self):
        failures = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    self._ok = False
                    break
                time.sleep(0.02)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self._ts = time.time()

    def read(self):
        """Returns (ok, frame, timestamp_seconds)."""
        if self.is_file:
            ok, frame = self.cap.read()
            pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            return ok, frame, pos_ms / 1000.0
        deadline = time.time() + 5
        while self._frame is None and self._ok and time.time() < deadline:
            time.sleep(0.01)
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return self._ok, self._frame.copy(), self._ts

    def release(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.cap.release()
