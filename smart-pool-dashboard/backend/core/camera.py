"""
Shared camera manager.

ONE stream is captured here and the same frame is handed to the crowd,
drowning and garbage modules. This replaces the previous design where
each component opened its own camera.

Supports:
  - integer webcam index (e.g. "0", "1" for Iriun)
  - IP stream URL (e.g. "http://192.168.1.5:8080/video")
  - "simulate": generated frames so the dashboard runs with no hardware

WINDOWS FIX: OpenCV defaults to the MSMF backend on Windows, which fails
to grab frames from virtual cameras (Iriun, OBS, DroidCam) — the symptom
is "CvCapture_MSMF::grabFrame ... can't grab frame" spam. We therefore
try DirectShow (CAP_DSHOW) first for integer indices, then fall back to
MSMF and finally CAP_ANY. Stream URLs skip DSHOW (it does not apply).
Reconnection is bounded so a dead camera reports failure instead of
looping forever.
"""

import platform
import time

import cv2
import numpy as np

from config import settings

IS_WINDOWS = platform.system() == "Windows"
MAX_RECONNECT_ATTEMPTS = 3


class CameraManager:
    def __init__(self, source=None):
        self.source = str(source if source is not None else settings.CAMERA_SOURCE)
        self.simulated = self.source.lower() == "simulate"
        self.cap = None
        self.connected = False
        self.backend_used = None
        self._sim_tick = 0
        self._failed_reads = 0
        self._reconnect_attempts = 0

    # ── Connection ───────────────────────────────────────────
    def _candidate_backends(self):
        """Backends to try, in order, for the current source type."""
        if self.source.isdigit() and IS_WINDOWS:
            # DirectShow first: required for Iriun / OBS / DroidCam
            return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
        return [("ANY", cv2.CAP_ANY)]

    def open(self):
        if self.simulated:
            self.connected = True
            return True

        cam = int(self.source) if self.source.isdigit() else self.source

        for name, backend in self._candidate_backends():
            cap = cv2.VideoCapture(cam, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

            # Opening is not proof of life: virtual cameras can open and
            # then deliver nothing. Require one real frame before accepting.
            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap = cap
                self.backend_used = name
                self.connected = True
                self._failed_reads = 0
                self._reconnect_attempts = 0
                print(f"  Camera opened: source={self.source} backend={name}")
                return True

            cap.release()
            print(f"  Camera source={self.source} opened on {name} but gave no frames — trying next backend")

        self.connected = False
        print(
            f"  ERROR: could not get frames from camera source '{self.source}'.\n"
            f"         Run 'python camera_test.py' to find a working index."
        )
        return False

    def reconnect(self):
        """Bounded reconnection — gives up instead of looping forever."""
        if self.simulated:
            return True
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            return False
        self._reconnect_attempts += 1
        if self.cap:
            self.cap.release()
            self.cap = None
        time.sleep(2)
        return self.open()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False

    # ── Frames ───────────────────────────────────────────────
    def read(self):
        """Return (ok, frame). Never raises; reports failure via ok."""
        if self.simulated:
            return True, self._simulated_frame()
        if not self.cap:
            return False, None

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._failed_reads = 0
            self.connected = True
            return True, frame

        # Tolerate occasional dropped frames; only report a real outage.
        self._failed_reads += 1
        if self._failed_reads < 10:
            return True, None  # transient: caller skips this frame
        self.connected = False
        return False, None

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
