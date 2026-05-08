"""Reusable webcam capture service for MJPEG streaming and future YOLO integration."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable

import cv2


logger = logging.getLogger(__name__)


class WebcamServiceError(RuntimeError):
    """Raised when the webcam service cannot start or stream frames."""


class WebcamService:
    def __init__(self, camera_index: int = 0, target_fps: int = 30) -> None:
        self.camera_index = camera_index
        self.target_fps = max(1, target_fps)
        self._capture: cv2.VideoCapture | None = None
        self._backend_name: str | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_frame: object | None = None
        self._latest_jpeg: bytes | None = None
        self._last_capture_time = 0.0
        self._frame_count = 0
        self._started_at = 0.0
        self._last_error: str | None = None

    def probe_frame(self, timeout_seconds: float = 2.0) -> dict[str, object]:
        if not self.is_running():
            raise WebcamServiceError("Webcam service is not running")

        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            with self._lock:
                frame = self._latest_frame

            if frame is not None:
                height, width = frame.shape[:2]
                return {
                    "status": "ok",
                    "running": True,
                    "camera_index": self.camera_index,
                    "frame_width": int(width),
                    "frame_height": int(height),
                }

            time.sleep(0.05)

        raise WebcamServiceError("No webcam frame available yet")

    def start(self) -> None:
        if self._running:
            return

        self._last_error = None

        capture = self._open_first_available_capture()
        if capture is None:
            message = "No available webcam found on indexes 0, 1, or 2"
            self._last_error = message
            logger.warning(message)
            raise WebcamServiceError(message)

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._capture = capture
        self._started_at = time.monotonic()
        self._frame_count = 0
        self._running = True
        logger.info(
            "Webcam started successfully",
            extra={"camera_index": self.camera_index, "backend": self._backend_name, "target_fps": self.target_fps},
        )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                logger.info(
                    "Webcam released",
                    extra={"camera_index": self.camera_index, "backend": self._backend_name},
                )

        self._capture = None
        self._backend_name = None
        self._thread = None
        self._latest_frame = None
        self._latest_jpeg = None

    def is_running(self) -> bool:
        return self._running and self._capture is not None and self._capture.isOpened()

    def get_status(self) -> dict[str, object]:
        actual_fps = 0.0
        if self._started_at > 0:
            elapsed = max(time.monotonic() - self._started_at, 0.001)
            actual_fps = round(self._frame_count / elapsed, 2)

        return {
            "running": self.is_running(),
            "camera_index": self.camera_index,
            "backend": self._backend_name,
            "target_fps": self.target_fps,
            "actual_fps": actual_fps,
            "last_error": self._last_error,
        }

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def mjpeg_frame_generator(
        self,
        overlay_fn: Callable[[object], object] | None = None,
    ):
        if not self.is_running():
            raise WebcamServiceError("Webcam stream is not running")

        frame_interval = 1.0 / self.target_fps

        while self.is_running():
            with self._lock:
                jpeg = self._latest_jpeg
                frame = self._latest_frame

            if frame is not None and overlay_fn is not None:
                try:
                    frame = overlay_fn(frame)
                    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        jpeg = encoded.tobytes()
                except Exception:
                    pass

            if jpeg is None:
                time.sleep(frame_interval)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(frame_interval)

    def _capture_loop(self) -> None:
        assert self._capture is not None

        min_interval = 1.0 / self.target_fps

        while self._running and self._capture.isOpened():
            now = time.monotonic()
            if now - self._last_capture_time < min_interval:
                time.sleep(min_interval / 4)
                continue

            ok, frame = self._capture.read()
            self._last_capture_time = now

            if not ok or frame is None:
                time.sleep(min_interval / 2)
                continue

            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_jpeg = encoded.tobytes()
                self._frame_count += 1

    def _open_first_available_capture(self) -> cv2.VideoCapture | None:
        candidate_indexes = [0, 1, 2]
        if self.camera_index not in candidate_indexes:
            candidate_indexes.insert(0, self.camera_index)

        backends = self._get_candidate_backends()

        for index in candidate_indexes:
            for backend_name, backend_id in backends:
                logger.info("Trying webcam index %s with backend %s", index, backend_name)
                capture = cv2.VideoCapture(index, backend_id)
                if not capture.isOpened():
                    capture.release()
                    continue

                if not self._warm_up_capture(capture):
                    logger.debug("Webcam index %s opened but did not return frames", index)
                    capture.release()
                    continue

                self.camera_index = index
                self._backend_name = backend_name
                return capture

        return None

    def _get_candidate_backends(self) -> list[tuple[str, int]]:
        if sys.platform.startswith("win"):
            return [
                ("CAP_DSHOW", cv2.CAP_DSHOW),
                ("CAP_MSMF", cv2.CAP_MSMF),
                ("CAP_ANY", cv2.CAP_ANY),
            ]

        return [("CAP_ANY", cv2.CAP_ANY)]

    def _warm_up_capture(self, capture: cv2.VideoCapture, timeout_seconds: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None:
                return True
            time.sleep(0.05)

        return False
