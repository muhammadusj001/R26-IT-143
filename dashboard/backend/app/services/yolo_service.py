"""YOLO inference service for real-time swimmer detection."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import torch
from ultralytics import YOLO


class YOLOServiceError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or used."""


class YOLOService:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        max_detections: int = 50,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = bool(torch.cuda.is_available())
        self._model: YOLO | None = None
        self._lock = Lock()

    def _resolve_model_path(self, model_path: str) -> Path:
        candidate = Path(model_path)
        if candidate.is_absolute():
            return candidate

        project_root = Path(__file__).resolve().parents[4]
        return project_root / candidate

    def load_model(self) -> None:
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise YOLOServiceError(f"Model file not found: {self.model_path}")

        try:
            self._model = YOLO(str(self.model_path))
            self._model.overrides["verbose"] = False
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            raise YOLOServiceError(f"Unable to load YOLO model: {exc}") from exc

    def is_ready(self) -> bool:
        return self._model is not None

    def get_status(self) -> dict[str, object]:
        return {
            "ready": self.is_ready(),
            "model_path": str(self.model_path),
            "device": "cuda" if self.device == 0 else "cpu",
            "conf_threshold": self.conf_threshold,
            "imgsz": self.imgsz,
            "max_detections": self.max_detections,
        }

    def predict_frame(self, frame) -> dict[str, object]:
        model = self._ensure_model()

        with self._lock:
            results = model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                half=self.use_half,
                max_det=self.max_detections,
                verbose=False,
            )

        result = results[0]
        boxes = result.boxes
        swimmer_count = 0 if boxes is None else len(boxes)
        annotated_frame = result.plot()

        cv2.putText(
            annotated_frame,
            f"Swimmers: {swimmer_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return {
            "count": swimmer_count,
            "annotated_frame": annotated_frame,
        }

    def annotate_frame(self, frame):
        return self.predict_frame(frame)["annotated_frame"]

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self.load_model()

        if self._model is None:
            raise YOLOServiceError("YOLO model is not ready")

        return self._model
