"""Evaluate the trained swimmer detection model with YOLOv8 validation metrics."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = BASE_DIR / "datasets" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
MODEL_CANDIDATES = [
    MODELS_DIR / "best_custom_swimmer_model.pt",
    MODELS_DIR / "best_swimmer_model.pt",
    MODELS_DIR / "best_swimmer_model_v2_combined.pt",
]


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No trained swimmer model found. Expected one of: "
        + ", ".join(str(path) for path in MODEL_CANDIDATES)
    )


def print_metrics(model_path: Path, metrics) -> None:
    precision = float(getattr(metrics.box, "mp", 0.0))
    recall = float(getattr(metrics.box, "mr", 0.0))
    map50 = float(getattr(metrics.box, "map50", 0.0))
    map5095 = float(getattr(metrics.box, "map", 0.0))

    print("=" * 60)
    print("SWIMMER MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 60)
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"mAP50       : {map50:.4f}")
    print(f"mAP50-95    : {map5095:.4f}")
    print("=" * 60)


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    model_path = resolve_model_path()
    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    print_metrics(model_path, metrics)


if __name__ == "__main__":
    main()