"""Train the swimmer detection model with a clean YOLOv8 workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = BASE_DIR / "datasets" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
RUN_NAME = "train_custom_model"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {'GPU' if device == '0' else 'CPU'}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Model output directory: {MODELS_DIR}")

    model = YOLO("yolov8m.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        project=str(MODELS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        save=True,
        plots=True,
        patience=15,
        workers=2,
        verbose=True,
    )

    run_dir = MODELS_DIR / RUN_NAME
    weights_dir = run_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    exported_best = MODELS_DIR / "best_custom_swimmer_model.pt"
    exported_last = MODELS_DIR / "last_custom_swimmer_model.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, exported_best)
        print(f"Best weights saved to: {exported_best}")

    if last_weights.exists():
        shutil.copy2(last_weights, exported_last)
        print(f"Last weights saved to: {exported_last}")

    print("Training complete.")
    print(f"Run artifacts stored in: {run_dir}")
    return results


if __name__ == "__main__":
    main()