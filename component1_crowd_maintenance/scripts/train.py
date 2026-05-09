"""
YOLOv8 Training Script
SLIIT Final Year Project - Component 1
Crowd-Aware Maintenance Scheduling Module
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# ================================================
# CONFIGURATION
# ================================================
BASE_DIR = Path(r"C:\Users\muham\Desktop\Final_Year_Project\component1_crowd_maintenance")
DATA_YAML = BASE_DIR / "datasets" / "combined_dataset" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Check GPU availability
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"\n{'='*50}")
print(f"Device: {'GPU ✅' if device == '0' else 'CPU ⚠️ (will be slow)'}")
print(f"{'='*50}\n")

# ================================================
# LOAD MODEL
# ================================================
print("Loading YOLOv8 nano model...")
print("(Downloading pre-trained weights if first time...)\n")

model = YOLO('yolov8n.pt')

# ================================================
# TRAIN
# ================================================
print("="*50)
print("STARTING TRAINING")
print("="*50)
print("Epochs: 50")
print("Image size: 640x640")
print("This will take 1-8 hours on CPU")
print("DO NOT close VS Code during training!")
print("="*50 + "\n")

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=8,
    name='swimmer_detector_v1',
    project=str(RESULTS_DIR),
    patience=15,
    save=True,
    plots=True,
    device=device,
    workers=2,
    verbose=True,
    exist_ok=True
)

# ================================================
# SAVE BEST MODEL
# ================================================
import shutil

best_model = RESULTS_DIR / 'swimmer_detector_v1' / 'weights' / 'best.pt'
if best_model.exists():
    shutil.copy2(best_model, MODELS_DIR / 'best_swimmer_model.pt')
    print(f"\n✅ Best model saved to: {MODELS_DIR / 'best_swimmer_model.pt'}")

print("\n" + "="*50)
print("✅ TRAINING COMPLETE!")
print("="*50)
print("Next step: Run evaluate.py to see accuracy")