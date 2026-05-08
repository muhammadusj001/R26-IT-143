"""
Real-Time Swimmer Detection
SLIIT FYP - Component 1
"""
from ultralytics import YOLO
from pathlib import Path
import cv2

BASE_DIR = Path(r"C:\Users\muham\Desktop\Final_Year_Project\component1_crowd_maintenance")
MODEL_PATH = BASE_DIR / "models" / "best_swimmer_model.pt"

# Load best model
model = YOLO(str(MODEL_PATH))
print("✅ Model loaded!")

# Test on sample image
test_dir = BASE_DIR / "datasets" / "combined_dataset" / "test" / "images"
test_images = list(test_dir.glob("*.jpg"))[:3]

for img_path in test_images:
    results = model(str(img_path), conf=0.25)
    count = len(results[0].boxes)
    print(f"Detected: {count} swimmers in {img_path.name}")
    
    annotated = results[0].plot()
    cv2.imshow(f"Detection - {img_path.name}", annotated)
    cv2.waitKey(2000)

cv2.destroyAllWindows()
print("✅ Detection test complete!")