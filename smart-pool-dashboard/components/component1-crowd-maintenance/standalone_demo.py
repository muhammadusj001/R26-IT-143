"""Standalone viva demo for Component 1 (no dashboard needed).
Webcam -> swimmer detection -> bather load -> maintenance report."""
from pathlib import Path
import cv2
from component1_crowd.detector import CrowdDetector

MODEL = Path(__file__).parent / "models" / "best_swimmer_model.pt"

det = CrowdDetector(model_path=MODEL)
det.load()
print(f"Model: {det.model_status}")
cap = cv2.VideoCapture(0)
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, result = det.process(frame)
        cv2.putText(frame, f"Swimmers: {result['swimmer_count']}  Density: {result['density_level']}  Load: {result['bather_load']} ph",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, result["maintenance_recommendation"][:70],
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("Component 1 - Crowd-Aware Maintenance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    det.scheduler.print_report()
