"""Standalone viva demo for Component 3 (no dashboard needed).
Place trained weights at models/best.pt first."""
from pathlib import Path
import cv2
from component3_drowning.detector import DrowningDetector

MODEL = Path(__file__).parent / "models" / "best.pt"

det = DrowningDetector(model_path=MODEL)
det.load()
print(f"Model: {det.model_status}")
cap = cv2.VideoCapture(0)  # or IP stream URL
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, result, alert = det.process(frame)
        if alert:
            print(f"ALERT {alert['time']}: {alert['count']} drowning")
        cv2.imshow("Component 3 - Drowning Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
