"""Standalone viva demo for Component 4 (no dashboard needed).
Webcam -> garbage detection with class disambiguation (ball/leaf != garbage)."""
from pathlib import Path
import cv2
from component4_garbage.detector import GarbageDetector

MODEL = Path(__file__).parent / "models" / "swimming_pool_garbage_yolo.pt"

det = GarbageDetector(model_path=MODEL)
det.load()
print(f"Model: {det.model_status}")
cap = cv2.VideoCapture(0)
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, result, alert = det.process(frame)
        if alert:
            cv2.putText(frame, "ALERT: Garbage Detected!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Component 4 - Garbage Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
