"""Shows every camera OpenCV can open, so you know which index is Iriun.

    python tools/list_cameras.py          # list + preview each camera for 2 s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from component4_garbage.camera import _backend, list_cameras  # noqa: E402

cams = list_cameras()
if not cams:
    print("No cameras found. Start the Iriun desktop app and the Iriun phone app first.")
for idx, w, h in cams:
    print(f"index {idx}: {w}x{h}")
    cap = cv2.VideoCapture(idx, _backend())
    for _ in range(60):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"camera index {idx} - is this Iriun?", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("camera preview", frame)
        cv2.waitKey(30)
    cap.release()
cv2.destroyAllWindows()
print("Use the index that showed your phone camera:  python standalone_demo.py --source <index>")
