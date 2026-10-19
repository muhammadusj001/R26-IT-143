"""Click the corners of the WATER area once, so the system knows where the pool is.

    python tools/draw_pool_roi.py --source 1                          (standalone, 1280x720)
    python tools/draw_pool_roi.py --source 1 --width 640 --height 480 (dashboard size)

Left-click = add corner   u = undo   c = clear   s = save   q = quit
Saved to config/pool_roi.json (as 0-1 fractions, so any resolution works).
Re-run it whenever you move the camera.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from component4_garbage import config  # noqa: E402
from component4_garbage.camera import LiveCamera  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")
ap.add_argument("--width", type=int, default=1280,
                help="use the SAME size the detector runs at (dashboard: 640)")
ap.add_argument("--height", type=int, default=720, help="dashboard: 480")
args = ap.parse_args()

cam = LiveCamera(args.source, args.width, args.height)
ok, frame, _ = cam.read()
cam.release()
if not ok:
    sys.exit("Could not read a frame from the camera.")
h, w = frame.shape[:2]
points = []


def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


cv2.namedWindow("pool area")
cv2.setMouseCallback("pool area", on_click)
while True:
    view = frame.copy()
    if points:
        cv2.polylines(view, [np.array(points, np.int32)], len(points) > 2, (255, 200, 0), 2)
        for p in points:
            cv2.circle(view, p, 5, (0, 0, 255), -1)
    cv2.putText(view, "Click water corners | s=save u=undo c=clear q=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("pool area", view)
    k = cv2.waitKey(30) & 0xFF
    if k == ord("u") and points:
        points.pop()
    elif k == ord("c"):
        points.clear()
    elif k == ord("s") and len(points) >= 3:
        config.CONFIG_DIR.mkdir(exist_ok=True)
        data = {"polygon": [[round(x / w, 4), round(y / h, 4)] for x, y in points],
                "size": [w, h]}
        config.POOL_ROI_FILE.write_text(json.dumps(data, indent=2))
        print("saved", config.POOL_ROI_FILE)
        break
    elif k == ord("q"):
        break
cv2.destroyAllWindows()
