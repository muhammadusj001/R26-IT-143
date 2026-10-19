"""Collect training images FROM YOUR OWN LIVE CAMERA (the #1 accuracy fix).

The Roboflow images were taken by someone else, at another pool, with
another phone and angle. Your live Iriun view looks different, so the
model is unsure. Adding a few hundred frames from YOUR camera, YOUR pool
angle, YOUR lighting fixes most missed / wrong detections.

    python tools/capture_dataset.py --source 1 --every 1.0

Keys:  SPACE = save this frame now   a = auto-save on/off (every N s)   q = quit
Images go to dataset_capture/<date>/ . Upload that folder to Roboflow,
draw boxes, and add EMPTY-POOL frames too (no boxes = "background"),
which teaches the model what is NOT garbage (ripples, glare, tiles, swimmers).
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from component4_garbage.camera import LiveCamera  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")
ap.add_argument("--every", type=float, default=1.0, help="seconds between auto-saves")
ap.add_argument("--out", default="dataset_capture")
args = ap.parse_args()

out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M")
out.mkdir(parents=True, exist_ok=True)
cam = LiveCamera(args.source)
auto, last, n = False, 0.0, 0
try:
    while True:
        ok, frame, _ = cam.read()
        if not ok:
            break
        now = time.time()
        save = False
        if auto and now - last >= args.every:
            save = True
        view = frame.copy()
        cv2.putText(view, f"saved {n} | auto {'ON' if auto else 'off'} | SPACE save, a auto, q quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("capture", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            save = True
        elif k == ord("a"):
            auto = not auto
        elif k == ord("q"):
            break
        if save:
            cv2.imwrite(str(out / f"pool_{int(now * 1000)}.jpg"), frame)
            n += 1
            last = now
finally:
    cam.release()
    cv2.destroyAllWindows()
print(f"{n} images saved in {out}")
