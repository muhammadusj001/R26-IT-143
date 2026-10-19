"""
Component 4 — run ALONE with a live camera (no dashboard needed).

Examples (run from the component4-garbage-detection folder):

    python standalone_demo.py                       # webcam 0 (laptop camera)
    python standalone_demo.py --source 1            # Iriun is usually index 1
    python standalone_demo.py --source iriun        # auto-pick Iriun
    python standalone_demo.py --source http://192.168.1.5:8080/video   # IP Webcam app
    python standalone_demo.py --source test_clip.mp4 --record out.mp4   # repeatable test
    python standalone_demo.py --source 1 --record pool_videos/clip1.mp4 --raw  # clean video for auto_label

Keys in the window:  q = quit   s = save screenshot   r = reload pool area
"""

import argparse
import os
import time
from pathlib import Path

import cv2

from component4_garbage import config
from component4_garbage.camera import LiveCamera
from component4_garbage.detector import GarbageDetector


def main():
    ap = argparse.ArgumentParser(description="Component 4 live garbage + intent detection")
    ap.add_argument("--source", default=os.environ.get("CAMERA_SOURCE", "0"),
                    help="0/1/2 webcam index, 'iriun', http(s)/rtsp URL, or video file")
    ap.add_argument("--model", default=str(config.GARBAGE_MODEL_PATH))
    ap.add_argument("--conf", type=float, default=config.CONF)
    ap.add_argument("--imgsz", type=int, default=config.IMGSZ,
                    help="960 = better for small/far garbage; 640 = faster on CPU")
    ap.add_argument("--record", default="", help="save the video to this .mp4")
    ap.add_argument("--raw", action="store_true",
                    help="record the camera image WITHOUT boxes (use this for training / auto_label)")
    ap.add_argument("--garbage-every", type=int,
                    default=int(os.environ.get("C4_GARBAGE_EVERY", "2")),
                    help="run the garbage model every Nth frame (2 = CPU laptop, 1 = GPU)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    config.GARBAGE_EVERY_N = max(1, args.garbage_every)
    det = GarbageDetector(model_path=Path(args.model), conf_threshold=args.conf,
                          imgsz=args.imgsz, show_hud=True)
    det.load()
    print(f"Garbage model : {det.model_status}")
    print(f"Open-vocab    : {det.ov_status}")
    print(f"People model  : {det.source_model_status}")
    print(f"Pose          : {det.pose.status}")
    print(f"Intent mode   : {det.intent_analyzer.mode}")
    print(f"Pool area     : {'loaded' if det.roi.defined else 'NOT SET (run tools/draw_pool_roi.py)'}")
    print(f"Tiny garbage  : {'tiles %dx%d' % (config.TILES, config.TILES) if config.TILES > 1 else 'off'}"
          f" | Underwater: {'on' if config.UNDERWATER else 'off'}")

    cam = LiveCamera(args.source, args.width, args.height)
    print(f"Camera        : {cam.source}  (press q to quit)")
    writer = None
    try:
        while True:
            ok, frame, ts = cam.read()
            if not ok or frame is None:
                print("Camera stopped / video finished.")
                break
            raw = frame.copy() if args.raw else None
            frame, result, alert = det.process(frame, ts if cam.is_file else None)
            if alert:
                print(time.strftime("%H:%M:%S"), alert)
            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             min(cam.fps, 30), (w, h))
                writer.write(raw if raw is not None else frame)
            cv2.imshow("Component 4 - Garbage & Intent (live)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                name = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(name, frame)
                print("saved", name)
            if key == ord("r"):
                det.roi.__init__()
                det.roi._cache.clear()
                print("pool area reloaded:", det.roi.defined)
    finally:
        cam.release()
        if writer is not None:
            writer.release()
        det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
