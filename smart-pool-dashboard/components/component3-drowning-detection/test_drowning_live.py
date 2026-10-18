"""
Standalone live tester for the Component 3 drowning model (no dashboard).

Usage:
    python test_drowning_live.py                      # webcam 0
    python test_drowning_live.py --source 1           # webcam 1 (Iriun)
    python test_drowning_live.py --source http://IP:8080/video
    python test_drowning_live.py --source clip.mp4 --conf 0.6 --window 20

Keys: q = quit, + / - = raise / lower confidence by 0.05
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASSES = ("Drowning", "Person out of water", "Swimming")
COLORS = {
    "Drowning": (0, 0, 255),
    "Person out of water": (0, 165, 255),
    "Swimming": (0, 255, 0),
}
DANGER_RATIO = 0.6


def parse_args():
    default_model = Path(__file__).resolve().parent / "models" / "best.pt"
    p = argparse.ArgumentParser(description="Live drowning model tester")
    p.add_argument("--source", default="0",
                   help="camera index (digits) or stream URL / video file")
    p.add_argument("--model", default=str(default_model))
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--window", type=int, default=15,
                   help="frames used for the smoothed DROWNING/SAFE label")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    conf = args.conf
    history = deque(maxlen=args.window)
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        counts = dict.fromkeys(CLASSES, 0)
        results = model(frame, imgsz=args.imgsz, conf=conf, verbose=False)
        for result in results:
            for box in result.boxes:
                name = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = COLORS.get(name, (255, 255, 255))
                label = f"{name} {score:.0%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                              4 if name == "Drowning" else 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if name in counts:
                    counts[name] += 1

        history.append(1 if counts["Drowning"] else 0)
        ratio = sum(history) / len(history)
        danger = len(history) == history.maxlen and ratio >= DANGER_RATIO
        verdict = "DROWNING" if danger else "SAFE"

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        hud = (f"conf {conf:.2f} | imgsz {args.imgsz} | FPS {fps:4.1f} | "
               f"Drown {counts['Drowning']} Out {counts['Person out of water']} "
               f"Swim {counts['Swimming']} | ratio {ratio:.2f}/{args.window} | {verdict}")
        print("\r" + hud + "   ", end="", flush=True)

        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, hud.rsplit(" | ", 1)[0], (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, verdict, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if danger else (0, 200, 0), 2)

        cv2.imshow("Drowning live test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("+"), ord("=")):
            conf = min(0.95, round(conf + 0.05, 2))
        elif key in (ord("-"), ord("_")):
            conf = max(0.05, round(conf - 0.05, 2))

    print()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
