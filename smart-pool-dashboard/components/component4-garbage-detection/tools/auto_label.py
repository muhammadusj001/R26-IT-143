"""
Label your own pool frames AUTOMATICALLY (no drawing boxes by hand).

Uses YOLO-World, an "open-vocabulary" detector: you describe the objects
in words ("plastic bottle", "styrofoam piece") and it finds them without
being trained on your data. Your current garbage model can vote too.
You then only CHECK and fix the boxes in Roboflow (minutes, not hours).

    # from recorded Iriun videos (one frame every 1 s):
    python tools/auto_label.py --input pool_videos/ --every 1.0 --out auto_labeled
    # or from frames saved with tools/capture_dataset.py:
    python tools/auto_label.py --input dataset_capture/ --out auto_labeled

Output: auto_labeled/images, auto_labeled/labels (YOLO format, your 6 class
names), auto_labeled/preview (boxes drawn, for a quick look) and data.yaml.
Zip images+labels+data.yaml and drag the zip into a Roboflow project
("Upload with annotations"), fix wrong boxes, export YOLOv8 ->
my_pool_frames.zip for the Colab notebook.

Needs internet once (downloads yolov8s-worldv2.pt and the CLIP text model).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

try:                                   # inside the component folder
    from component4_garbage import config  # noqa: E402
    CLASSES = config.ALL_CLASSES
    DEFAULT_GARBAGE_MODEL = str(config.GARBAGE_MODEL_PATH)
except ImportError:                    # copied alone, e.g. to Colab
    CLASSES = ["aluminium foil", "ball", "bottle", "juice", "leaf", "thermocol"]
    DEFAULT_GARBAGE_MODEL = "swimming_pool_garbage_yolo.pt"
# Words YOLO-World understands -> your class name
PROMPTS = {
    "plastic bottle": "bottle", "water bottle": "bottle",
    "aluminium foil": "aluminium foil", "crumpled foil wrapper": "aluminium foil",
    "juice box": "juice", "drink carton": "juice", "plastic cup": "juice",
    "styrofoam piece": "thermocol", "white foam board": "thermocol",
    "leaf": "leaf", "ball": "ball",
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv")


def iter_frames(inp, every):
    paths = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.is_file())
    for p in paths:
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            img = cv2.imread(str(p))
            if img is not None:
                yield p.stem, img
        elif ext in VID_EXT:
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            step, i = max(1, int(round(fps * every))), 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    yield f"{p.stem}_{i:06d}", frame
                i += 1
            cap.release()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of videos and/or images, or one file")
    ap.add_argument("--out", default="auto_labeled")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between video frames")
    ap.add_argument("--conf", type=float, default=0.15, help="YOLO-World confidence")
    ap.add_argument("--world", default="yolov8s-worldv2.pt")
    ap.add_argument("--garbage-model", default=DEFAULT_GARBAGE_MODEL,
                    help="your current model; it adds boxes it is sure about (conf 0.4)")
    ap.add_argument("--no-garbage-model", action="store_true",
                    help="do not let your current garbage model add boxes")
    a = ap.parse_args()

    from ultralytics import YOLO
    world = YOLO(a.world)
    prompt_words = list(PROMPTS)
    world.set_classes(prompt_words)
    own = None
    if not a.no_garbage_model and Path(a.garbage_model).exists():
        own = YOLO(a.garbage_model)

    out = Path(a.out)
    for d in ("images", "labels", "preview"):
        (out / d).mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_empty = 0
    for name, img in iter_frames(Path(a.input), a.every):
        h, w = img.shape[:2]
        cands = []                                      # (conf, cls_idx, box)
        for r in world.predict(img, conf=a.conf, imgsz=960, verbose=False):
            for b in r.boxes:
                word = prompt_words[int(b.cls[0])]
                cands.append((float(b.conf[0]), CLASSES.index(PROMPTS[word]),
                              tuple(map(int, b.xyxy[0].tolist()))))
        if own is not None:
            for r in own.predict(img, conf=0.4, imgsz=960, verbose=False):
                for b in r.boxes:
                    cname = own.names[int(b.cls[0])]
                    if cname in CLASSES:
                        cands.append((float(b.conf[0]), CLASSES.index(cname),
                                      tuple(map(int, b.xyxy[0].tolist()))))
        # merge duplicates (both models / two prompts on one object)
        cands.sort(reverse=True)
        keep = []
        for c in cands:
            if c[2][2] - c[2][0] > 0.6 * w or c[2][3] - c[2][1] > 0.6 * h:
                continue                                # whole-pool false box
            if all(iou(c[2], k[2]) < 0.5 for k in keep):
                keep.append(c)
        lines, prev = [], img.copy()
        for conf, ci, (x1, y1, x2, y2) in keep:
            lines.append(f"{ci} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(prev, f"{CLASSES[ci]} {conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img)
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
        cv2.imwrite(str(out / "preview" / f"{name}.jpg"), prev)
        n_img += 1
        n_box += len(lines)
        n_empty += not lines
    (out / "data.yaml").write_text(
        "train: images\nval: images\nnc: %d\nnames: %s\n" % (len(CLASSES), CLASSES))
    print(f"{n_img} frames, {n_box} boxes, {n_empty} frames with no object (kept as background)")
    print(f"Check {out/'preview'} then upload images+labels+data.yaml to Roboflow to fix boxes.")


if __name__ == "__main__":
    main()
