"""
Make MORE training images from the dataset you already have - no new
photos, no manual labelling (synthetic "copy-paste" augmentation).

What it does
  1. Cuts every labelled object (bottle, foil, juice, thermocol, ball, leaf)
     out of your training images -> an "object bank".
  2. Pastes 1-3 random objects onto other pool images at NEW places, sizes
     (incl. small/far), angles and brightness, with soft edges, partly
     "under water" (faded + blue tint), and writes the YOLO labels itself.
     Rare classes (ball, thermocol) are picked more often to balance them.
  3. Optionally makes EMPTY-POOL images (--negatives): it paints the
     objects out (inpainting) so the model learns what plain water,
     ripples and glare look like - this cuts false boxes on water.

    python tools/synth_pool_data.py --data /content/merged --out /content/synth --n 1500 --negatives 300

The Colab notebook (ml/train_v2_colab.ipynb, cell 4b) runs this for you.
Only TRAIN images are used and produced; valid/test stay real, so your
scores stay honest. Dependencies: opencv-python, numpy, pyyaml (all
already installed with ultralytics).
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_labels(path, w, h):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            boxes.append((c, max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return boxes


def to_yolo(c, x1, y1, x2, y2, w, h):
    return f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def overlaps(box, others, thr=0.05):
    x1, y1, x2, y2 = box
    for o in others:
        ix = max(0, min(x2, o[3]) - max(x1, o[1]))
        iy = max(0, min(y2, o[4]) - max(y1, o[2]))
        if ix * iy > thr * (x2 - x1) * (y2 - y1):
            return True
    return False


def soft_mask(h, w):
    """Feathered ellipse: object edges fade into the water."""
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (max(1, int(w * 0.46)), max(1, int(h * 0.46))),
                0, 0, 360, 1.0, -1)
    k = max(3, (min(h, w) // 6) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def build_bank(train_dir, max_per_class=400):
    bank = {}
    imgs = [p for p in (train_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT]
    random.shuffle(imgs)
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for c, x1, y1, x2, y2 in read_labels(train_dir / "labels" / f"{img_path.stem}.txt", w, h):
            if x2 - x1 < 12 or y2 - y1 < 12 or len(bank.get(c, [])) >= max_per_class:
                continue
            px, py = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
            crop = img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)].copy()
            if crop.size:
                bank.setdefault(c, []).append(crop)
    return bank


def augment_crop(crop, scale):
    h, w = crop.shape[:2]
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if random.random() < 0.5:
        crop = cv2.flip(crop, 1)
    mask = soft_mask(nh, nw)
    angle = random.uniform(-25, 25)
    if abs(angle) > 3:
        diag = int(np.hypot(nh, nw)) + 2
        pad_crop = np.zeros((diag, diag, 3), np.uint8)
        pad_mask = np.zeros((diag, diag), np.float32)
        oy, ox = (diag - nh) // 2, (diag - nw) // 2
        pad_crop[oy:oy + nh, ox:ox + nw] = crop
        pad_mask[oy:oy + nh, ox:ox + nw] = mask
        M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
        crop = cv2.warpAffine(pad_crop, M, (diag, diag))
        mask = cv2.warpAffine(pad_mask, M, (diag, diag))
    # brightness / contrast jitter (sun, shade)
    alpha, beta = random.uniform(0.75, 1.25), random.uniform(-25, 25)
    crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # partly submerged: lower part fades and turns blue
    if random.random() < 0.35:
        hh = mask.shape[0]
        fade = np.linspace(1.0, random.uniform(0.25, 0.6), hh, dtype=np.float32)[:, None]
        mask = mask * fade
        tint = np.zeros_like(crop)
        tint[:] = (200, 150, 40)          # BGR pool blue
        crop = cv2.addWeighted(crop, 0.75, tint, 0.25, 0)
    if random.random() < 0.3:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
    return crop, mask


def paste(bg, crop, mask, x, y):
    h, w = mask.shape
    H, W = bg.shape[:2]
    if x + w > W or y + h > H:
        return None
    roi = bg[y:y + h, x:x + w].astype(np.float32)
    m = mask[..., None]
    bg[y:y + h, x:x + w] = (roi * (1 - m) + crop.astype(np.float32) * m).astype(np.uint8)
    ys, xs = np.where(mask > 0.35)
    if len(xs) < 20:
        return None
    return x + xs.min(), y + ys.min(), x + xs.max(), y + ys.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root with data.yaml + train/")
    ap.add_argument("--out", required=True, help="where synthetic train images go")
    ap.add_argument("--n", type=int, default=1500, help="synthetic images with objects")
    ap.add_argument("--negatives", type=int, default=300, help="empty-pool images (objects painted out)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    root, out = Path(a.data), Path(a.out)
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    train = root / "train"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    bank = build_bank(train)
    if not bank:
        raise SystemExit("No labelled objects found in train/ - check the dataset path.")
    counts = Counter({c: len(v) for c, v in bank.items()})
    print("object bank:", {names[c]: n for c, n in sorted(counts.items())})
    classes = list(bank)
    weights = [1.0 / max(counts[c], 1) ** 0.5 for c in classes]   # favour rare classes

    bgs = [p for p in (train / "images").iterdir()
           if p.suffix.lower() in IMG_EXT and not p.name.startswith("syn")]
    made = Counter()
    for i in range(a.n):
        bg_path = random.choice(bgs)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        boxes = read_labels(train / "labels" / f"{bg_path.stem}.txt", W, H)
        lines = [to_yolo(*b, W, H) for b in boxes]
        for _ in range(random.randint(1, 3)):
            c = random.choices(classes, weights)[0]
            src = random.choice(bank[c])
            # 30 % small = far away objects
            scale = random.uniform(0.3, 0.6) if random.random() < 0.3 else random.uniform(0.6, 1.4)
            crop, mask = augment_crop(src, scale)
            mh, mw = mask.shape
            if mw >= W - 2 or mh >= H - 2:
                continue
            for _try in range(15):
                x, y = random.randint(0, W - mw - 1), random.randint(0, H - mh - 1)
                if not overlaps((x, y, x + mw, y + mh), boxes):
                    box = paste(bg, crop, mask, x, y)
                    if box:
                        boxes.append((c, *box))
                        lines.append(to_yolo(c, *box, W, H))
                        made[c] += 1
                    break
        cv2.imwrite(str(out / "images" / f"syn_{i:05d}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / f"syn_{i:05d}.txt").write_text("\n".join(lines))

    for i in range(a.negatives):
        bg_path = random.choice(bgs)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        boxes = read_labels(train / "labels" / f"{bg_path.stem}.txt", W, H)
        if not boxes:
            continue
        m = np.zeros((H, W), np.uint8)
        for _, x1, y1, x2, y2 in boxes:
            pad = int(0.15 * max(x2 - x1, y2 - y1)) + 4
            cv2.rectangle(m, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 255, -1)
        clean = cv2.inpaint(bg, m, 7, cv2.INPAINT_TELEA)
        cv2.imwrite(str(out / "images" / f"synneg_{i:05d}.jpg"), clean, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / f"synneg_{i:05d}.txt").write_text("")   # empty = background

    print("pasted objects:", {names[c]: n for c, n in sorted(made.items())})
    print(f"wrote {a.n} synthetic + {a.negatives} empty-pool images to {out}")


if __name__ == "__main__":
    main()
