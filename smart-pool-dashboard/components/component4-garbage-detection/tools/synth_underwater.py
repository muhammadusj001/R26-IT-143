"""Make UNDER-WATER versions of your existing training images (no new photos).

Water absorbs red light, adds a blue-green haze and blurs edges. This tool
applies that physical effect (random strength) to each image of a YOLO
dataset and copies the label file unchanged, so the boxes stay correct.
Training on original + underwater images teaches the model to find
submerged / sunk garbage (panel feedback).

    python tools/synth_underwater.py --data "path/to/dataset" --out "path/to/underwater_dataset"

The dataset folder must contain train/images + train/labels (valid/ too if
present). Upload the --out folder to Kaggle as a Dataset and fine-tune with
ml/train_v2_kaggle.ipynb (MODE = "finetune"); it finds every images/labels
folder under /kaggle/input automatically.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def underwater(img, rng):
    a = rng.uniform(0.25, 0.6)                 # haze / depth
    red = rng.uniform(0.25, 0.6)               # red light left
    green = rng.uniform(0.8, 0.95)
    blur = rng.uniform(0.6, 2.0)
    water = np.array([rng.uniform(150, 200), rng.uniform(120, 170), rng.uniform(20, 70)],
                     np.float32)               # B, G, R pool-water colour
    out = img.astype(np.float32) * np.array([1.0, green, red], np.float32)
    out = (1 - a) * out + a * water
    out = cv2.GaussianBlur(out, (0, 0), blur)
    # soft light ripples (caustics)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f1, f2, ph = rng.uniform(0.01, 0.04), rng.uniform(0.01, 0.04), rng.uniform(0, 6.28)
    caustic = (np.sin(xx * f1 + ph) * np.sin(yy * f2 + ph * 0.7)) * rng.uniform(4, 12)
    out += caustic[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO dataset folder (has train/images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src, dst = Path(args.data), Path(args.out)
    total = 0
    for split in args.splits.split(","):
        img_dir, lab_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            continue
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(p))
            lab = lab_dir / (p.stem + ".txt")
            if img is None or not lab.exists():
                continue
            name = f"uw_{p.stem}"
            cv2.imwrite(str(dst / split / "images" / f"{name}.jpg"), underwater(img, rng))
            shutil.copy(lab, dst / split / "labels" / f"{name}.txt")
            total += 1
    print(f"made {total} underwater images in {dst}")


if __name__ == "__main__":
    main()
