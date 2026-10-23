"""
Helpers for the two panel requests:

  * very TINY garbage  -> tiled ("zoomed") detection over the pool area
  * garbage UNDER the water surface -> underwater colour correction, then a
    second search; objects found only after correction and with low
    contrast against the surrounding water are marked "underwater".

Pure OpenCV / NumPy, no extra installs.
"""

import cv2
import numpy as np

from component4_garbage.tracker import iou


# ── underwater image enhancement ──────────────────────────────────
def enhance_underwater(img):
    """Colour-correct an underwater / through-the-water image.

    1. red-channel compensation (water absorbs red first, Ancuti et al. 2018)
    2. grey-world white balance (removes the blue-green cast)
    3. CLAHE on the lightness channel (brings back local contrast/edges)
    """
    f = img.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    r = r + 1.0 * (g.mean() - r.mean()) * (1.0 - r) * g
    b = b + 0.5 * (g.mean() - b.mean()) * (1.0 - b) * g
    f = np.stack([np.clip(b, 0, 1), g, np.clip(r, 0, 1)], axis=-1)
    means = f.reshape(-1, 3).mean(0) + 1e-6
    f = np.clip(f * (means.mean() / means), 0, 1)
    u8 = (f * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def box_contrast(frame, box):
    """Lab colour distance between the inside of a box and a ring of water
    around it. Floating garbage (bright, sharp, reflections) -> high;
    submerged garbage (blue-tinted, blurred) -> low."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    pad = max(4, int(0.5 * max(x2 - x1, y2 - y1)))
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    region = cv2.cvtColor(frame[Y1:Y2, X1:X2], cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = np.ones(region.shape[:2], bool)
    mask[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
    inside = region[~mask]
    ring = region[mask]
    if len(ring) < 10 or len(inside) < 4:
        return 0.0
    return float(np.linalg.norm(inside.mean(0) - ring.mean(0)))


def bright_fraction(frame, box):
    """Share of near-white pixels in the box: sun glare / reflections only
    happen on objects AT the surface."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float((gray > 200).mean())


# ── tiled detection for tiny objects ──────────────────────────────
def make_tiles(x0, y0, x1, y1, n, overlap):
    """Split the region into n x n overlapping tiles -> list of (x1,y1,x2,y2)."""
    rw, rh = x1 - x0, y1 - y0
    if n <= 1:
        return [(x0, y0, x1, y1)]
    tw = int(min(rw, rw / n * (1 + overlap)))
    th = int(min(rh, rh / n * (1 + overlap)))
    xs = [x0 + int(i * (rw - tw) / (n - 1)) for i in range(n)]
    ys = [y0 + int(j * (rh - th) / (n - 1)) for j in range(n)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def _overlap_small(a, b):
    """Intersection over the SMALLER box (catches a half box cut by a tile edge)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(min(sa, sb) + 1e-6)


def is_duplicate(box, others, iou_thr=0.4, small_thr=0.6):
    return any(iou(box, o) > iou_thr or _overlap_small(box, o) > small_thr for o in others)


def nms(dets, iou_thr=0.4, small_thr=0.6):
    """Class-agnostic NMS on [(box, label, conf), ...], keeps the most confident."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[2]):
        if not is_duplicate(d[0], [k[0] for k in keep], iou_thr, small_thr):
            keep.append(d)
    return keep


def merge_new(base, extra):
    """Detections from `extra` that are not already in `base`."""
    boxes = [b[0] for b in base]
    out = []
    for d in nms(extra):
        if not is_duplicate(d[0], boxes):
            out.append(d)
            boxes.append(d[0])
    return out
