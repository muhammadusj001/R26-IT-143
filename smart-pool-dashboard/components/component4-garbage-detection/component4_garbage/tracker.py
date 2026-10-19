"""
Lightweight multi-object tracker (IoU + centre distance), no extra installs.

Why tracking matters for a LIVE camera:
  * A single-frame detector flickers: the same bottle is found in frame 1,
    missed in frame 2, found in frame 3. Without tracking you get repeated
    alerts and random false alarms from one bad frame.
  * With tracking each real object gets ONE id. We alert only after the
    object is confirmed in N of the last M frames, and only ONCE per id.
  * The track keeps the object's path over time, which is what lets us
    measure how it was launched (thrown vs dropped) and notice when it
    VANISHES inside the pool (= it sank).
"""

import math
from collections import deque
from itertools import count


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class Track:
    _ids = count(1)

    def __init__(self, box, label, conf, ts, window):
        self.id = next(Track._ids)
        self.box = box
        self.label = label
        self.conf = conf
        self.first_seen = ts
        self.last_seen = ts
        self.points = deque(maxlen=120)        # (ts, cx, cy)
        self.boxes = deque(maxlen=120)         # (ts, box)
        self.hits = deque(maxlen=window)       # 1 = seen in that frame
        self.confirmed = False
        self.confirmed_at = None
        self.alerted = False
        self.last_alert = -1e9
        self.ended = False
        self.meta = {}                         # free slot for event logic
        self._add(box, ts)

    def _add(self, box, ts):
        cx, cy = center(box)
        self.points.append((ts, cx, cy))
        self.boxes.append((ts, box))
        self.hits.append(1)

    def update(self, box, label, conf, ts):
        self.box = box
        self.label = label
        self.conf = max(conf, self.conf * 0.9)
        self.last_seen = ts
        self._add(box, ts)

    def mark_missed(self):
        self.hits.append(0)

    def box_at(self, ts):
        """Box nearest in time to ts (used for looking back in history)."""
        best = None
        best_dt = 1e9
        for t, b in self.boxes:
            dt = abs(t - ts)
            if dt < best_dt:
                best, best_dt = b, dt
        return best


class SimpleTracker:
    def __init__(self, iou_thr=0.2, dist_ratio=0.08, max_age_sec=1.5,
                 confirm_frames=4, confirm_window=8, same_label_only=False):
        self.iou_thr = iou_thr
        self.dist_ratio = dist_ratio
        self.max_age_sec = max_age_sec
        self.confirm_frames = confirm_frames
        self.confirm_window = confirm_window
        self.same_label_only = same_label_only
        self.tracks = []

    def update(self, detections, ts, frame_diag):
        """detections: list of (box, label, conf). Returns
        (active_tracks, newly_confirmed_tracks, ended_tracks)."""
        unmatched = list(range(len(detections)))
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di in unmatched:
                box, label, _ = detections[di]
                if self.same_label_only and label != tr.label:
                    continue
                score = iou(tr.box, box)
                if score < self.iou_thr:
                    (tx, ty), (dx, dy) = center(tr.box), center(box)
                    dist = math.hypot(tx - dx, ty - dy)
                    if dist > self.dist_ratio * frame_diag:
                        continue
                    score = 0.01 + 0.19 * (1 - dist / (self.dist_ratio * frame_diag))
                pairs.append((score, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            box, label, conf = detections[di]
            self.tracks[ti].update(box, label, conf, ts)

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.mark_missed()
        for di in range(len(detections)):
            if di not in used_d:
                box, label, conf = detections[di]
                self.tracks.append(Track(box, label, conf, ts, self.confirm_window))

        newly_confirmed = []
        for tr in self.tracks:
            if not tr.confirmed and sum(tr.hits) >= self.confirm_frames:
                tr.confirmed = True
                tr.confirmed_at = ts
                newly_confirmed.append(tr)

        ended, alive = [], []
        for tr in self.tracks:
            if ts - tr.last_seen > self.max_age_sec:
                tr.ended = True
                ended.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        active = [t for t in alive if t.last_seen == ts]
        return active, newly_confirmed, ended
