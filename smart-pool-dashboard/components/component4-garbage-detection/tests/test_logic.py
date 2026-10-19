"""Logic tests with FAKE detections (no camera, no GPU, no model files needed).

    python -m pytest tests -q        (or)        python tests/test_logic.py

They prove the event rules behave as designed: thrown -> intentional/HIGH,
dropped -> accidental/MEDIUM, vanished + splash -> SUNK_OBJECT,
throw from the deck with nothing floating -> THROW_NO_FLOATING_OBJECT,
and one object -> one alert (no repeated alerts every frame).
"""
import json
import sys
import tempfile
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from component4_garbage import config  # noqa: E402
from component4_garbage.detector import GarbageDetector  # noqa: E402
from component4_garbage.intent_analyzer import IntentAnalyzer  # noqa: E402

W, H, FPS = 1280, 720, 20.0
PERSON_BOX = (100, 200, 260, 640)          # standing on the left deck
POOL = {"polygon": [[0.30, 0.25], [0.98, 0.25], [0.98, 0.98], [0.30, 0.98]]}


def pose_entry(ts, wrist_xy, body_h=440):
    sx, sy = 200, 300
    kp = {"l_sh": (sx - 40, sy, 0.9), "r_sh": (sx, sy, 0.9),
          "l_el": (sx - 50, sy + 80, 0.9), "r_el": ((sx + wrist_xy[0]) / 2, (sy + wrist_xy[1]) / 2, 0.9),
          "l_wr": (sx - 55, sy + 160, 0.9), "r_wr": (wrist_xy[0], wrist_xy[1], 0.9),
          "l_hip": (sx - 35, sy + 170, 0.9), "r_hip": (sx, sy + 170, 0.9)}
    return {"ts": ts, "box": PERSON_BOX, "body_h": float(body_h), "kp": kp}


def make_detector(tmp):
    roi = Path(tmp) / "roi.json"
    roi.write_text(json.dumps(POOL))
    config.FLOOR_CHANGE_ENABLED = False
    config.EVENTS_DIR = Path(tmp) / "events"
    config.GARBAGE_EVERY_N = 1
    config.ALREADY_PRESENT_SEC = 0.5      # test scenes start 1 s before the action
    det = GarbageDetector(model_path=Path(tmp) / "none.pt", roi_path=roi, log_events=False)
    det.intent_analyzer = IntentAnalyzer(clf_path=Path(tmp) / "no_clf.joblib")
    det.model = object()                      # pretend a model is loaded
    det.source_identification_enabled = True
    det.pose.available = True
    det.pose.status = "fake"
    return det


def run(det, script, n_frames):
    """script(i, ts) -> (garbage_dets, persons, litter, wrist_xy or None, splash_xy or None)"""
    events = []
    state = {}

    def fake_garbage(_frame):
        return state["g"]

    def fake_sources(_frame):
        return state["p"], [], state["l"]

    def fake_pose(_frame, person_tracks, ts):
        if state["wrist"] is None:
            return
        for tr in person_tracks:
            det.pose.people.setdefault(tr.id, deque()).append(pose_entry(ts, state["wrist"]))

    det._detect_garbage = fake_garbage
    det._detect_sources = fake_sources
    det.pose.update = fake_pose
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        ts = 1000 + i / FPS
        g, p, l, wrist, splash = script(i, ts)
        state.update(g=g, p=p, l=l, wrist=wrist)
        frame = np.full((H, W, 3), (200, 160, 60), np.uint8)
        if splash:
            x, y = splash                       # white foam blobs that change every frame
            for _ in range(6):
                dx, dy = rng.integers(-45, 45, 2)
                cv2.circle(frame, (int(x + dx), int(y + dy)), int(rng.integers(8, 20)),
                           (255, 255, 255), -1)
        _, result, _ = det.process(frame, ts)
        events += result["events"]
    det.close()          # release MediaPipe explicitly (its __del__ can hang)
    return events


def test_thrown_bottle_is_intentional():
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)

        def script(i, ts):
            persons = [(PERSON_BOX, "person", 0.9)]
            # arm swings fast & high for frames 10-18 (throw), then rests
            if 10 <= i <= 18:
                k = i - 10
                wrist = (200 + 30 * k, 420 - 45 * k)
            else:
                wrist = (205, 470)
            litter = []
            if 16 <= i <= 22:                        # bottle flying sideways into pool
                x = 300 + 90 * (i - 16)
                litter = [((x - 15, 250, x + 15, 290), "bottle", 0.6)]
            g = []
            if i >= 24:                              # floating bottle in the pool
                g = [((860, 400, 900, 440), "bottle", 0.8)]
            return g, persons, litter, wrist, None

        events = run(det, script, 60)
        garbage = [e for e in events if e["type"] == "GARBAGE_IN_POOL"]
        assert len(garbage) == 1, events                      # one object -> one alert
        assert garbage[0]["source"] == "human"
        assert garbage[0]["intent"] == "intentional", garbage[0]
        assert garbage[0]["risk_level"] == "HIGH"


def test_dropped_item_is_accidental():
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)
        box = (440, 200, 600, 640)                           # person standing at the pool edge

        def script(i, ts):
            persons = [(box, "person", 0.9)]
            wrist = (205 + (i % 2), 470)                     # arm almost still
            litter = []
            if 14 <= i <= 20:                                # falls straight down
                y = 450 + 25 * (i - 14)
                litter = [((600, y - 15, 630, y + 15), "cup", 0.6)]
            g = [((598, 610, 640, 650), "juice", 0.8)] if i >= 22 else []
            return g, persons, litter, wrist, None

        events = run(det, script, 60)
        garbage = [e for e in events if e["type"] == "GARBAGE_IN_POOL"]
        assert len(garbage) == 1, events
        assert garbage[0]["intent"] == "accidental", garbage[0]
        assert garbage[0]["risk_level"] == "MEDIUM"


def test_heavy_object_vanishes_with_splash_is_sunk():
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)

        def script(i, ts):
            persons = [(PERSON_BOX, "person", 0.9)] if i < 40 else []
            litter, splash = [], None
            if 10 <= i <= 18:                                # thrown bottle, flies into pool
                x = 320 + 60 * (i - 10)
                litter = [((x - 15, 380, x + 15, 420), "bottle", 0.6)]
            if 19 <= i <= 26:
                splash = (820, 400)                          # splash where it vanished
            return [], persons, litter, None, splash

        events = run(det, script, 140)
        sunk = [e for e in events if e["type"] == "SUNK_OBJECT"]
        assert len(sunk) == 1, events
        assert sunk[0]["risk_level"] == "HIGH" and sunk[0]["splash_strength"] > 0


def test_throw_from_deck_with_nothing_floating():
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)

        def script(i, ts):
            persons = [(PERSON_BOX, "person", 0.9)]
            if 10 <= i <= 18:
                k = i - 10
                wrist = (200 + 30 * k, 420 - 45 * k)
            else:
                wrist = (205, 470)
            return [], persons, [], wrist, None

        events = run(det, script, 120)
        assert any(e["type"] == "THROW_NO_FLOATING_OBJECT" for e in events), events


def test_swimmer_strokes_are_not_throws():
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)
        swimmer = (700, 300, 860, 500)                       # inside the pool area

        def script(i, ts):
            k = i % 8
            wrist = (200 + 30 * k, 420 - 45 * k)              # fast arm, like a stroke
            return [], [(swimmer, "person", 0.9)], [], wrist, None

        events = run(det, script, 120)
        assert not any(e["type"] == "THROW_NO_FLOATING_OBJECT" for e in events), events


def test_garbage_present_at_start_is_not_blamed():
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)

        def script(i, ts):
            wrist = (200 + 30 * (i % 8), 420 - 45 * (i % 8))   # someone waving
            return ([((860, 400, 900, 440), "bottle", 0.8)],
                    [(PERSON_BOX, "person", 0.9)], [], wrist, None)

        events = run(det, script, 40)
        garbage = [e for e in events if e["type"] == "GARBAGE_IN_POOL"]
        assert len(garbage) == 1 and garbage[0]["intent"] == "already_present", garbage


def test_scene_cut_does_not_fake_sunk_objects():
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)
        state = {"i": 0}
        orig = det.process

        def process(frame, ts=None, clean_frame=None):
            if state["i"] >= 30:                       # new "scene": very different picture
                frame[:] = (40, 40, 40)
            state["i"] += 1
            return orig(frame, ts, clean_frame)
        det.process = process

        def script(i, ts):
            g = [((860, 400, 900, 440), "bottle", 0.8)] if i < 30 else []
            return g, [], [], None, None

        events = run(det, script, 120)
        assert not any(e["type"] == "SUNK_OBJECT" for e in events), events


def test_late_garbage_is_linked_to_earlier_throw_action():
    """Garbage confirmed ~5 s after the throw (it had to settle first): the
    pose look-back alone misses the swing, the human-action link catches it."""
    with tempfile.TemporaryDirectory() as tmp:
        det = make_detector(tmp)

        def script(i, ts):
            persons = [(PERSON_BOX, "person", 0.9)]
            if 10 <= i <= 18:
                k = i - 10
                wrist = (200 + 30 * k, 420 - 45 * k)
            else:
                wrist = (205, 470)
            g = [((860, 400, 900, 440), "bottle", 0.8)] if i >= 115 else []
            return g, persons, [], wrist, None

        events = run(det, script, 150)
        garbage = [e for e in events if e["type"] == "GARBAGE_IN_POOL"]
        assert len(garbage) == 1, events
        assert garbage[0]["intent"] == "intentional", garbage[0]
        assert "throwing motion" in garbage[0]["message"], garbage[0]


def _uw_detector(tmp, uw_box, from_frame=0):
    det = make_detector(tmp)
    saved = config.TILES
    config.TILES = 0                     # only the underwater pass in these tests

    def fake_crops(model, crops, offsets, conf, imgsz, ov=False):
        if det._frame_idx < from_frame:
            return []
        return [(uw_box, "bottle", 0.6)]
    det._run_on_crops = fake_crops
    return det, saved


def test_submerged_garbage_is_reported_as_underwater():
    with tempfile.TemporaryDirectory() as tmp:
        det, saved = _uw_detector(tmp, (860, 400, 900, 440))
        try:
            def script(i, ts):
                return [], [], [], None, None     # normal search finds nothing
            events = run(det, script, 60)
        finally:
            config.TILES = saved
        uw = [e for e in events if e["type"] == "SUBMERGED_GARBAGE"]
        assert len(uw) == 1 and uw[0]["position"] == "underwater", events
        assert det._feed[0]["title"] == "Garbage under the water"


def test_floating_garbage_that_goes_under_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        det, saved = _uw_detector(tmp, (862, 402, 902, 442), from_frame=45)
        try:
            def script(i, ts):
                # floating (normal model sees it) until frame 50, then only
                # the underwater pass sees it
                g = [((860, 400, 900, 440), "bottle", 0.8)] if 20 <= i < 50 else []
                return g, [], [], None, None
            events = run(det, script, 90)
        finally:
            config.TILES = saved
        types = [e["type"] for e in events]
        assert types.count("GARBAGE_IN_POOL") == 1, events
        assert any(e["type"] == "SUNK_OBJECT" and "UNDER the surface" in e["message"]
                   for e in events), events


def test_tiles_and_merge_helpers():
    from component4_garbage import enhance as enh
    tiles = enh.make_tiles(0, 0, 1000, 600, 2, 0.2)
    assert len(tiles) == 4 and tiles[0][:2] == (0, 0) and tiles[-1][2:] == (1000, 600)
    base = [((100, 100, 140, 140), "bottle", 0.9)]
    extra = [((102, 101, 141, 139), "bottle", 0.5),     # duplicate of base
             ((500, 300, 510, 310), "can", 0.4),        # new tiny object
             ((500, 300, 506, 310), "can", 0.3)]        # half box at a tile edge
    new = enh.merge_new(base, extra)
    assert [d[1] for d in new] == ["can"], new
    water = np.full((200, 200, 3), (190, 150, 40), np.uint8)
    low = water.copy()
    cv2.rectangle(low, (90, 90), (110, 110), (175, 140, 50), -1)    # blue-tinted, faint
    high = water.copy()
    cv2.rectangle(high, (90, 90), (110, 110), (250, 250, 250), -1)  # bright floating
    assert enh.box_contrast(low, (90, 90, 110, 110)) < config.UNDERWATER_CONTRAST
    assert enh.box_contrast(high, (90, 90, 110, 110)) > config.UNDERWATER_CONTRAST
    out = enh.enhance_underwater(water)
    assert out.shape == water.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
