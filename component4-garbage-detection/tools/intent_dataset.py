"""Record, evaluate and (optionally) train the THROW vs ACCIDENTAL decision.

Do you need to collect action data?  YES, a SMALL amount. The rule in
intent_analyzer.py works without training, but you cannot claim any
accuracy in the thesis without labelled test clips, and the thresholds
must be tuned for YOUR camera. 3 labels, short clips (3-6 s, one action
each), from your Iriun camera at the real pool angle:

    intentional/  person throws a bottle/cup/wrapper into the pool
    accidental/   object slips from hand / falls from pocket, bag, bench
    normal/       person walks, drinks, waves, stretches - NO litter

Aim for 40+ clips per label, several people, both hands, different
distances. (Use empty plastic bottles / foam, never glass, and fish
them out afterwards!)

1) Record clips (keys: i = intentional, a = accidental, n = normal, q = quit;
   press the key, do the action, the clip stops after --seconds):
       python tools/intent_dataset.py record --source 1 --seconds 5

2) Check how good the RULE is on your clips (accuracy + confusion matrix):
       python tools/intent_dataset.py evaluate

3) Train a small classifier on the same features (saved to
   models/intent_clf.joblib and then used automatically by the detector):
       python tools/intent_dataset.py train
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from component4_garbage import config  # noqa: E402
from component4_garbage.intent_analyzer import FEATURE_ORDER, IntentAnalyzer, PoseHistory  # noqa: E402
from component4_garbage.tracker import SimpleTracker  # noqa: E402

CLIPS = config.ROOT / "intent_clips"
LABELS = {"i": "intentional", "a": "accidental", "n": "normal"}


def record(args):
    from component4_garbage.camera import LiveCamera
    cam = LiveCamera(args.source)
    for lab in LABELS.values():
        (CLIPS / lab).mkdir(parents=True, exist_ok=True)
    writer, until, label = None, 0, None
    try:
        while True:
            ok, frame, _ = cam.read()
            if not ok:
                break
            now = time.time()
            if writer is not None:
                writer.write(frame)
                if now >= until:
                    writer.release()
                    writer = None
                    print("saved", label)
            view = frame.copy()
            counts = {k: len(list((CLIPS / k).glob("*.mp4"))) for k in LABELS.values()}
            txt = ("REC " + label) if writer else "i=intentional a=accidental n=normal q=quit"
            cv2.putText(view, f"{txt} | {counts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if writer else (0, 255, 255), 2)
            cv2.imshow("record intent clips", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if writer is None and k < 128 and chr(k) in LABELS:
                label = LABELS[chr(k)]
                h, w = frame.shape[:2]
                path = CLIPS / label / f"{label}_{int(now)}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                until = now + args.seconds
    finally:
        if writer is not None:
            writer.release()
        cam.release()
        cv2.destroyAllWindows()


def clip_features(path, people_model, analyzer):
    """Runs people/pose/litter tracking over a clip and returns features
    around the moment of the fastest arm movement."""
    pose = PoseHistory(history_sec=60)
    pt = SimpleTracker(confirm_frames=1, confirm_window=2, max_age_sec=1.0, dist_ratio=0.12)
    lt = SimpleTracker(confirm_frames=1, confirm_window=3, max_age_sec=0.8, iou_thr=0.1,
                       dist_ratio=0.15)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    litter_tracks, idx, h = [], 0, 720
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        idx += 1
        h, w = frame.shape[:2]
        diag = (w * w + h * h) ** 0.5
        persons, litter = [], []
        res = people_model(frame, conf=0.35, imgsz=640, verbose=False,
                           classes=[0] + list(config.COCO_LITTER_CLASSES))
        for r in res:
            for b in r.boxes:
                cid = int(b.cls[0])
                box = tuple(map(int, b.xyxy[0].tolist()))
                (persons if cid == 0 else litter).append((box, "x", float(b.conf[0])))
        pa, _, _ = pt.update(persons, ts, diag)
        lt.update(litter, ts, diag)
        pose.update(frame, pa, ts)
        litter_tracks = list({id(t): t for t in litter_tracks + lt.tracks}.values())
    cap.release()
    # main person = longest pose history
    if not pose.people:
        feats = {k: 0.0 for k in FEATURE_ORDER}
        return feats, analyzer.classify([], [], h)
    pid = max(pose.people, key=lambda k: len(pose.people[k]))
    entries = list(pose.people[pid])
    f = analyzer.pose_features
    best_t, best_s = entries[0]["ts"], -1
    for a, b in zip(entries, entries[1:]):
        s = f([a, b])["wrist_peak_speed"]
        if s > best_s:
            best_s, best_t = s, b["ts"]
    window = [e for e in entries if best_t - 1.5 <= e["ts"] <= best_t + 1.0]
    path = []
    for t in litter_tracks:
        pts = [p for p in t.points if best_t - 1.0 <= p[0] <= best_t + 1.5]
        if len(pts) > len(path):
            path = pts
    out = analyzer.classify(window, path, h)
    return out["signals"], out


def evaluate(args, train=False):
    from ultralytics import YOLO
    model = YOLO(str(config.SOURCE_MODEL_PATH) if config.SOURCE_MODEL_PATH.exists()
                 else "yolov8n.pt")
    rule = IntentAnalyzer(clf_path=Path("__none__"))
    rows = []
    for lab in LABELS.values():
        for clip in sorted((CLIPS / lab).glob("*.mp4")):
            feats, out = clip_features(clip, model, rule)
            # "normal" clips: the rule should NOT call them intentional
            pred = out["intent"] if out["intent"] in ("intentional", "accidental") else "accidental"
            rows.append({"clip": clip.name, "label": lab, "rule_pred": pred, **feats})
            print(f"{clip.name:40s} label={lab:12s} rule={pred:12s} reason={out.get('reason')}")
    if not rows:
        sys.exit(f"No clips found in {CLIPS}. Record some first.")
    out_csv = CLIPS / "features.csv"
    with out_csv.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("features saved to", out_csv)

    # rule accuracy on intentional vs (accidental + normal)
    y_true = ["intentional" if r["label"] == "intentional" else "accidental" for r in rows]
    y_rule = [r["rule_pred"] for r in rows]
    correct = sum(a == b for a, b in zip(y_true, y_rule))
    print(f"\nRULE accuracy (intentional vs not): {correct}/{len(rows)} = {correct / len(rows):.1%}")
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_true, y_rule, labels=["intentional", "accidental"]))
        print(classification_report(y_true, y_rule, zero_division=0))
    except ImportError:
        pass

    if train:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import classification_report
        X = [[float(r[k]) for k in FEATURE_ORDER] for r in rows]
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0,
                                     class_weight="balanced")
        folds = min(5, min(y_true.count("intentional"), y_true.count("accidental")))
        if folds >= 2:
            pred = cross_val_predict(clf, X, y_true, cv=StratifiedKFold(folds, shuffle=True,
                                                                        random_state=0))
            print(f"\nTRAINED classifier ({folds}-fold cross-validation):")
            print(classification_report(y_true, pred, zero_division=0))
        clf.fit(X, y_true)
        joblib.dump(clf, config.INTENT_CLF_PATH)
        print("saved", config.INTENT_CLF_PATH, "- the detector will use it automatically.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--source", default="0")
    r.add_argument("--seconds", type=float, default=5)
    sub.add_parser("evaluate")
    sub.add_parser("train")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a)
    elif a.cmd == "evaluate":
        evaluate(a)
    else:
        evaluate(a, train=True)
