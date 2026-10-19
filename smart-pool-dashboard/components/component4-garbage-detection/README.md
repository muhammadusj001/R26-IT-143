# Component 4 — Intent-Aware Garbage Intrusion Detection (v2, live camera)
**Project:** R26-IT-143 · AI-Based Smart Swimming Pool Monitoring System
**Owner:** TODO — add member name

Live camera only (Iriun / webcam / IP camera / CCTV). No image upload.

## What it does, per live frame
1. **Garbage model** (YOLOv8, 6 classes) finds floating objects. Only `aluminium foil`, `bottle`, `juice`, `thermocol` raise alerts; `ball` and `leaf` are shown but ignored.
2. **People model** (COCO `yolov8n.pt`) finds people, animals, and bottles/cups held in a hand or flying through the air.
3. **Tracker** gives each object one ID → one alert per object, and a path over time.
4. **Pose** (MediaPipe) runs on people all the time and keeps ~4 s of arm history.
5. **Intent**: when garbage appears, look back at the nearest person's arm swing and at how the object was launched → `intentional` / `accidental` (with a reason).
6. **Sinking objects**: vanished-inside-pool + splash, pool-floor change, throw-with-nothing-floating.
7. **Events** → `events/events.jsonl` + a snapshot image (MongoDB too if `MONGO_URI` is set).

8. **Tiny garbage** (v2.3): the pool area is cut into 2x2 overlapping tiles, each enlarged before detection (`C4_TILES`, `C4_TILE_EVERY`).
9. **Garbage under the water** (v2.3): the pool area is colour-corrected for water (red-channel compensation, grey-world white balance, CLAHE) and searched again; objects visible only after correction, with low contrast and no surface glare, are marked UNDERWATER (`C4_UNDERWATER`). Test on 80 frames with simulated water (47 objects): trained model alone found 0-6%; with the correction 30-77% (shallow 77%, medium 49%, deep 30%).
10. **Human action recognition** (v2.3): every visible person gets a live action (`THROWING`, `arm moving`, `standing / still`, `swimming`). Garbage confirmed up to 6 s after a nearby throwing motion is judged intentional.

| Event | Plain title on the dashboard | Risk | Meaning |
|---|---|---|---|
| `GARBAGE_IN_POOL` | Garbage on the water surface | HIGH if a person threw it, MEDIUM otherwise | floating garbage confirmed |
| `SUBMERGED_GARBAGE` | Garbage under the water | HIGH if thrown, MEDIUM otherwise | garbage found below the surface |
| `SUNK_OBJECT` | Object sank | HIGH with splash, MEDIUM without | object vanished inside the pool, or a floating object went under |
| `THROW_NO_FLOATING_OBJECT` | Throw seen, nothing floating | MEDIUM | throw from the deck, nothing floating after it |
| `UNDERWATER_OBJECT` | New object on the pool floor | MEDIUM | new still object on the pool floor |

## Files
| File | Purpose |
|---|---|
| `component4_garbage/detector.py` | main pipeline (`GarbageDetector.process(frame)`), same API as v1 |
| `component4_garbage/intent_analyzer.py` | continuous pose history + throw/drop decision |
| `component4_garbage/submersion.py` | pool area, splash detector, pool-floor change |
| `component4_garbage/tracker.py` | lightweight object tracker |
| `component4_garbage/camera.py` | Iriun / webcam / URL / video input (always newest frame) |
| `component4_garbage/config.py` | every threshold in one place |
| `component4_garbage/events.py` | event log + snapshots (+ optional MongoDB) |
| `standalone_demo.py` | run this component alone |
| `INTEGRATION.md` | exact steps + Claude Code prompt to plug v2 into smart-pool-dashboard |
| `integration_example.py` | C4 alone behind a small web server (test) |
| `tools/list_cameras.py` | find Iriun's camera index |
| `tools/draw_pool_roi.py` | click the water area once per camera position |
| `tools/capture_dataset.py` | collect frames from your own camera for retraining |
| `tools/intent_dataset.py` | record/evaluate/train throw vs accidental clips |
| `component4_garbage/enhance.py` | underwater colour correction, tiles, merging |
| `tools/synth_underwater.py` | makes underwater versions of training images (to retrain for sunk garbage) |
| `tools/synth_pool_data.py` | makes synthetic training images from your existing dataset (no new photos) |
| `tools/auto_label.py` | auto-labels your own videos/frames with YOLO-World (no box drawing) |
| `ml/train_v2_colab.ipynb` | Colab retraining notebook (yolov8s, 960 px, own frames) |
| `tests/test_logic.py` | logic tests with fake detections |

## Run alone
```bash
pip install -r requirements.txt
python tools/list_cameras.py              # Iriun is usually index 1
python tools/draw_pool_roi.py --source 1  # once per camera position
python standalone_demo.py --source 1
```
`models/` must hold `swimming_pool_garbage_yolo.pt`, `pose_landmarker_lite.task`, `yolov8n.pt` (people model) and `pool_garbage_openvocab.pt` (open-vocabulary second detector: paper, plastic bag, food tray, can, close-ups; `C4_OPEN_VOCAB=0` turns it off).
On a Mac where MediaPipe crashes, set `DISABLE_INTENT_ANALYSIS=1`: garbage detection keeps working without intent. If `yolov8n.pt` is missing, it downloads automatically the first time (needs internet).

## Model performance (current weights)
| Metric | Value |
|---|---|
| Weights | `models/swimming_pool_garbage_yolo.pt` = YOLOv8n, 50 epochs, Colab, 640 px |
| Precision / Recall (Roboflow valid split) | 0.932 / 0.927 |
| mAP50 / mAP50-95 | 0.949 / 0.446 |
| Note | Roboflow images only (one pool); valid 154 images, test only 25. These numbers do not reflect your live camera. Re-measure on your own camera frames after retraining (`ml/train_v2_colab.ipynb`). `ml/runs/pool_yolo_model-4` is an older 10-epoch CPU run (mAP50 0.903). |

## Dataset attribution (CC BY 4.0, required)
Roboflow "swimming-pool" dataset, Bannari Amman Institute of Technology:
https://universe.roboflow.com/bannari-mman-institute-of-technology/swimming-pool-f2wcg/dataset/1
