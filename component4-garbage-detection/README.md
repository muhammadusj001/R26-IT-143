# Component 4 — Garbage Intrusion Detection
**Owner:** TODO — add member name
**AI Task:** Detects foreign objects in the pool (YOLOv8n, 6 classes) and alerts only on true garbage — ball/leaf are detected but excluded.

## Where everything lives
| What | File | Purpose |
|---|---|---|
| AI inference | `component4_garbage/detector.py` | loads the YOLOv8n model, runs detection per frame |
| Business logic | `component4_garbage/detector.py` (same file) | `GARBAGE_CLASSES` disambiguation — ball/leaf are detected but never raise an alert |
| Utilities | `component4_garbage/drawing.py` | bounding-box drawing helpers |
| Trained model | `models/swimming_pool_garbage_yolo.pt` (6.25 MB) | YOLOv8n, 6-class garbage-detection weights |
| Training code | `ml/train.ipynb` (also `ml/GarbageDetection.ipynb`) | how the model was trained |
| Evaluation code | `ml/detect.ipynb`; raw per-epoch numbers in `ml/runs/pool_yolo_model-4/results.csv` | how metrics were produced |
| Dataset config | `ml/data.yaml` | classes and split paths (note: lives at `ml/data.yaml`, not `ml/datasets/data.yaml`) |
| Standalone demo | `standalone_demo.py` | run this component alone |

## Model performance
| Metric | Value |
|---|---|
| Precision | 0.853 |
| Recall | 0.878 |
| mAP50 | 0.903 |
| mAP50-95 | 0.393 |
| Epochs | 10 (CPU, yolov8n) |
| Caveat | Only 25 test images — `MODEL_ASSESSMENT.md` flags this as too few for credible claims |

## How to run this component alone
```bash
pip install -r requirements.txt
python standalone_demo.py
```

---

# Component 4 — Garbage Intrusion Detection (R26-IT-143)

YOLOv8n (6 classes) with class disambiguation: only actual garbage
(aluminium foil, bottle, juice, thermocol) raises alerts; `ball` (toy)
and `leaf` are detected but excluded — reducing false positives on
legitimate pool objects.

Metrics (10 epochs, CPU): P=0.853 · R=0.878 · mAP50=0.903 · mAP50-95=0.393

## Structure
- `component4_garbage/` — importable package (detector) — NEW, componentized
- `legacy_flask_app/` — the original standalone Flask app (upload/video/webcam)
- `ml/` — training notebooks + best run results + data.yaml
- `models/` — trained swimming_pool_garbage_yolo.pt
- `standalone_demo.py` — run alone for the viva

## Run standalone
```bash
pip install -r requirements.txt
python standalone_demo.py
```

## REQUIRED dataset attribution (license obligation)
The dataset is the public Roboflow "swimming-pool" dataset by Bannari Amman
Institute of Technology, CC BY 4.0:
https://universe.roboflow.com/bannari-mman-institute-of-technology/swimming-pool-f2wcg/dataset/1
CC BY 4.0 requires attribution — cite it in the thesis and mention it at
the viva. Also improve: only 25 test images (too few for credible metrics);
re-train with more epochs on GPU and evaluate on a larger test split.
