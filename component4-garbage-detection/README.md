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
