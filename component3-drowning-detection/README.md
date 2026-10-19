# Component 3 — Swimmer Behaviour & Drowning Detection
**Owner:** TODO — add member name
**AI Task:** Detects Swimming / Drowning / Person-out-of-water per frame and raises an alert after 3 consecutive drowning frames.

## Where everything lives
| What | File | Purpose |
|---|---|---|
| AI inference | `component3_drowning/detector.py` | loads the YOLOv8 model (3 classes), runs detection per frame |
| Business logic | `component3_drowning/detector.py` (same file) | 3-consecutive-frame drowning alert rule and SAFE/DANGER status — combined with inference, no separate module |
| Utilities | `component3_drowning/drawing.py` | bounding-box and alert-banner drawing helpers |
| Trained model | `models/best.pt` (19.95 MB) | YOLOv8 Swimming/Drowning/Out-of-water weights |
| Training code | TODO — not present in this repo. `ml/scripts/pool_live_detection.py` is the original standalone live-inference prototype, not a training script |
| Evaluation code | TODO — not present in this repo; the metrics below are quoted from `MODEL_ASSESSMENT.md` |
| Dataset config | TODO — no `ml/datasets` folder exists for this component |
| Standalone demo | `standalone_demo.py` | run this component alone |

## Model performance
| Metric | Value |
|---|---|
| Precision | 0.877 |
| Recall | 0.883 |
| mAP50 | 0.907 |
| mAP50-95 | 0.594 |
| Epochs | 100 |

## How to run this component alone
```bash
pip install -r requirements.txt
python standalone_demo.py
```

---

# Component 3 — Swimmer Behaviour & Drowning Detection (R26-IT-143)

YOLOv8 (3 classes: Swimming / Drowning / Person out of water) with a
3-consecutive-frame alert rule to suppress false positives.

Test metrics (100 epochs): P=0.877 · R=0.883 · mAP50=0.907 · mAP50-95=0.594

## Structure
- `component3_drowning/` — importable package (detector)
- `ml/` — original live-detection script + full training results/curves
- `models/` — PLACE best.pt HERE (weights were gitignored; copy from the
  training machine: results/drowning_model/weights/best.pt)
- `standalone_demo.py` — run alone for the viva

## Run standalone
```bash
pip install -r requirements.txt
# copy best.pt into models/ first
python standalone_demo.py
```
