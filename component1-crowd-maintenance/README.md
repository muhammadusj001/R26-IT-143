# Component 1 — Crowd-Aware Maintenance Scheduling
**Owner:** TODO — add member name
**AI Task:** Detects swimmers per camera frame and estimates real-time pool crowd density.

## Where everything lives
| What | File | Purpose |
|---|---|---|
| AI inference | `component1_crowd/detector.py` | loads the YOLO swimmer/COCO-person model, runs detection per frame, computes density level |
| Business logic | `component1_crowd/bather_load.py`, `component1_crowd/scheduler.py` | bather-load (person-hours) accumulation; rule-based maintenance recommendations with priority escalation |
| Utilities | `component1_crowd/drawing.py` | bounding-box and label drawing helpers |
| Trained model | `models/best_swimmer_model.pt` (6.23 MB, default/v1); `models/best_swimmer_model_v2_combined.pt` (6.25 MB, combined-dataset variant) | swimmer-detection YOLOv8 weights |
| Training code | `ml/scripts/train_custom_model.py` | how the model was trained |
| Evaluation code | `ml/scripts/evaluate_custom_model.py` | how metrics were produced |
| Dataset config | `ml/datasets/data.yaml` | 1 class ("swimmer"); train/valid/test split paths |
| Standalone demo | `standalone_demo.py` | run this component alone |

## Model performance
| Metric | Value |
|---|---|
| Reported accuracy — v1 (single-source dataset, the default model) | ~85.5% (per `MODEL_ASSESSMENT.md`) |
| Reported accuracy — v2 (combined-dataset variant) | ~80.6% |
| Precision / Recall / mAP50 / mAP50-95 | TODO — no metrics file exists in the repo yet; re-run `ml/scripts/evaluate_custom_model.py` to generate these for the thesis |

## How to run this component alone
```bash
pip install -r requirements.txt
python standalone_demo.py
```

---

# Component 1 — Crowd-Aware Maintenance Scheduling (R26-IT-143)

YOLOv8 swimmer detection → density level → bather-load accumulation
(person-hours) → rule-based maintenance recommendations with priority
escalation (MEDIUM → HIGH at 1.5× → CRITICAL at 2× threshold).

## Structure
- `component1_crowd/` — importable package (detector, bather_load, scheduler)
- `ml/` — training & evaluation scripts (YOLOv8m workflow), dataset config
- `models/` — trained swimmer models (v1 = default, v2_combined = larger dataset)
- `standalone_demo.py` — run this component alone for the viva

## Run standalone
```bash
pip install -r requirements.txt
python standalone_demo.py
```

## Research note (viva)
v1 (single-source dataset) outperformed v2 (combined datasets) — an example
of the dataset-heterogeneity problem: combining sources with different camera
angles/lighting/annotation styles reduced generalization consistency.

## Note on recovered files
`bather_load.py` and `scheduler.py` were reconstructed from compiled
bytecode (the original .py sources were lost); logic is identical.
