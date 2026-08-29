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
