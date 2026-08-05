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
