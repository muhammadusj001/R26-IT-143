# Project Structure — R26-IT-143 Smart Swimming Pool Monitoring System

Fast-navigation map for examiners/supervisors. Every path below was verified to exist.

## 1. Directory tree (top 3 levels)

```
R26-IT-143-multirepo-system/
├── component1-crowd-maintenance/     # Member 1's standalone repo — crowd/swimmer detection + maintenance scheduling
├── component2-water-quality/         # Member 2's standalone repo — water sensor readings + XGBoost classifier
├── component3-drowning-detection/    # Member 3's standalone repo — drowning detection (YOLOv8)
├── component4-garbage-detection/     # Member 4's standalone repo — pool garbage/object detection (YOLOv8n)
├── smart-pool-dashboard/             # SHARED integration layer — combines all 4 components into one live dashboard
├── MODEL_ASSESSMENT.md               # Honest model-quality notes for all 4 components (viva prep)
├── README.md                         # Top-level project overview
└── .gitignore                        # Repo-wide ignore rules (model weights, datasets, caches, venv, node_modules)
```

```
component1-crowd-maintenance/
├── component1_crowd/         # importable Python package — the actual runtime code (detector, bather_load, scheduler, drawing)
├── ml/                       # training/research code — NOT imported at runtime
│   ├── datasets/               # train/valid/test images + data.yaml (1 class: swimmer)
│   ├── models/                 # duplicate copies of the trained .pt weights (mirrors models/ below)
│   └── scripts/                # train_custom_model.py, evaluate_custom_model.py, old run history
├── models/                   # the .pt weights actually loaded by detector.py / standalone_demo.py
├── requirements.txt          # pip dependencies for this component alone
├── standalone_demo.py        # run this ONE component in isolation (no dashboard needed)
└── README.md                 # component navigation table + viva notes
```

```
component2-water-quality/
├── component2_water/         # importable Python package — predictor.py (XGBoost) + sensor_reader.py (Arduino/simulate)
├── ml/                       # training/research code — NOT imported at runtime
│   ├── datasets/               # pool_water_quality_augmented_dataset.csv
│   ├── models/                 # duplicate copies of the trained model/scaler/encoder
│   └── scripts/                # train_xgboost.py, original notebook, Arduino sketch (sketch_may9a)
├── models/                   # the .pkl model/scaler/encoder actually loaded by predictor.py
├── requirements.txt
├── standalone_demo.py        # run this ONE component in isolation
└── README.md
```

```
component3-drowning-detection/
├── component3_drowning/      # importable Python package — detector.py (AI inference + alert logic combined), drawing.py
├── ml/
│   └── scripts/                # pool_live_detection.py — ORIGINAL prototype script only (no training code checked in)
├── models/                   # best.pt — the YOLOv8 weights actually loaded
├── requirements.txt
├── standalone_demo.py
└── README.md
```

```
component4-garbage-detection/
├── component4_garbage/       # importable Python package — detector.py (AI inference + class-disambiguation logic), drawing.py
├── legacy_flask_app/         # original pre-integration standalone Flask app (kept for provenance, not used by the dashboard)
├── ml/                       # notebooks (train/detect/eda), data.yaml, runs/ (training curves + results.csv)
├── models/                   # swimming_pool_garbage_yolo.pt — the weights actually loaded
├── requirements.txt
├── standalone_demo.py
└── README.md
```

```
smart-pool-dashboard/
├── backend/                  # SHARED — Flask-SocketIO server: camera loop, decision engine, state, settings
│   ├── config/                  # settings.py — all env-configurable paths/thresholds
│   └── core/                    # camera.py, state.py, decision_engine.py, water_monitor.py
├── components/                # mirror copies of all 4 component packages — what the backend actually imports at runtime
│   ├── component1-crowd-maintenance/
│   ├── component2-water-quality/
│   ├── component3-drowning-detection/
│   └── component4-garbage-detection/
├── frontend/                  # Next.js dashboard UI (npm run dev, port 3000) — talks to backend via socket.io-client
│   ├── app/
│   └── components/
├── frontend-vanilla/          # plain HTML/CSS/JS UI — served directly by Flask at port 5000 (zero-build fallback)
│   ├── static/
│   └── templates/
├── venv/                      # Python virtual environment (not tracked in git)
├── requirements.txt           # pip dependencies for the dashboard/backend
└── README.md                  # integration architecture + run instructions
```

> Note: `smart-pool-dashboard/components/*` are byte-identical mirrors of the root `componentN-*/component*_*` packages. The dashboard only ever imports the copies under `smart-pool-dashboard/components/`.

---

## 2. Which folder belongs to which team member

| Component | Folder | Owner | AI Task |
|---|---|---|---|
| Component 1 | `component1-crowd-maintenance/` | TODO — add member name | Detects swimmers per camera frame and estimates real-time pool crowd density |
| Component 2 | `component2-water-quality/` | TODO — add member name | Classifies pool-water sensor readings (pH, Temperature, Chlorine, Turbidity, TDS) into SAFE / WARNING / CRITICAL |
| Component 3 | `component3-drowning-detection/` | TODO — add member name | Detects Swimming / Drowning / Person-out-of-water per frame, raises an alert after 3 consecutive drowning frames |
| Component 4 | `component4-garbage-detection/` | TODO — add member name | Detects foreign objects in the pool (YOLOv8n, 6 classes); alerts only on true garbage — ball/leaf are excluded |

---

## 3. Where is the logic?

| Layer | Location | What it does |
|---|---|---|
| Data acquisition — camera | `smart-pool-dashboard/backend/core/camera.py` | opens/reads the shared camera or IP-webcam stream; feeds one frame to Components 1, 3, and 4 |
| Data acquisition — sensors | `smart-pool-dashboard/components/component2-water-quality/component2_water/sensor_reader.py` | reads Arduino serial water-chemistry values (pH, Temp, Chlorine, Turbidity, TDS), or simulates them |
| AI inference — Component 1 (Crowd) | `smart-pool-dashboard/components/component1-crowd-maintenance/component1_crowd/detector.py` | YOLO swimmer detection per frame → density level |
| AI inference — Component 2 (Water Quality) | `smart-pool-dashboard/components/component2-water-quality/component2_water/predictor.py` | XGBoost classification → SAFE / WARNING / CRITICAL |
| AI inference — Component 3 (Drowning) | `smart-pool-dashboard/components/component3-drowning-detection/component3_drowning/detector.py` | YOLO Swimming / Drowning / Out-of-water detection |
| AI inference — Component 4 (Garbage) | `smart-pool-dashboard/components/component4-garbage-detection/component4_garbage/detector.py` | YOLOv8n garbage detection + class disambiguation |
| Event processing | `smart-pool-dashboard/backend/app.py` — `vision_loop()` | the shared per-frame loop: reads the camera, runs Components 1/3/4 on the same frame, updates `SystemState`, fires alerts |
| Decision intelligence (cross-module) | `smart-pool-dashboard/backend/core/decision_engine.py` | the ONLY place all 4 modules' outputs are combined — computes `overall_risk` and `maintenance_urgency` |
| Visualization | `smart-pool-dashboard/frontend-vanilla/` (served directly by Flask, port 5000) and `smart-pool-dashboard/frontend/` (Next.js, `npm run dev`, port 3000) | renders the live dashboard — video feed + all 4 modules' state |
| API / WebSocket endpoints | `smart-pool-dashboard/backend/app.py` | HTTP: `GET /`, `GET /api/status`. Socket.IO events: `connect`, `start_detection`, `stop_detection`, `frame_update`, `state_update`, `drowning_alert`, `camera_error` |

---

## 4. Shared vs Individual work

- **`smart-pool-dashboard/`** is the **shared integration layer**, built collaboratively by the group. It contains the Flask-SocketIO backend (camera loop, decision engine, system state, settings) and both frontends. It does not contain any AI model logic of its own — it imports each member's package from `smart-pool-dashboard/components/` and orchestrates them.
- **Each `componentN-*/` folder at the project root is that member's individual contribution** — a fully standalone repo with its own importable package, trained model(s), training/evaluation code, dataset, `requirements.txt`, and `standalone_demo.py`. Each one runs and can be examined completely on its own, with no dependency on the dashboard.
- The copies inside `smart-pool-dashboard/components/` are mirrors of those same individual repos (originally wired up as git submodules — see `smart-pool-dashboard/README.md`), not separate work.

---

## 5. How to run

### The integrated dashboard (all 4 components + shared logic)
```bash
cd smart-pool-dashboard
pip install -r requirements.txt
cd backend
python app.py          # simulation mode by default; http://localhost:5000
```
Optional Next.js frontend instead of the built-in one:
```bash
cd smart-pool-dashboard/frontend
npm install
npm run dev             # http://localhost:3000
```

### Each component standalone (no dashboard needed)
```bash
cd component1-crowd-maintenance
pip install -r requirements.txt
python standalone_demo.py
```
```bash
cd component2-water-quality
pip install -r requirements.txt
python standalone_demo.py
```
```bash
cd component3-drowning-detection
pip install -r requirements.txt
# place trained weights at models/best.pt first if not already present
python standalone_demo.py
```
```bash
cd component4-garbage-detection
pip install -r requirements.txt
python standalone_demo.py
```
