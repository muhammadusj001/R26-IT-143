# Component 2 — AI Water Quality Prediction
**Owner:** TODO — add member name
**AI Task:** Classifies pool-water sensor readings (pH, Temperature, Chlorine, Turbidity, TDS) into SAFE / WARNING / CRITICAL.

## Where everything lives
| What | File | Purpose |
|---|---|---|
| AI inference | `component2_water/predictor.py` | loads the XGBoost model, scaler, and label encoder; runs classification per reading |
| Business logic | `component2_water/predictor.py` (same file) | the SAFE/WARNING/CRITICAL decision is the classification itself — there is no separate rules module for this component |
| Utilities | `component2_water/sensor_reader.py` | Arduino serial parsing, plus a simulated-sensor fallback |
| Trained model | `models/water_quality_model.pkl` (824 KB), `models/scaler.pkl`, `models/label_encoder.pkl` | XGBClassifier + preprocessing artifacts |
| Training code | `ml/scripts/train_xgboost.py` | how the model was trained |
| Evaluation code | `ml/scripts/train_xgboost.py` (same script) | metrics are produced during training and saved to `models/training_metrics_xgboost.json` — no separate eval script exists |
| Dataset config | `ml/datasets/pool_water_quality_augmented_dataset.csv` | tabular dataset (not YOLO — there is no `data.yaml` for this component) |
| Standalone demo | `standalone_demo.py` | run this component alone |

## Model performance
| Metric | Value |
|---|---|
| Accuracy | 1.00 (per `models/training_metrics_xgboost.json`) |
| Macro F1 | 1.00 |
| Classes | SAFE, WARNING, CRITICAL |
| Caveat | Perfect score is on the synthetic augmented dataset — see "Honest evaluation note" below before citing this at the viva |

## How to run this component alone
```bash
pip install -r requirements.txt
python standalone_demo.py
```

---

# Component 2 — AI Water Quality Prediction (R26-IT-143)

Arduino sensors (pH, Temperature, Chlorine, Turbidity, TDS) → XGBoost
classifier → SAFE / WARNING / CRITICAL.

## Structure
- `component2_water/` — importable package (sensor_reader, predictor)
- `ml/` — training script, notebook, Arduino sketch, dataset
- `models/` — trained model + scaler + label encoder
- `standalone_demo.py` — run alone (simulated sensors or real serial port)

## Run standalone
```bash
pip install -r requirements.txt
python standalone_demo.py
```

## Honest evaluation note (viva)
Training accuracy = 1.00 on the synthetic augmented dataset, which means
the classes are trivially separable by thresholds. Be ready to discuss:
test on noisy/unseen ranges or real samples to demonstrate genuine
generalization, and compare against a plain threshold baseline.
