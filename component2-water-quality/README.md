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
