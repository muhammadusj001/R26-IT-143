"""
Component 2 — AI prediction layer.

Model inference preserved UNCHANGED from the original scripts/app.py:
  DataFrame with columns [pH, Temperature, Chlorine, Turbidity, TDS]
  -> StandardScaler.transform -> XGBClassifier.predict
  -> LabelEncoder.inverse_transform -> SAFE / WARNING / CRITICAL

Only the file paths were centralised (they were relative "../models/.."
paths that broke when run from another directory).
"""

import pandas as pd

FEATURE_COLUMNS = ["pH", "Temperature", "Chlorine", "Turbidity", "TDS"]


class WaterQualityPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_status = "not_loaded"

    def load(self):
        try:
            import joblib

            d = self.model_dir
            self.model = joblib.load(d / "water_quality_model.pkl")
            self.scaler = joblib.load(d / "scaler.pkl")
            self.label_encoder = joblib.load(d / "label_encoder.pkl")
            self.model_status = "loaded"
            return True
        except Exception as exc:  # noqa: BLE001
            self.model_status = f"error: {exc}"
            return False

    def predict(self, reading: dict) -> str:
        """reading keys: ph, temperature, chlorine, turbidity, tds."""
        if self.model is None:
            return "UNKNOWN"
        # ── Original Component 2 inference (unchanged) ────────
        sample = pd.DataFrame(
            [[reading["ph"], reading["temperature"], reading["chlorine"],
              reading["turbidity"], reading["tds"]]],
            columns=FEATURE_COLUMNS,
        )
        sample_scaled = self.scaler.transform(sample)
        prediction = self.model.predict(sample_scaled)
        return str(self.label_encoder.inverse_transform(prediction)[0])
