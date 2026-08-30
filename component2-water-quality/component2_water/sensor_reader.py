"""
Component 2 — sensor input layer.

Real-serial parsing matches the calibrated Arduino sketch in
arduino/pool_water_quality_sensors/pool_water_quality_sensors.ino:
  - reads a line, skips blanks and anything that isn't a "DATA:" line
    (banners, CALHELP text, "CALGOOD saved.", the separate "RAW:" line, etc.)
  - a "DATA:" line carries exactly 4 comma-separated values, in this order:
    pH, Temperature, Turbidity, TDS
  - skips malformed lines instead of crashing

No chlorine sensor is wired on the physical rig, but the trained model
(component2-water-quality/models/) expects 5 features including Chlorine
(see predictor.py). CHLORINE_PLACEHOLDER_PPM below stands in for a real
reading until either a chlorine sensor is added to the Arduino sketch, or
the model is retrained on 4 features. This is a placeholder, not a fix.

Added around the original parsing (architecture only):
  - the serial port comes from settings instead of a hardcoded path
  - a simulator mode generates realistic values (including chlorine) when
    no Arduino is connected, so the dashboard card always works
  - sensor failure is reported instead of raising
"""

import random

CHLORINE_PLACEHOLDER_PPM = 1.8  # mid-range "ideal" value -- see module docstring


class SensorReader:
    def __init__(self, source="simulate", baud_rate=9600):
        self.source = source
        self.baud_rate = baud_rate
        self.serial = None
        self.connected = False
        self.simulated = str(source).lower() == "simulate"
        # Simulator baseline: plausible pool values
        self._sim = {"ph": 7.4, "temperature": 27.5, "chlorine": 1.8,
                     "turbidity": 1.0, "tds": 380.0}

    def open(self):
        if self.simulated:
            self.connected = True
            return True
        try:
            import serial
            import time

            self.serial = serial.Serial(self.source, self.baud_rate, timeout=2)
            time.sleep(3)
            self.serial.reset_input_buffer()  # preserved from original
            self.connected = True
            return True
        except Exception:  # noqa: BLE001
            self.connected = False
            return False

    def read(self):
        """Return dict of 5 readings, or None if unavailable."""
        if self.simulated:
            return self._simulate()
        if not self.serial:
            return None
        try:
            line = self.serial.readline().decode(errors="ignore").strip()
            if not line.startswith("DATA:"):
                return None  # banner / CALHELP / RAW: / calibration ack line

            values = line[len("DATA:"):].split(",")
            if len(values) != 4:
                return None

            ph, temperature, turbidity, tds = (float(v) for v in values)
            return {
                "ph": ph,
                "temperature": temperature,
                "chlorine": CHLORINE_PLACEHOLDER_PPM,
                "turbidity": turbidity,
                "tds": tds,
            }
        except (ValueError, OSError):
            self.connected = False
            return None

    def _simulate(self):
        """Random walk around realistic pool chemistry values."""
        s = self._sim
        s["ph"] = min(8.6, max(6.4, s["ph"] + random.uniform(-0.05, 0.05)))
        s["temperature"] = min(33, max(24, s["temperature"] + random.uniform(-0.2, 0.2)))
        s["chlorine"] = min(3.5, max(0.2, s["chlorine"] + random.uniform(-0.06, 0.05)))
        s["turbidity"] = min(6.0, max(0.2, s["turbidity"] + random.uniform(-0.15, 0.16)))
        s["tds"] = min(520, max(280, s["tds"] + random.uniform(-4, 4)))
        return {k: round(v, 2) for k, v in s.items()}
