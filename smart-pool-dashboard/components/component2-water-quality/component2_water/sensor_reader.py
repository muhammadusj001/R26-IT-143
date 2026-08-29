"""
Component 2 — sensor input layer.

The Arduino serial-parsing logic is preserved UNCHANGED from the
original scripts/app.py:
  - reads a line, skips blanks and the "System Ready" banner
  - expects exactly 5 comma-separated values:
    pH, Temperature, Chlorine, Turbidity, TDS
  - skips malformed lines instead of crashing

Added around it (architecture only):
  - the serial port comes from settings instead of a hardcoded
    "/dev/cu.usbmodem21401"
  - a simulator mode generates realistic values when no Arduino is
    connected, so the dashboard card always works
  - sensor failure is reported instead of raising
"""

import random

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
            # ── Original Component 2 parsing logic (unchanged) ──
            line = self.serial.readline().decode(errors="ignore").strip()
            if line == "" or line == "System Ready":
                return None
            values = line.split(",")
            if len(values) != 5:
                return None
            return {
                "ph": float(values[0]),
                "temperature": float(values[1]),
                "chlorine": float(values[2]),
                "turbidity": float(values[3]),
                "tds": float(values[4]),
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
