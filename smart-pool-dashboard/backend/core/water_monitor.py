"""
Dashboard-side monitor thread for Component 2.
Wires the component2_water package (sensor reader + predictor) into the
shared SystemState. Lives in the dashboard repo because it touches
integration concerns; the pure component logic stays in its own repo.
"""

import threading
import time

from config import settings
from core.state import state
from component2_water.sensor_reader import SensorReader
from component2_water.predictor import WaterQualityPredictor


class WaterQualityMonitor:
    def __init__(self, on_update=None, session_tracker=None):
        self.reader = SensorReader(settings.SENSOR_SOURCE, settings.SENSOR_BAUD_RATE)
        self.predictor = WaterQualityPredictor(settings.WATER_MODEL_DIR)
        self.on_update = on_update
        self.session_tracker = session_tracker
        self._running = False
        self._last_status = None

    def start(self):
        self.predictor.load()
        self.reader.open()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            reading = self.reader.read()
            if reading is None:
                state.update_module("water_quality", {
                    "sensor_connected": self.reader.connected,
                    "model_status": self.predictor.model_status,
                })
                time.sleep(settings.SENSOR_INTERVAL_SECONDS)
                continue

            status = self.predictor.predict(reading)
            state.update_module("water_quality", {
                **reading, "status": status, "sensor_connected": True,
                "model_status": self.predictor.model_status,
            })
            if self.session_tracker:
                self.session_tracker.record_water(
                    reading["ph"], reading["temperature"], reading["chlorine"],
                    reading["turbidity"], reading["tds"], status,
                )
            if status != self._last_status and status in ("WARNING", "CRITICAL"):
                water_severity = "danger" if status == "CRITICAL" else "warning"
                water_message = f"Water quality {status}"
                state.add_alert("water_quality", water_message, water_severity)
                if self.session_tracker:
                    self.session_tracker.record_alert("water_quality", water_message, water_severity)
            self._last_status = status

            if self.on_update:
                self.on_update()
            time.sleep(settings.SENSOR_INTERVAL_SECONDS)
