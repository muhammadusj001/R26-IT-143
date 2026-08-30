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
from core.water_test import WaterQualityTest
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
        self.test = None

    def start(self):
        self.predictor.load()
        self.reader.open()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def start_test(self):
        """Kick off a 60s "final reading" test -- fed by this same loop's
        readings (see _tick_test), not a separate sensor connection."""
        self.test = WaterQualityTest(self.predictor)
        state.update_module("water_quality", {"test": self.test.progress()})

    def _tick_test(self, reading):
        if self.test is None:
            return
        if not self.test.finished:
            if reading is not None:
                self.test.collect(reading)
            if self.test.expired:
                self.test.finalize()
                result = self.test.result
                if result["status"] in ("WARNING", "CRITICAL"):
                    severity = "danger" if result["status"] == "CRITICAL" else "warning"
                    message = f"Water quality test: {result['status']} — {result['reasons'][0]}"
                    state.add_alert("water_quality", message, severity)
                    if self.session_tracker:
                        self.session_tracker.record_alert("water_quality", message, severity)
        state.update_module("water_quality", {"test": self.test.progress()})

    def _loop(self):
        while self._running:
            if not self.reader.connected and not self.reader.simulated:
                # Real-hardware mode with nothing connected yet (or a
                # dropped connection) — keep probing so plugging the
                # Arduino in mid-session picks it up with no restart.
                self.reader.try_connect()

            reading = self.reader.read()
            if reading is None:
                state.update_module("water_quality", {
                    # No real reading available — never show stale/last
                    # values as if they were current.
                    "ph": None, "temperature": None, "chlorine": None,
                    "turbidity": None, "tds": None, "status": "UNKNOWN",
                    "sensor_connected": self.reader.connected,
                    "simulated": self.reader.simulated,
                    "warming_up": self.reader.warming_up,
                    "model_status": self.predictor.model_status,
                })
                self._tick_test(None)
                time.sleep(settings.SENSOR_INTERVAL_SECONDS)
                continue

            status = self.predictor.predict(reading)
            state.update_module("water_quality", {
                **reading, "status": status, "sensor_connected": True,
                "simulated": self.reader.simulated,
                "warming_up": False,
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

            self._tick_test(reading)

            if self.on_update:
                self.on_update()
            time.sleep(settings.SENSOR_INTERVAL_SECONDS)
