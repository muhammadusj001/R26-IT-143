"""
Dashboard-side "final reading" test.

Collects live water-quality samples over a fixed window and reports one
averaged, ML-classified result with a human-readable explanation, instead
of leaving the user to eyeball whichever noisy instantaneous value happens
to be on screen. Samples from the first `warmup` seconds are discarded (the
sensor may still be settling right after the test starts); everything from
`warmup` up to `duration` seconds is averaged into the final reading.

Fed by WaterQualityMonitor's existing read loop (see water_monitor.py) --
this does not open its own connection to the sensor, since a real serial
port can only be read reliably by one thread at a time.
"""

import time

from component2_water.explain import explain_status

FIELDS = ("ph", "temperature", "chlorine", "turbidity", "tds")


class WaterQualityTest:
    def __init__(self, predictor, duration=60, warmup=20):
        self.predictor = predictor
        self.duration = duration
        self.warmup = warmup
        self.start_time = time.time()
        self.samples = []
        self.finished = False
        self.result = None

    @property
    def elapsed(self):
        return time.time() - self.start_time

    @property
    def expired(self):
        return self.elapsed >= self.duration

    def collect(self, reading: dict):
        if self.finished:
            return
        if self.elapsed >= self.warmup:
            self.samples.append(reading)

    def finalize(self):
        if self.finished:
            return
        self.finished = True

        if self.samples:
            averaged = {
                field: round(sum(s[field] for s in self.samples) / len(self.samples), 2)
                for field in FIELDS
            }
            status = self.predictor.predict(averaged)
            reasons = explain_status(averaged, status)
        else:
            # No samples landed inside the warmup..duration window -- sensor
            # was disconnected/unavailable for the whole test.
            averaged = {field: None for field in FIELDS}
            status = "UNKNOWN"
            reasons = ["No sensor readings were available during the test window."]

        self.result = {
            **averaged,
            "status": status,
            "reasons": reasons,
            "sample_count": len(self.samples),
        }

    def progress(self) -> dict:
        return {
            "status": "complete" if self.finished else "running",
            "elapsed": round(min(self.elapsed, self.duration), 1),
            "duration": self.duration,
            "warmup": self.warmup,
            "samples_collected": len(self.samples),
            "result": self.result,
        }
