"""
Dashboard-side "final reading" test.

Collects live water-quality samples over a fixed window and reports one
median, ML-classified result with a human-readable explanation, instead
of leaving the user to eyeball whichever noisy instantaneous value happens
to be on screen. Median (not mean) so one noisy/glitched sample — a single
bad ADC read, a momentary probe disturbance — can't skew the reported
value the way an outlier pulls an average. Samples from the first `warmup`
seconds are discarded (the sensor may still be settling right after the
test starts); everything from `warmup` up to `duration` seconds is folded
into the final reading.

Fed by WaterQualityMonitor's existing read loop (see water_monitor.py) --
this does not open its own connection to the sensor, since a real serial
port can only be read reliably by one thread at a time.
"""

import statistics
import time

from component2_water.explain import explain_status

FIELDS = ("ph", "temperature", "chlorine", "turbidity", "tds")
# chlorine has no real sensor at all (see sensor_reader.py) -- it's always
# the same documented placeholder, so it's excluded from the genuine-vs-
# fallback split below; the other 4 fields have a real sensor that can
# individually error on a given cycle (see SensorReader.read()'s
# per-sensor fallback), and this test should only median real readings
# for each, not readings diluted by that cycle's safe-default filler.
FALLBACK_ELIGIBLE_FIELDS = ("ph", "temperature", "turbidity", "tds")


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
            median = {}
            fallback_only = []
            for field in FIELDS:
                if field in FALLBACK_ELIGIBLE_FIELDS:
                    # Only samples where THIS field was a genuine sensor
                    # read this cycle -- excludes cycles where it was
                    # filled in from SAFE_DEFAULTS because that one
                    # sensor errored (other fields from the same cycle
                    # are unaffected and still used normally).
                    genuine = [s[field] for s in self.samples
                               if field not in (s.get("fallback_fields") or [])]
                else:
                    genuine = [s[field] for s in self.samples]
                values = genuine or [s[field] for s in self.samples]
                median[field] = round(statistics.median(values), 2)
                if field in FALLBACK_ELIGIBLE_FIELDS and not genuine:
                    fallback_only.append(field)

            status = self.predictor.predict(median)
            reasons = explain_status(median, status)
            if fallback_only:
                reasons.append(
                    f"{', '.join(fallback_only)}: this sensor never produced a "
                    "valid reading during the test — its value above is a safe "
                    "default, not a real measurement"
                )
        else:
            # No samples landed inside the warmup..duration window -- sensor
            # was disconnected/unavailable for the whole test.
            median = {field: None for field in FIELDS}
            status = "UNKNOWN"
            reasons = ["No sensor readings were available during the test window."]

        self.result = {
            **median,
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
