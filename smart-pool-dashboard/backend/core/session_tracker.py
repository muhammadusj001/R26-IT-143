"""
Session data tracker — first step toward the PDF session report feature.

Purely observational: it records a bounded history of what the vision
loop and water monitor already compute (crowd samples, water samples,
alerts, garbage/drowning event counts) so a summary — and later a PDF —
can be built from it. It does not touch detection logic, thresholds, or
the decision engine; it only reads results those already produce.

Crowd/water samples are throttled to at most one every SAMPLE_INTERVAL
seconds so a long session doesn't need an unbounded amount of memory;
combined with the bounded deques (maxlen), memory stays capped regardless
of session length.
"""

import time
from collections import deque
from datetime import datetime

MAX_SAMPLES = 2000
SAMPLE_INTERVAL_SECONDS = 5  # throttle: at most one crowd/water sample per this many seconds

WATER_FIELDS = ("ph", "temperature", "chlorine", "turbidity", "tds")
_DENSITY_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "OVER CAPACITY": 3}


class SessionTracker:
    def __init__(self, maxlen=MAX_SAMPLES, sample_interval=SAMPLE_INTERVAL_SECONDS):
        self.sample_interval = sample_interval
        self._maxlen = maxlen
        self.start_time = None
        self.end_time = None

        # (timestamp, swimmer_count, density_level, bather_load)
        self.crowd_samples = deque(maxlen=maxlen)
        # (timestamp, ph, temperature, chlorine, turbidity, tds, status)
        self.water_samples = deque(maxlen=maxlen)
        # {module, message, severity, time} — same shape as state.add_alert
        self.alerts = deque(maxlen=maxlen)

        self.garbage_events = 0
        self.drowning_alerts = 0

        self._latest_recommendations = []
        self._last_crowd_sample_ts = 0.0
        self._last_water_sample_ts = 0.0

    # ── Lifecycle ────────────────────────────────────────────
    def start(self):
        """Begin a new session, resetting all recorded data."""
        self.start_time = time.time()
        self.end_time = None
        self.crowd_samples.clear()
        self.water_samples.clear()
        self.alerts.clear()
        self.garbage_events = 0
        self.drowning_alerts = 0
        self._latest_recommendations = []
        self._last_crowd_sample_ts = 0.0
        self._last_water_sample_ts = 0.0

    def stop(self):
        """Mark the session as ended. Recorded data is kept for get_summary()."""
        self.end_time = time.time()

    # ── Recording ────────────────────────────────────────────
    def record_crowd(self, swimmer_count, density_level, bather_load, recommendations=None):
        """Feed one crowd result. Throttled to one sample per sample_interval
        seconds, but the current maintenance recommendations are always
        kept up to date regardless of throttling."""
        if recommendations is not None:
            self._latest_recommendations = recommendations

        now = time.time()
        if now - self._last_crowd_sample_ts < self.sample_interval:
            return
        self._last_crowd_sample_ts = now
        self.crowd_samples.append((now, swimmer_count, density_level, bather_load))

    def record_water(self, ph, temperature, chlorine, turbidity, tds, status):
        """Feed one water reading. Throttled to one sample per sample_interval seconds."""
        now = time.time()
        if now - self._last_water_sample_ts < self.sample_interval:
            return
        self._last_water_sample_ts = now
        self.water_samples.append((now, ph, temperature, chlorine, turbidity, tds, status))

    def record_alert(self, module, message, severity="warning", time_str=None):
        """Record an alert. Not throttled — alerts are already infrequent
        and each one matters for the report."""
        self.alerts.append({
            "module": module,
            "message": message,
            "severity": severity,
            "time": time_str or datetime.now().strftime("%H:%M:%S"),
        })
        if module == "drowning":
            self.drowning_alerts += 1
        elif module == "garbage":
            self.garbage_events += 1

    # ── Summary ──────────────────────────────────────────────
    def get_summary(self) -> dict:
        now = time.time()
        duration_seconds = (
            round((self.end_time or now) - self.start_time, 1) if self.start_time else 0.0
        )

        occupancies = [c[1] for c in self.crowd_samples if c[1] is not None]
        bather_loads = [c[3] for c in self.crowd_samples if c[3] is not None]

        water_stats = {}
        for idx, field in enumerate(WATER_FIELDS, start=1):
            values = [w[idx] for w in self.water_samples if w[idx] is not None]
            water_stats[field] = {
                "min": round(min(values), 2) if values else None,
                "max": round(max(values), 2) if values else None,
                "avg": round(sum(values) / len(values), 2) if values else None,
            }

        alert_counts_by_module = {}
        alert_counts_by_severity = {}
        for a in self.alerts:
            alert_counts_by_module[a["module"]] = alert_counts_by_module.get(a["module"], 0) + 1
            alert_counts_by_severity[a["severity"]] = alert_counts_by_severity.get(a["severity"], 0) + 1

        densities = [c[2] for c in self.crowd_samples if c[2] is not None]
        peak_density_level = (
            max(densities, key=lambda d: _DENSITY_RANK.get(d, -1)) if densities else None
        )

        # Approximate — derived from periodic (throttled) sampling, not a
        # continuous measurement: each recorded water sample is assumed to
        # represent one sample_interval of real time in that status.
        water_status_seconds = {}
        for w in self.water_samples:
            wstatus = w[6]
            if wstatus:
                water_status_seconds[wstatus] = water_status_seconds.get(wstatus, 0) + self.sample_interval

        return {
            "start_time": _fmt(self.start_time),
            "end_time": _fmt(self.end_time),
            "duration_seconds": duration_seconds,
            "duration_formatted": f"{int(duration_seconds // 60):02d}:{int(duration_seconds % 60):02d}",
            "peak_occupancy": max(occupancies) if occupancies else 0,
            "average_occupancy": round(sum(occupancies) / len(occupancies), 2) if occupancies else 0,
            "total_bather_hours": round(bather_loads[-1], 2) if bather_loads else 0.0,
            "peak_density_level": peak_density_level,
            "water_quality": water_stats,
            "water_status_seconds": water_status_seconds,
            "total_alerts": len(self.alerts),
            "alerts": list(self.alerts),
            "alert_counts_by_module": alert_counts_by_module,
            "alert_counts_by_severity": alert_counts_by_severity,
            "garbage_events": self.garbage_events,
            "drowning_alerts": self.drowning_alerts,
            "maintenance_recommendations": self._latest_recommendations,
            "crowd_sample_count": len(self.crowd_samples),
            "water_sample_count": len(self.water_samples),
        }


def _fmt(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else None
