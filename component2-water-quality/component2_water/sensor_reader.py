"""
Component 2 — sensor input layer.

Real-serial parsing matches the calibrated Arduino sketch in
arduino/pool_water_quality_sensors/pool_water_quality_sensors.ino, which
prints one labelled line per sensor per cycle (not a single CSV line):
    ---------------------------------
    Temp: 30.4 C
    pH: 4.11
    Turbidity: 0.0 NTU  ->  GOOD (safe/clear)
    TDS: 5 ppm  ->  GOOD (excellent/good)
read() accumulates lines (each "---" separator starts a fresh block) until
one line for each of the 4 sensors has been seen, then returns a reading.
The sketch always prints exactly one line per sensor per cycle even on
error ("Temp: ERROR ...", "pH: not calibrated"), so "one line seen" and
"a valid number was on it" are tracked separately: if a sensor's line
didn't parse as a number, ONLY that sensor's value is replaced with
SAFE_DEFAULTS[sensor] for this reading — the other 3 sensors' real
readings are still returned normally, not withheld. Which fields (if any)
were substituted this way is reported back in "fallback_fields", so a
transient glitch on one sensor never blanks the whole card and is never
silently presented as a live reading. A full Arduino cycle here takes
several seconds (each sensor averages 100 samples), so read() blocks up
to READ_BLOCK_TIMEOUT_S waiting for a complete cycle, returning None only
if no cycle's worth of lines arrives at all in that time (real
disconnect) — not merely because one sensor keeps erroring.

Freshly-powered sensors (pH especially) drift for the first ~20s after
the serial connection opens, so read() discards readings collected
before WARMUP_SECONDS has elapsed since connecting — see `warming_up`.

No chlorine sensor is wired on the physical rig, but the trained model
(component2-water-quality/models/) expects 5 features including Chlorine
(see predictor.py). CHLORINE_PLACEHOLDER_PPM below stands in for a real
reading until either a chlorine sensor is added to the Arduino sketch, or
the model is retrained on 4 features. This is a placeholder, not a fix.

Added around the original parsing (architecture only):
  - the serial port comes from settings instead of a hardcoded path
  - source="auto" (the default) probes for a plugged-in Arduino
    (/dev/cu.usbmodem*, /dev/cu.usbserial*, /dev/ttyACM*, /dev/ttyUSB*,
    COM ports) and uses it if found. If nothing is found, the reader
    stays disconnected — it does NOT fall back to fake data. Callers see
    connected=False and read() returns None until a real device shows
    up; try_connect() can be called again later (e.g. on a timer) to
    pick up a device plugged in after startup, with no restart needed.
  - simulate mode (source="simulate") generates realistic values
    (including chlorine) but must be requested explicitly — it is never
    an automatic fallback, so a "connected" reading is always real
    hardware. self.simulated tells callers which mode is active so they
    never display simulated data as if it were a live sensor reading.
  - sensor failure is reported instead of raising
"""

import glob
import random
import re
import time

CHLORINE_PLACEHOLDER_PPM = 1.8  # mid-range "ideal" value -- see module docstring
WARMUP_SECONDS = 20        # discard readings until this long after connecting
READ_BLOCK_TIMEOUT_S = 10  # give one full Arduino cycle (~5-6s) headroom

# Used ONLY to fill in a single sensor whose line errored/wasn't
# calibrated this cycle ("Temp: ERROR ...", "pH: not calibrated") — never
# as a substitute for a whole reading. Each is a safe mid-range pool
# value, chosen so a one-sensor glitch can't itself push the model's
# classification toward WARNING/CRITICAL.
SAFE_DEFAULTS = {
    "temperature": 27.0,
    "ph": 7.4,
    "turbidity": 0.5,
    "tds": 350.0,
}

# One (prefix, regex) pair per sensor line the sketch prints. The prefix
# alone identifies "a line for this sensor was seen" (even on error); the
# regex additionally captures the leading number when there is one, and
# ignores whatever unit/verdict text follows ("Temp: 30.4 C", "TDS: 5 ppm
# -> GOOD (excellent/good)", ...).
_LINE_PATTERNS = {
    "temperature": ("Temp:", re.compile(r"^Temp:\s*(-?\d+\.?\d*)")),
    "ph": ("pH:", re.compile(r"^pH:\s*(-?\d+\.?\d*)")),
    "turbidity": ("Turbidity:", re.compile(r"^Turbidity:\s*(-?\d+\.?\d*)")),
    "tds": ("TDS:", re.compile(r"^TDS:\s*(-?\d+\.?\d*)")),
}


def autodetect_port():
    """Return the path of a plugged-in Arduino-like USB-serial device,
    or None if nothing is found. Checked in a fixed order so the result
    is stable across calls when multiple devices are present.

    macOS/Linux: matched by the device path pattern USB-serial adapters
    register under (cu.usbmodem/cu.usbserial, ttyACM/ttyUSB) — these
    patterns never match built-in virtual ports (Bluetooth, debug
    console), so a match is a reliable signal of real USB hardware.

    Windows: device paths (COM3, COM4, ...) carry no such signal, so
    instead this asks pyserial for each port's description/manufacturer
    and only accepts one that looks like a USB-serial adapter — an
    unfiltered port list would also include non-USB COM ports.
    """
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*",  # macOS
                    "/dev/ttyACM*", "/dev/ttyUSB*"):             # Linux
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    try:
        from serial.tools import list_ports

        candidates = sorted(
            p.device for p in list_ports.comports()
            if p.device.upper().startswith("COM")
            and (p.vid is not None or "usb" in (p.description or "").lower())
        )
        return candidates[0] if candidates else None
    except Exception:  # noqa: BLE001
        return None


class SensorReader:
    def __init__(self, source="auto", baud_rate=9600):
        self.requested_source = source
        self.auto = str(source).lower() == "auto"
        self.simulated = str(source).lower() == "simulate"
        self.source = "simulate" if self.simulated else (None if self.auto else source)
        self.baud_rate = baud_rate
        self.serial = None
        self.connected = False
        self.connected_at = None
        # Simulator baseline: plausible pool values (only used if
        # simulate is explicitly requested — never as an auto fallback)
        self._sim = {"ph": 7.4, "temperature": 27.5, "chlorine": 1.8,
                     "turbidity": 1.0, "tds": 380.0}

    @property
    def warming_up(self):
        """True for WARMUP_SECONDS after a real connection is made —
        freshly-powered sensors (pH especially) drift during this
        window, so read() discards readings until it passes. Always
        False in simulate mode (nothing to warm up)."""
        if self.simulated or not self.connected or self.connected_at is None:
            return False
        return (time.time() - self.connected_at) < WARMUP_SECONDS

    def open(self):
        """Initial connection attempt. See try_connect() for retrying
        later without tearing down/recreating the reader."""
        if self.simulated:
            self.connected = True
            return True
        if self.auto:
            self.source = autodetect_port()
        return self._connect_to(self.source)

    def try_connect(self):
        """Call periodically while disconnected: re-probes for a
        newly-plugged-in device (auto mode) or retries the configured
        port. No-op if already connected or running in simulate mode."""
        if self.simulated or self.connected:
            return self.connected
        source = autodetect_port() if self.auto else self.source
        return self._connect_to(source)

    def _connect_to(self, source):
        if not source:
            self.connected = False
            self.connected_at = None
            return False
        try:
            import serial

            self.source = source
            self.serial = serial.Serial(source, self.baud_rate, timeout=2)
            time.sleep(3)
            self.serial.reset_input_buffer()  # preserved from original
            self.connected = True
            self.connected_at = time.time()
            return True
        except Exception:  # noqa: BLE001
            self.connected = False
            self.connected_at = None
            return False

    def read(self):
        """Return dict of 5 readings plus "fallback_fields" (sensor keys
        substituted from SAFE_DEFAULTS this cycle, if any), or None if
        unavailable (no complete cycle seen yet, still warming up, or
        disconnected). See module docstring for the line format and the
        per-sensor fallback behaviour."""
        if self.simulated:
            return self._simulate()
        if not self.serial or not self.connected:
            return None
        try:
            block = {}
            seen = set()
            deadline = time.time() + READ_BLOCK_TIMEOUT_S
            while time.time() < deadline:
                line = self.serial.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("---"):
                    block = {}  # separator marks the start of a fresh cycle
                    seen = set()
                    continue
                for key, (prefix, pattern) in _LINE_PATTERNS.items():
                    if not line.startswith(prefix):
                        continue
                    seen.add(key)  # a line for this sensor arrived, valid or not
                    match = pattern.match(line)
                    if match:
                        block[key] = float(match.group(1))
                    break

                if seen == _LINE_PATTERNS.keys():
                    if self.warming_up:
                        return None  # sensors still settling — discard
                    fallback_fields = [k for k in _LINE_PATTERNS if k not in block]
                    for key in fallback_fields:
                        block[key] = SAFE_DEFAULTS[key]
                    block["chlorine"] = CHLORINE_PLACEHOLDER_PPM
                    block["fallback_fields"] = fallback_fields
                    return block
            return None  # no complete cycle observed within the timeout
        except (ValueError, OSError):
            self.connected = False
            self.connected_at = None
            return None

    def _simulate(self):
        """Random walk around realistic pool chemistry values."""
        s = self._sim
        s["ph"] = min(8.6, max(6.4, s["ph"] + random.uniform(-0.05, 0.05)))
        s["temperature"] = min(33, max(24, s["temperature"] + random.uniform(-0.2, 0.2)))
        s["chlorine"] = min(3.5, max(0.2, s["chlorine"] + random.uniform(-0.06, 0.05)))
        s["turbidity"] = min(6.0, max(0.2, s["turbidity"] + random.uniform(-0.15, 0.16)))
        s["tds"] = min(520, max(280, s["tds"] + random.uniform(-4, 4)))
        return {**{k: round(v, 2) for k, v in s.items()}, "fallback_fields": []}
