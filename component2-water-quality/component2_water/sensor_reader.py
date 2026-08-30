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

CHLORINE_PLACEHOLDER_PPM = 1.8  # mid-range "ideal" value -- see module docstring


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
        # Simulator baseline: plausible pool values (only used if
        # simulate is explicitly requested — never as an auto fallback)
        self._sim = {"ph": 7.4, "temperature": 27.5, "chlorine": 1.8,
                     "turbidity": 1.0, "tds": 380.0}

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
            return False
        try:
            import serial
            import time

            self.source = source
            self.serial = serial.Serial(source, self.baud_rate, timeout=2)
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
        if not self.serial or not self.connected:
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
