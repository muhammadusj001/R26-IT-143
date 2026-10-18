"""Standalone viva demo for Component 2 (no dashboard needed).
Reads Arduino (or simulates) and prints AI water-quality predictions."""
import time
from pathlib import Path
from component2_water.sensor_reader import SensorReader
from component2_water.predictor import WaterQualityPredictor

SOURCE = "auto"  # probes for a plugged-in Arduino, falls back to simulate;
                 # or force one explicitly: "COM3" / "/dev/cu.usbmodem21401" / "simulate"

reader = SensorReader(SOURCE)
predictor = WaterQualityPredictor(Path(__file__).parent / "models")
reader.open()
predictor.load()
print(f"Sensor: {'simulated (no Arduino found)' if reader.simulated else reader.source} | Model: {predictor.model_status}")
while True:
    reading = reader.read()
    if reading:
        status = predictor.predict(reading)
        print(f"pH={reading['ph']:.2f} T={reading['temperature']:.1f}C "
              f"Cl={reading['chlorine']:.2f} NTU={reading['turbidity']:.2f} "
              f"TDS={reading['tds']:.0f}  ->  {status}")
    time.sleep(2)
