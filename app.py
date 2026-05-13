import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "Swimming-pool-garbage-detection" / "app.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Cannot find application entry point: {TARGET}")

os.chdir(TARGET.parent)
runpy.run_path(str(TARGET), run_name="__main__")
