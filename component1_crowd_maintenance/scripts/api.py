"""
FastAPI Endpoint - Component 1
SLIIT FYP - Crowd-Aware Maintenance Scheduling
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from bather_load import BatherLoadCalculator
from scheduler import MaintenanceScheduler

app = FastAPI(
    title="Pool Maintenance API - Component 1",
    description="Swimmer detection and maintenance scheduling",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

calculator = BatherLoadCalculator()
scheduler = MaintenanceScheduler()

class SwimmerReading(BaseModel):
    swimmer_count: int
    interval_seconds: int = 5

@app.get("/")
def root():
    return {
        "component": "1 - Crowd-Aware Maintenance Scheduling",
        "status": "running ✅",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detection/add-reading")
def add_reading(reading: SwimmerReading):
    record = calculator.add_reading(
        reading.swimmer_count,
        reading.interval_seconds
    )
    return {"success": True, "record": record}

@app.get("/detection/current-load")
def get_current_load():
    summary = calculator.get_summary()
    return {
        "bather_load_hours": summary['current_bather_load_hours'],
        "total_readings": summary['total_readings'],
        "peak_hour": summary['peak_hour']
    }

@app.get("/maintenance/recommendations")
def get_recommendations():
    load = calculator.get_current_load()
    scheduler.update_load(load)
    return {
        "current_load": round(load, 2),
        "recommendations": scheduler.get_recommendations(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/maintenance/report")
def get_report():
    load = calculator.get_current_load()
    scheduler.update_load(load)
    scheduler.print_report()
    return {"status": "Report generated", "load": round(load, 2)}

if __name__ == "__main__":
    print("Starting API...")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)