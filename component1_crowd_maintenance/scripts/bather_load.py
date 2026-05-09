"""
Bather Load Calculator
SLIIT FYP - Component 1
Crowd-Aware Maintenance Scheduling Module
"""

from datetime import datetime
import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\muham\Desktop\Final_Year_Project\component1_crowd_maintenance")

class BatherLoadCalculator:
    
    def __init__(self):
        self.records = []
        self.hourly_loads = {}
        
    def add_reading(self, swimmer_count, interval_seconds=5):
        timestamp = datetime.now()
        person_seconds = swimmer_count * interval_seconds
        
        record = {
            'timestamp': timestamp.isoformat(),
            'swimmer_count': swimmer_count,
            'person_seconds': person_seconds,
            'hour': timestamp.strftime("%Y-%m-%d %H:00")
        }
        
        self.records.append(record)
        
        hour_key = record['hour']
        if hour_key not in self.hourly_loads:
            self.hourly_loads[hour_key] = 0
        self.hourly_loads[hour_key] += person_seconds
        
        return record
    
    def get_current_load(self):
        today = datetime.now().strftime("%Y-%m-%d")
        total = sum(
            r['person_seconds'] 
            for r in self.records 
            if r['timestamp'].startswith(today)
        )
        return total / 3600  # Convert to person-hours
    
    def get_peak_hour(self):
        if not self.hourly_loads:
            return None
        return max(self.hourly_loads, key=self.hourly_loads.get)
    
    def get_summary(self):
        return {
            'total_readings': len(self.records),
            'current_bather_load_hours': round(self.get_current_load(), 2),
            'peak_hour': self.get_peak_hour()
        }
    
    def save_to_file(self):
        output = BASE_DIR / "results" / "bather_load_data.json"
        output.parent.mkdir(exist_ok=True)
        with open(output, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"✅ Saved to: {output}")


# Test
if __name__ == "__main__":
    import random
    
    print("Testing Bather Load Calculator...")
    calc = BatherLoadCalculator()
    
    # Simulate 1 hour of readings
    for i in range(720):
        count = random.randint(0, 25)
        calc.add_reading(count, interval_seconds=5)
    
    summary = calc.get_summary()
    
    print("\n" + "="*40)
    print("BATHER LOAD SUMMARY")
    print("="*40)
    print(f"Total readings:  {summary['total_readings']}")
    print(f"Bather load:     {summary['current_bather_load_hours']} person-hours")
    print(f"Peak hour:       {summary['peak_hour']}")
    print("="*40)
    
    calc.save_to_file()
    print("✅ Bather load calculator working!")