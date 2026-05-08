"""
Maintenance Scheduler
SLIIT FYP - Component 1
Core Novelty: AI-driven maintenance scheduling
"""

from datetime import datetime
import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\muham\Desktop\Final_Year_Project\component1_crowd_maintenance")

class MaintenanceScheduler:
    
    # WHO/PHTA Standard thresholds (person-hours)
    THRESHOLDS = {
        'chlorine_dose':   20,
        'filter_backwash': 50,
        'skimmer_clean':   30,
        'shock_treatment': 100,
        'deep_clean':      200,
    }
    
    def __init__(self):
        self.maintenance_log = []
        self.last_maintenance = {k: 0 for k in self.THRESHOLDS}
        self.total_load = 0
    
    def update_load(self, bather_load_hours):
        self.total_load = bather_load_hours
    
    def get_recommendations(self):
        recommendations = []
        for action, threshold in self.THRESHOLDS.items():
            load_since_last = self.total_load - self.last_maintenance[action]
            if load_since_last >= threshold:
                ratio = load_since_last / threshold
                if ratio >= 2.0:
                    priority = "CRITICAL 🔴"
                elif ratio >= 1.5:
                    priority = "HIGH 🟠"
                else:
                    priority = "MEDIUM 🟡"
                    
                recommendations.append({
                    'action': action.replace('_', ' ').title(),
                    'priority': priority,
                    'overdue_by': round(load_since_last - threshold, 2)
                })
        
        return sorted(recommendations, key=lambda x: x['overdue_by'], reverse=True)
    
    def mark_completed(self, action):
        self.last_maintenance[action] = self.total_load
        self.maintenance_log.append({
            'action': action,
            'completed_at': datetime.now().isoformat(),
            'load_at_completion': self.total_load
        })
    
    def print_report(self):
        recommendations = self.get_recommendations()
        
        print("\n" + "="*50)
        print("🏊 POOL MAINTENANCE SCHEDULE REPORT")
        print("="*50)
        print(f"Current Bather Load: {self.total_load} person-hours")
        print(f"Pending Actions:     {len(recommendations)}")
        print("="*50)
        
        if not recommendations:
            print("✅ No maintenance needed!")
        else:
            print("\n📋 RECOMMENDED ACTIONS:")
            print("-"*50)
            for rec in recommendations:
                print(f"  Action:   {rec['action']}")
                print(f"  Priority: {rec['priority']}")
                print(f"  Overdue:  {rec['overdue_by']} person-hours")
                print("-"*50)
        
        # Save report
        output = BASE_DIR / "results" / "maintenance_schedule.json"
        output.parent.mkdir(exist_ok=True)
        report = {
            'timestamp': datetime.now().isoformat(),
            'bather_load': self.total_load,
            'recommendations': recommendations
        }
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Report saved!")


# Test
if __name__ == "__main__":
    scheduler = MaintenanceScheduler()
    
    # Test with different load levels
    test_loads = [10, 25, 55, 105, 210]
    
    for load in test_loads:
        print(f"\nTesting load: {load} person-hours")
        scheduler.update_load(load)
        scheduler.print_report()
    
    print("\n✅ Scheduler working!")