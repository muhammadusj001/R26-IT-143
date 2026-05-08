"""Crowd occupancy intelligence service for swimmer analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CrowdAnalyticsThresholds:
    low: int
    medium: int
    high: int
    capacity: int
    chlorine_dose: float = 20.0
    filter_backwash: float = 50.0
    skimmer_clean: float = 30.0
    shock_treatment: float = 100.0
    deep_clean: float = 200.0


@dataclass
class CrowdAnalyticsService:
    thresholds: CrowdAnalyticsThresholds
    total_person_seconds: float = 0.0
    total_readings: int = 0
    last_swimmer_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def record_reading(self, swimmer_count: int, interval_seconds: int = 5) -> dict[str, Any]:
        swimmer_count = max(0, swimmer_count)
        interval_seconds = max(1, interval_seconds)

        person_seconds = swimmer_count * interval_seconds
        self.total_person_seconds += person_seconds
        self.total_readings += 1
        self.last_swimmer_count = swimmer_count

        status = self._build_status(swimmer_count=swimmer_count, interval_seconds=interval_seconds)
        self.history.append(status)
        return status

    def get_current_status(self) -> dict[str, Any]:
        return self._build_status(swimmer_count=self.last_swimmer_count, interval_seconds=5)

    def _build_status(self, swimmer_count: int, interval_seconds: int) -> dict[str, Any]:
        occupancy_percentage = self._calculate_occupancy_percentage(swimmer_count)
        density_level, density_label = self._calculate_density_level(swimmer_count)
        bather_load_person_hours = self._calculate_bather_load(swimmer_count, interval_seconds)
        maintenance_recommendations = self._calculate_maintenance_recommendations(bather_load_person_hours)

        return {
            "swimmer_count": swimmer_count,
            "occupancy_capacity": self.thresholds.capacity,
            "occupancy_percentage": round(occupancy_percentage, 2),
            "crowd_density_level": density_level,
            "crowd_density_label": density_label,
            "bather_load_person_hours": round(bather_load_person_hours, 4),
            "real_time_status": self._calculate_real_time_status(swimmer_count),
            "maintenance_urgency": self._calculate_maintenance_urgency(bather_load_person_hours),
            "maintenance_recommendations": maintenance_recommendations,
            "summary": {
                "total_readings": self.total_readings,
                "total_person_hours": round(self.total_person_seconds / 3600.0, 4),
            },
        }

    def _calculate_occupancy_percentage(self, swimmer_count: int) -> float:
        if self.thresholds.capacity <= 0:
            return 0.0
        return (swimmer_count / self.thresholds.capacity) * 100.0

    def _calculate_density_level(self, swimmer_count: int) -> tuple[str, str]:
        if swimmer_count < self.thresholds.low:
            return "LOW", "Light occupancy"
        if swimmer_count < self.thresholds.medium:
            return "MEDIUM", "Moderate occupancy"
        if swimmer_count < self.thresholds.high:
            return "HIGH", "Busy occupancy"
        return "CRITICAL", "Overloaded occupancy"

    def _calculate_bather_load(self, swimmer_count: int, interval_seconds: int) -> float:
        return swimmer_count * interval_seconds / 3600.0

    def _calculate_real_time_status(self, swimmer_count: int) -> str:
        if swimmer_count == 0:
            return "EMPTY"
        if swimmer_count < self.thresholds.low:
            return "NORMAL"
        if swimmer_count < self.thresholds.medium:
            return "WATCH"
        if swimmer_count < self.thresholds.high:
            return "BUSY"
        return "OVERCAPACITY"

    def _calculate_maintenance_urgency(self, bather_load_person_hours: float) -> str:
        if bather_load_person_hours >= self.thresholds.deep_clean:
            return "CRITICAL"
        if bather_load_person_hours >= self.thresholds.shock_treatment:
            return "HIGH"
        if bather_load_person_hours >= self.thresholds.filter_backwash:
            return "MEDIUM"
        return "LOW"

    def _calculate_maintenance_recommendations(self, bather_load_person_hours: float) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []

        maintenance_matrix = [
            ("Chlorine Dose", self.thresholds.chlorine_dose),
            ("Skimmer Clean", self.thresholds.skimmer_clean),
            ("Filter Backwash", self.thresholds.filter_backwash),
            ("Shock Treatment", self.thresholds.shock_treatment),
            ("Deep Clean", self.thresholds.deep_clean),
        ]

        for action, threshold in maintenance_matrix:
            overdue_by = bather_load_person_hours - threshold
            if overdue_by >= 0:
                recommendations.append(
                    {
                        "action": action,
                        "priority": self._priority_for_threshold(overdue_by, threshold),
                        "overdue_by": round(overdue_by, 4),
                    }
                )

        return sorted(recommendations, key=lambda item: item["overdue_by"], reverse=True)

    def _priority_for_threshold(self, overdue_by: float, threshold: float) -> str:
        if threshold <= 0:
            return "LOW"
        ratio = overdue_by / threshold
        if ratio >= 1.0:
            return "CRITICAL"
        if ratio >= 0.5:
            return "HIGH"
        return "MEDIUM"
