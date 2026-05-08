"""Schemas for crowd analytics and occupancy intelligence."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CrowdAnalyticsRequest(BaseModel):
    swimmer_count: int = Field(ge=0)
    interval_seconds: int = Field(default=5, ge=1)


class MaintenanceRecommendation(BaseModel):
    action: str
    priority: str
    overdue_by: float


class CrowdAnalyticsResponse(BaseModel):
    swimmer_count: int
    occupancy_capacity: int
    occupancy_percentage: float
    crowd_density_level: str
    crowd_density_label: str
    bather_load_person_hours: float
    real_time_status: str
    maintenance_urgency: str
    maintenance_recommendations: list[MaintenanceRecommendation]
"""Analytics response schemas placeholder."""
