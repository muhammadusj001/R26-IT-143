"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = "Smart Swimming Pool Monitoring API"
    project_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/pool_monitoring",
        validation_alias="DATABASE_URL",
    )
    yolo_model_path: str = "component1_crowd_maintenance/models/best_swimmer_model.pt"
    webcam_default_index: int = 0
    websocket_heartbeat_seconds: int = 30
    pool_capacity: int = 50
    density_low_threshold: int = 10
    density_medium_threshold: int = 25
    density_high_threshold: int = 40
    maintenance_chlorine_dose_threshold: float = 20.0
    maintenance_filter_backwash_threshold: float = 50.0
    maintenance_skimmer_clean_threshold: float = 30.0
    maintenance_shock_treatment_threshold: float = 100.0
    maintenance_deep_clean_threshold: float = 200.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
