"""Aggregate all version 1 routers here."""

from fastapi import APIRouter

from app.api.v1.routers.alerts import router as alerts_router
from app.api.v1.routers.analytics import router as analytics_router
from app.api.v1.routers.cameras import router as cameras_router
from app.api.v1.routers.detections import router as detections_router
from app.api.v1.routers.health import router as health_router
from app.api.v1.routers.maintenance import router as maintenance_router
from app.websocket.routes import router as websocket_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(cameras_router)
api_router.include_router(detections_router)
api_router.include_router(alerts_router)
api_router.include_router(analytics_router)
api_router.include_router(maintenance_router)
api_router.include_router(websocket_router)

