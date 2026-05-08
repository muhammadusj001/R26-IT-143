"""FastAPI application factory and global app instance."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.api import api_router
from app.core.config import settings
from app.db.database import verify_database_connection
from app.services.crowd_analytics_service import CrowdAnalyticsService, CrowdAnalyticsThresholds
from app.services.webcam_service import WebcamService, WebcamServiceError
from app.services.yolo_service import YOLOService, YOLOServiceError
from app.websocket.routes import manager as websocket_manager
from app.websocket.swimmer_analytics_broadcaster import SwimmerAnalyticsBroadcaster


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    webcam_service = WebcamService(camera_index=settings.webcam_default_index)
    yolo_service = YOLOService(model_path=settings.yolo_model_path)
    crowd_analytics_service = CrowdAnalyticsService(
        thresholds=CrowdAnalyticsThresholds(
            low=settings.density_low_threshold,
            medium=settings.density_medium_threshold,
            high=settings.density_high_threshold,
            capacity=settings.pool_capacity,
            chlorine_dose=settings.maintenance_chlorine_dose_threshold,
            filter_backwash=settings.maintenance_filter_backwash_threshold,
            skimmer_clean=settings.maintenance_skimmer_clean_threshold,
            shock_treatment=settings.maintenance_shock_treatment_threshold,
            deep_clean=settings.maintenance_deep_clean_threshold,
        )
    )
    analytics_broadcaster = SwimmerAnalyticsBroadcaster(
        manager=websocket_manager,
        webcam_service=webcam_service,
        yolo_service=yolo_service,
        crowd_analytics_service=crowd_analytics_service,
        interval_seconds=1.0,
    )
    app.state.webcam_service = webcam_service
    app.state.yolo_service = yolo_service
    app.state.crowd_analytics_service = crowd_analytics_service
    app.state.websocket_manager = websocket_manager
    app.state.analytics_broadcaster = analytics_broadcaster

    try:
        webcam_service.start()
    except WebcamServiceError:
        pass

    try:
        yolo_service.load_model()
    except YOLOServiceError:
        pass

    try:
        await verify_database_connection()
    except SQLAlchemyError as exc:
        logger.exception("Database connection verification failed")
        raise RuntimeError("Database connection verification failed") from exc

    analytics_broadcaster.start()

    try:
        yield
    finally:
        await analytics_broadcaster.stop()
        webcam_service.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.project_name,
        version=settings.project_version,
        description="Backend for real-time AI swimming pool monitoring.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(_: object, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("Database operation failed")
        return JSONResponse(status_code=503, content={"detail": "Database operation failed"})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: object, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled application error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
