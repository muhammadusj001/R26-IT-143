"""Analytics and reporting routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.analytics import CrowdAnalyticsRequest, CrowdAnalyticsResponse
from app.services.crowd_analytics_service import CrowdAnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


def get_crowd_analytics_service(request: Request) -> CrowdAnalyticsService:
    return request.app.state.crowd_analytics_service


@router.get("", response_model=CrowdAnalyticsResponse)
async def get_summary(
    crowd_service: CrowdAnalyticsService = Depends(get_crowd_analytics_service),
) -> CrowdAnalyticsResponse:
    return CrowdAnalyticsResponse(**crowd_service.get_current_status())


@router.post("/crowd", response_model=CrowdAnalyticsResponse)
async def record_crowd_reading(
    payload: CrowdAnalyticsRequest,
    crowd_service: CrowdAnalyticsService = Depends(get_crowd_analytics_service),
) -> CrowdAnalyticsResponse:
    try:
        status = crowd_service.record_reading(payload.swimmer_count, payload.interval_seconds)
        return CrowdAnalyticsResponse(**status)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        raise HTTPException(status_code=503, detail=str(exc)) from exc
