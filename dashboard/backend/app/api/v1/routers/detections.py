"""Detection and inference routes."""

from fastapi import APIRouter, Depends, Request

from app.services.yolo_service import YOLOService

router = APIRouter(prefix="/detections", tags=["detections"])


def get_yolo_service(request: Request) -> YOLOService:
    return request.app.state.yolo_service


@router.get("")
async def list_detections() -> dict[str, list[dict[str, str]]]:
    return {"items": []}


@router.get("/model/status")
async def model_status(yolo_service: YOLOService = Depends(get_yolo_service)) -> dict[str, object]:
    return yolo_service.get_status()
