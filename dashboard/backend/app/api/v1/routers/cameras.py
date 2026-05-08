"""Camera registration and stream configuration routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.services.webcam_service import WebcamService, WebcamServiceError
from app.services.yolo_service import YOLOService, YOLOServiceError

router = APIRouter(prefix="/cameras", tags=["cameras"])


def get_webcam_service(request: Request) -> WebcamService:
    return request.app.state.webcam_service


def get_yolo_service(request: Request) -> YOLOService:
    return request.app.state.yolo_service


@router.get("")
async def list_cameras() -> dict[str, list[dict[str, str]]]:
    return {"items": []}


@router.get("/webcam/status")
async def webcam_status(webcam_service: WebcamService = Depends(get_webcam_service)) -> dict[str, object]:
    status = webcam_service.get_status()
    if not status["running"]:
        raise HTTPException(status_code=503, detail="Webcam service is not available")
    return status


@router.get("/webcam/test")
async def test_webcam(webcam_service: WebcamService = Depends(get_webcam_service)) -> dict[str, object]:
    try:
        return webcam_service.probe_frame()
    except WebcamServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/webcam/detection-test")
async def test_webcam_detection(
    webcam_service: WebcamService = Depends(get_webcam_service),
    yolo_service: YOLOService = Depends(get_yolo_service),
) -> dict[str, object]:
    if not yolo_service.is_ready():
        raise HTTPException(status_code=503, detail="YOLO model is not ready")

    try:
        webcam_status = webcam_service.probe_frame()
        return {
            **webcam_status,
            "detection": yolo_service.get_status(),
        }
    except (WebcamServiceError, YOLOServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/webcam/stream")
async def webcam_stream(
    webcam_service: WebcamService = Depends(get_webcam_service),
    yolo_service: YOLOService = Depends(get_yolo_service),
) -> StreamingResponse:
    if not yolo_service.is_ready():
        raise HTTPException(status_code=503, detail="YOLO model is not ready")

    try:
        webcam_service.start()
    except WebcamServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StreamingResponse(
        webcam_service.mjpeg_frame_generator(overlay_fn=yolo_service.annotate_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

