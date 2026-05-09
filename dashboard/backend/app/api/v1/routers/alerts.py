"""Alert and incident routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
async def list_alerts() -> dict[str, list[dict[str, str]]]:
    return {"items": []}
