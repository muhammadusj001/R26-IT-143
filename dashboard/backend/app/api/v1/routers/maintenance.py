"""Maintenance recommendation routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/maintenance", tags=["maintenance"])


@router.get("")
async def list_recommendations() -> dict[str, list[dict[str, str]]]:
    return {"items": []}
