"""WebSocket routes for live swimmer analytics updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websocket.manager import ConnectionManager

router = APIRouter(tags=["websocket"])
manager = ConnectionManager()


@router.websocket("/ws/swimmer-analytics")
async def swimmer_analytics_socket(websocket: WebSocket, client_id: str = "default") -> None:
    """Accept WebSocket connection, track client, receive ping, and broadcast analytics updates."""
    await manager.connect(client_id=client_id, websocket=websocket)
    try:
        while True:
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_json({"type": "pong", "client_id": client_id})
    except WebSocketDisconnect:
        await manager.disconnect(client_id=client_id, websocket=websocket)
    except Exception:
        await manager.disconnect(client_id=client_id, websocket=websocket)
