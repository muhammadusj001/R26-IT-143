"""WebSocket connection manager with reconnect-aware client tracking."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            previous = self.active_connections.get(client_id)
            if previous is not None and previous is not websocket:
                await self._safe_close(previous)
            self.active_connections[client_id] = websocket

    async def disconnect(self, client_id: str, websocket: WebSocket | None = None) -> None:
        async with self._lock:
            current = self.active_connections.get(client_id)
            if websocket is None or current is websocket:
                self.active_connections.pop(client_id, None)

    async def broadcast_json(self, message: dict) -> None:
        stale_clients: list[str] = []

        async with self._lock:
            items = list(self.active_connections.items())

        for client_id, connection in items:
            try:
                await connection.send_json(message)
            except Exception:
                stale_clients.append(client_id)

        if stale_clients:
            async with self._lock:
                for client_id in stale_clients:
                    self.active_connections.pop(client_id, None)

    async def _safe_close(self, websocket: WebSocket) -> None:
        try:
            await websocket.close()
        except Exception:
            pass
