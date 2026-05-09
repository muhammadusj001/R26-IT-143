"""Async broadcaster for live swimmer analytics updates."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.services.crowd_analytics_service import CrowdAnalyticsService
from app.services.webcam_service import WebcamService
from app.services.yolo_service import YOLOService, YOLOServiceError
from app.websocket.manager import ConnectionManager


@dataclass
class SwimmerAnalyticsBroadcaster:
    manager: ConnectionManager
    webcam_service: WebcamService
    yolo_service: YOLOService
    crowd_analytics_service: CrowdAnalyticsService
    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False

        if self._task is None:
            return

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while self._running:
            await self._broadcast_once()
            await asyncio.sleep(self.interval_seconds)

    async def _broadcast_once(self) -> None:
        if not self.webcam_service.is_running() or not self.yolo_service.is_ready():
            return

        frame = self.webcam_service.get_latest_frame()
        if frame is None:
            return

        try:
            prediction = await asyncio.to_thread(self.yolo_service.predict_frame, frame)
        except YOLOServiceError:
            return

        status = self.crowd_analytics_service.record_reading(
            swimmer_count=prediction["count"],
            interval_seconds=max(1, int(self.interval_seconds)),
        )

        webcam_status = self.webcam_service.get_status()
        payload = {
            "type": "swimmer.analytics",
            "swimmer_count": status["swimmer_count"],
            "crowd_level": status["crowd_density_level"],
            "maintenance_urgency": status["maintenance_urgency"],
            "fps": webcam_status["actual_fps"],
            "status": status,
        }
        await self.manager.broadcast_json(payload)
