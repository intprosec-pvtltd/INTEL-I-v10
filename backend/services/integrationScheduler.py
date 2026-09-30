"""Async lifecycle wrapper for bounded automatic catalogue synchronization."""
from __future__ import annotations

import asyncio
import logging
import os

from services.cameraIntegration import sync_due_integrations


logger = logging.getLogger(__name__)
_task: asyncio.Task | None = None
_stop_event: asyncio.Event | None = None


def enabled() -> bool:
    return os.getenv("CAMERA_INTEGRATION_AUTO_SYNC_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def scheduler_interval_seconds() -> float:
    try:
        return max(15.0, min(300.0, float(os.getenv("CAMERA_INTEGRATION_SCHEDULER_SECONDS", "30"))))
    except (TypeError, ValueError):
        return 30.0


async def _loop() -> None:
    global _stop_event
    interval = scheduler_interval_seconds()
    while _stop_event is not None and not _stop_event.is_set():
        try:
            await asyncio.to_thread(sync_due_integrations)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Automatic camera catalogue synchronization iteration failed")
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


def start_integration_scheduler() -> asyncio.Task | None:
    global _task, _stop_event
    if not enabled() or (_task and not _task.done()):
        return _task
    _stop_event = asyncio.Event()
    _task = asyncio.create_task(_loop(), name="camera-integration-sync")
    return _task


async def stop_integration_scheduler() -> None:
    global _task, _stop_event
    if _stop_event is not None:
        _stop_event.set()
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
    _task = None
    _stop_event = None

