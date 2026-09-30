"""Canonical timestamp and camera-clock synchronization primitives."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import platform
import re
import subprocess
import threading
import time
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class TimestampBundle:
    camera_timestamp: datetime | None
    server_timestamp: datetime
    ingestion_timestamp: datetime
    normalized_timestamp: datetime
    clock_offset_ms: float | None
    sync_status: str
    sync_method: str
    pts_seconds: float | None = None
    timestamp_quality: str = "UNKNOWN"
    discontinuity: bool = False
    discontinuity_reason: str | None = None


@dataclass
class CameraClock:
    offset_ms: float | None = None
    jitter_ms: float | None = None
    samples: int = 0
    last_sync_at: datetime | None = None
    status: str = "UNSYNCED"
    method: str = "ARRIVAL_TIME"
    pts_seconds: float | None = None
    timestamp_quality: str = "UNKNOWN"
    discontinuity_count: int = 0
    last_discontinuity_reason: str | None = None


class CameraTimeSynchronizer:
    """Thread-safe per-camera clock estimator.

    Production callers provide the PTS-anchored timestamp from MediaTimeline.
    Arrival time exists only for the explicitly configured non-strict legacy
    mode and is marked ESTIMATED rather than presented as source timing.
    """
    def __init__(self, max_offset_ms: float = 5000.0):
        self.max_offset_ms = float(max_offset_ms)
        self._clocks: dict[str, CameraClock] = {}
        self._lock = threading.RLock()

    def _clock(self, camera_id: str) -> CameraClock:
        return self._clocks.setdefault(str(camera_id), CameraClock())

    def observe(
        self,
        camera_id: str,
        *,
        camera_timestamp: datetime | None,
        server_timestamp: datetime | None = None,
        pts_seconds: float | None = None,
        timestamp_source: str | None = None,
        timestamp_quality: str | None = None,
        discontinuity: bool = False,
        discontinuity_reason: str | None = None,
    ) -> TimestampBundle:
        server = server_timestamp or utc_now()
        ingestion = utc_now()
        camera_ts = camera_timestamp.astimezone(timezone.utc) if camera_timestamp and camera_timestamp.tzinfo else (camera_timestamp.replace(tzinfo=timezone.utc) if camera_timestamp else None)
        with self._lock:
            clock = self._clock(camera_id)
            method = str(timestamp_source or "").strip().upper()
            quality = str(timestamp_quality or "").strip().upper() or "UNKNOWN"
            clock.pts_seconds = float(pts_seconds) if pts_seconds is not None else None
            clock.timestamp_quality = quality
            if discontinuity:
                clock.discontinuity_count += 1
                clock.last_discontinuity_reason = str(discontinuity_reason or "UNKNOWN")[:80]
            if camera_ts is None:
                clock.method = method or "ARRIVAL_TIME"
                clock.status = "ESTIMATED"
                normalized = server
                offset = 0.0
            elif method.startswith("PTS"):
                # Relative decoder PTS is anchored once to UTC by MediaTimeline.
                # It provides correct frame-to-frame timing without pretending
                # to be an independent camera-wall-clock measurement.
                offset = 0.0
                clock.offset_ms = 0.0
                clock.samples += 1
                clock.last_sync_at = server
                clock.method = method
                clock.status = "PTS_ALIGNED"
                normalized = camera_ts
            else:
                offset = (server - camera_ts).total_seconds() * 1000.0
                if clock.offset_ms is not None:
                    delta = abs(offset - clock.offset_ms)
                    clock.jitter_ms = delta if clock.jitter_ms is None else 0.2 * delta + 0.8 * clock.jitter_ms
                clock.offset_ms = offset if clock.offset_ms is None else 0.2 * offset + 0.8 * clock.offset_ms
                clock.samples += 1
                clock.last_sync_at = server
                clock.method = method or "SOURCE_TIMESTAMP"
                clock.status = "SYNCED" if abs(clock.offset_ms) <= self.max_offset_ms else "OUT_OF_SYNC"
                normalized = camera_ts + __import__('datetime').timedelta(milliseconds=clock.offset_ms)
            return TimestampBundle(
                camera_timestamp=camera_ts,
                server_timestamp=server,
                ingestion_timestamp=ingestion,
                normalized_timestamp=normalized,
                clock_offset_ms=clock.offset_ms if camera_ts is not None else 0.0,
                sync_status=clock.status,
                sync_method=clock.method,
                pts_seconds=clock.pts_seconds,
                timestamp_quality=clock.timestamp_quality,
                discontinuity=bool(discontinuity),
                discontinuity_reason=clock.last_discontinuity_reason if discontinuity else None,
            )

    def status(self, camera_id: str) -> dict[str, Any]:
        with self._lock:
            c = self._clock(camera_id)
            return {
                "camera_id": str(camera_id),
                "offset_ms": c.offset_ms,
                "jitter_ms": c.jitter_ms,
                "samples": c.samples,
                "last_sync_at": c.last_sync_at.isoformat() if c.last_sync_at else None,
                "sync_status": c.status,
                "method": c.method,
                "source_timestamp_available": c.samples > 0,
                "pts_seconds": c.pts_seconds,
                "timestamp_quality": c.timestamp_quality,
                "discontinuity_count": c.discontinuity_count,
                "last_discontinuity_reason": c.last_discontinuity_reason,
            }

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.status(k) for k in self._clocks]


def system_clock_health() -> dict[str, Any]:
    """Best-effort OS clock synchronization status; never exposes secrets."""
    system = platform.system().lower()
    command: list[str] | None = None
    if system == "windows":
        command = ["w32tm", "/query", "/status"]
    elif system == "linux":
        command = ["timedatectl", "show-timesync", "--all", "--no-pager"]
    if not command:
        return {"available": False, "status": "UNSUPPORTED", "platform": system}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
        text = (result.stdout or "")[:4000]
        ok = result.returncode == 0 and bool(text.strip())
        return {
            "available": ok,
            "status": "SYNC_SERVICE_REPORTED" if ok else "UNKNOWN",
            "platform": system,
            "details": text if ok else None,
        }
    except Exception:
        return {"available": False, "status": "UNKNOWN", "platform": system}


def safe_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


time_synchronizer = CameraTimeSynchronizer(
    max_offset_ms=float(os.getenv("CAMERA_MAX_CLOCK_OFFSET_MS", "5000"))
)
