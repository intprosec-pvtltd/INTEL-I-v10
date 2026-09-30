"""Track-scoped caches for expensive secondary AI operations."""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Any


def _float_env(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


@dataclass
class CacheEntry:
    value: Any = None
    quality: float = 0.0
    last_attempt: float = 0.0
    updated_at: float = 0.0
    expires_at: float = 0.0
    successful: bool = False


class TrackResultCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._values: dict[tuple[str, str, str], CacheEntry] = {}
        self.default_ttl = _float_env("TRACK_CACHE_TTL_SECONDS", 15.0, 1.0, 600.0)
        self.quality_improvement = _float_env(
            "TRACK_CACHE_QUALITY_IMPROVEMENT", 0.08, 0.0, 1.0
        )

    @staticmethod
    def _key(camera_id: str, track_id: Any, kind: str) -> tuple[str, str, str]:
        return (str(camera_id), str(track_id), str(kind).strip().lower())

    def get(self, camera_id: str, track_id: Any, kind: str) -> CacheEntry | None:
        key = self._key(camera_id, track_id, kind)
        now = time.monotonic()
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            if entry.expires_at and entry.expires_at <= now:
                self._values.pop(key, None)
                return None
            return CacheEntry(**entry.__dict__)

    def should_run(
        self,
        camera_id: str,
        track_id: Any,
        kind: str,
        *,
        quality: float = 0.0,
        refresh_seconds: float = 2.0,
        success_refresh_seconds: float | None = None,
        force: bool = False,
    ) -> tuple[bool, str]:
        if force:
            return True, "forced"
        now = time.monotonic()
        quality = max(0.0, min(1.0, float(quality or 0.0)))
        entry = self.get(camera_id, track_id, kind)
        if entry is None:
            return True, "new_track"
        refresh = max(0.05, float(refresh_seconds))
        if entry.successful and success_refresh_seconds is not None:
            refresh = max(refresh, float(success_refresh_seconds))
        if quality >= entry.quality + self.quality_improvement:
            return True, "better_quality"
        if now - entry.last_attempt >= refresh:
            return True, "refresh"
        return False, "cache"

    def mark_attempt(
        self,
        camera_id: str,
        track_id: Any,
        kind: str,
        *,
        quality: float = 0.0,
        value: Any = None,
        successful: bool = False,
        ttl_seconds: float | None = None,
    ) -> CacheEntry:
        key = self._key(camera_id, track_id, kind)
        now = time.monotonic()
        ttl = self.default_ttl if ttl_seconds is None else max(0.5, float(ttl_seconds))
        with self._lock:
            previous = self._values.get(key)
            best_quality = max(float(quality or 0.0), previous.quality if previous else 0.0)
            selected_value = value
            selected_success = bool(successful)
            if value is None and previous is not None and previous.successful:
                selected_value = previous.value
                selected_success = True
            entry = CacheEntry(
                value=selected_value,
                quality=best_quality,
                last_attempt=now,
                updated_at=now,
                expires_at=now + ttl,
                successful=selected_success,
            )
            self._values[key] = entry
            return CacheEntry(**entry.__dict__)

    def clear_camera(self, camera_id: str) -> int:
        prefix = str(camera_id)
        with self._lock:
            keys = [key for key in self._values if key[0] == prefix]
            for key in keys:
                self._values.pop(key, None)
            return len(keys)

    def cleanup(self) -> int:
        now = time.monotonic()
        with self._lock:
            keys = [
                key for key, value in self._values.items()
                if value.expires_at and value.expires_at <= now
            ]
            for key in keys:
                self._values.pop(key, None)
            return len(keys)

    def snapshot(self) -> dict[str, int]:
        self.cleanup()
        with self._lock:
            by_kind: dict[str, int] = {}
            for _, _, kind in self._values:
                by_kind[kind] = by_kind.get(kind, 0) + 1
            return {"entries": len(self._values), "by_kind": by_kind}


TRACK_CACHE = TrackResultCache()
