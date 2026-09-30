"""Bounded, per-camera stream buffers and ingestion telemetry."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time
from collections import deque
from typing import Any


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame: Any
    frame_number: int
    received_at: datetime
    received_monotonic: float
    source_timestamp: datetime | None
    width: int
    height: int
    pts_seconds: float | None = None
    time_base: str | None = None
    timestamp_source: str = "UNKNOWN"
    timestamp_quality: str = "UNKNOWN"
    analytics_seconds: float = 0.0
    discontinuity: bool = False
    discontinuity_reason: str | None = None
    key_frame: bool = False


class BoundedFrameBuffer:
    def __init__(self, capacity: int = 3):
        self.capacity = max(1, int(capacity))
        self._items: deque[FramePacket] = deque(maxlen=self.capacity)
        self._lock = threading.Condition()
        self.dropped = 0

    def put_latest(self, packet: FramePacket) -> int:
        with self._lock:
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped += 1
            self._items.append(packet)
            self._lock.notify()
            return self.dropped

    def get_latest(self, timeout: float = 0.2) -> FramePacket | None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(remaining)
            packet = self._items[-1]
            self._items.clear()
            return packet

    def depth(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class StreamManager:
    def __init__(self, default_buffer_size: int = 3):
        self.default_buffer_size = max(1, int(default_buffer_size))
        self._buffers: dict[str, BoundedFrameBuffer] = {}
        self._lock = threading.RLock()

    def ensure(self, camera_id: str, capacity: int | None = None) -> BoundedFrameBuffer:
        camera_id = str(camera_id)
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = BoundedFrameBuffer(capacity or self.default_buffer_size)
            return self._buffers[camera_id]

    def ingest(self, packet: FramePacket) -> int:
        return self.ensure(packet.camera_id).put_latest(packet)

    def next_frame(self, camera_id: str, timeout: float = 0.2) -> FramePacket | None:
        return self.ensure(camera_id).get_latest(timeout)

    def stats(self, camera_id: str) -> dict[str, int]:
        buf = self.ensure(camera_id)
        return {"queue_depth": buf.depth(), "queue_capacity": buf.capacity, "dropped_frames": buf.dropped}

    def clear(self, camera_id: str) -> None:
        with self._lock:
            buf = self._buffers.get(str(camera_id))
            if buf:
                buf.clear()

    def remove(self, camera_id: str) -> None:
        with self._lock:
            self._buffers.pop(str(camera_id), None)

    def all_stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {camera_id: self.stats(camera_id) for camera_id in self._buffers}


stream_manager = StreamManager()
