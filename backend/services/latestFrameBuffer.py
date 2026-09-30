"""Thread-safe single-frame handoff for latency-sensitive live video."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass(frozen=True)
class LatestFrame:
    frame: Any
    timestamp: float
    sequence: int
    metadata: dict[str, Any]


class LatestFrameBuffer:
    """Keep only the newest frame; old frames are intentionally discarded."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._latest: LatestFrame | None = None
        self._sequence = 0
        self._closed = False
        self._consumed_sequence = 0
        self.dropped_frames = 0

    def put(self, frame: Any, timestamp: float, **metadata: Any) -> LatestFrame:
        with self._condition:
            if self._closed:
                raise RuntimeError("LatestFrameBuffer is closed")
            if self._latest is not None and self._latest.sequence > self._consumed_sequence:
                self.dropped_frames += 1
            self._sequence += 1
            self._latest = LatestFrame(frame, float(timestamp), self._sequence, dict(metadata))
            self._condition.notify_all()
            return self._latest

    def get(self, *, after_sequence: int = 0, timeout: float = 0.2) -> LatestFrame | None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed and (
                self._latest is None or self._latest.sequence <= int(after_sequence)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._latest is not None:
                self._consumed_sequence = self._latest.sequence
            return self._latest

    def pending_after(self, sequence: int) -> int:
        """Return 1 when a newer frame is waiting, otherwise 0.

        This buffer intentionally never grows beyond one frame, so exposing a
        binary depth gives load balancers a meaningful pressure signal without
        confusing cumulative stale-frame drops with current queue depth.
        """
        with self._condition:
            return int(self._latest is not None and self._latest.sequence > int(sequence))

    def clear(self) -> None:
        with self._condition:
            self._latest = None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._latest = None
            self._condition.notify_all()
