"""Real-time source-timeline pacing for finite uploaded/recorded video.

The decoder for a local file can return frames much faster than wall-clock time.
This helper maps media presentation timestamps (PTS) to ``time.monotonic()`` so
preview and analytics advance at the recording's original speed.

If source PTS is unavailable, the clock falls back to the decoder-reported FPS.
A bounded lag rebase prevents a slow inference/CPU spike from causing a later
"catch-up burst" where several frames are published too quickly.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in _TRUE_VALUES


def _env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


class UploadPlaybackClock:
    """Map uploaded-video media time to monotonic wall time."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        playback_rate: float = 1.0,
        fallback_fps: float = 25.0,
        max_lag_seconds: float = 0.75,
        rewind_tolerance_seconds: float = 0.25,
    ) -> None:
        self.enabled = bool(enabled)
        self.playback_rate = max(0.05, min(8.0, float(playback_rate)))
        self.fallback_fps = max(0.1, min(240.0, float(fallback_fps)))
        self.max_lag_seconds = max(0.0, min(30.0, float(max_lag_seconds)))
        self.rewind_tolerance_seconds = max(
            0.0,
            min(10.0, float(rewind_tolerance_seconds)),
        )
        self._wall_anchor: float | None = None
        self._media_anchor: float | None = None
        self._last_media_seconds: float | None = None
        self._last_delay_seconds: float = 0.0
        self._rebases: int = 0

    @classmethod
    def from_env(
        cls,
        *,
        fallback_fps: float | None = None,
    ) -> "UploadPlaybackClock":
        fallback = (
            float(fallback_fps)
            if fallback_fps is not None
            else _env_float("UPLOAD_FALLBACK_FPS", 25.0, 0.1, 240.0)
        )
        return cls(
            enabled=_env_bool("UPLOAD_REALTIME_PACING_ENABLED", True),
            playback_rate=_env_float("UPLOAD_PLAYBACK_RATE", 1.0, 0.05, 8.0),
            fallback_fps=fallback,
            max_lag_seconds=_env_float(
                "UPLOAD_PACING_MAX_LAG_SECONDS",
                0.75,
                0.0,
                30.0,
            ),
            rewind_tolerance_seconds=_env_float(
                "UPLOAD_PTS_REWIND_TOLERANCE_SECONDS",
                0.25,
                0.0,
                10.0,
            ),
        )

    @staticmethod
    def _valid_seconds(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return result

    def set_fallback_fps(self, fps: Any) -> None:
        value = self._valid_seconds(fps)
        if value is not None and 0.1 <= value <= 240.0:
            self.fallback_fps = value

    def reset(self, *, fallback_fps: Any = None) -> None:
        if fallback_fps is not None:
            self.set_fallback_fps(fallback_fps)
        self._wall_anchor = None
        self._media_anchor = None
        self._last_media_seconds = None
        self._last_delay_seconds = 0.0

    def _resolve_media_seconds(self, pts_seconds: Any) -> float:
        media = self._valid_seconds(pts_seconds)

        if media is None:
            if self._last_media_seconds is None:
                return 0.0
            return self._last_media_seconds + (1.0 / self.fallback_fps)

        if self._last_media_seconds is not None:
            if media < (
                self._last_media_seconds
                - self.rewind_tolerance_seconds
            ):
                # A seek/discontinuity/reopen starts a new wall-clock mapping.
                self._wall_anchor = None
                self._media_anchor = None
                self._last_media_seconds = None
            elif media <= self._last_media_seconds:
                # Duplicate/non-increasing PTS must not publish frames in a
                # zero-time burst. Fall back to one source-frame interval.
                media = self._last_media_seconds + (1.0 / self.fallback_fps)

        return media

    def delay_seconds(
        self,
        pts_seconds: Any,
        *,
        now_monotonic: float | None = None,
    ) -> float:
        """Return delay required before publishing this media frame."""
        now = (
            float(now_monotonic)
            if now_monotonic is not None
            else time.monotonic()
        )
        media = self._resolve_media_seconds(pts_seconds)

        if not self.enabled:
            self._last_media_seconds = media
            self._last_delay_seconds = 0.0
            return 0.0

        if self._wall_anchor is None or self._media_anchor is None:
            self._wall_anchor = now
            self._media_anchor = media
            self._last_media_seconds = media
            self._last_delay_seconds = 0.0
            return 0.0

        elapsed_media = max(0.0, media - self._media_anchor)
        target = self._wall_anchor + (elapsed_media / self.playback_rate)
        lag = now - target

        # Rebase when processing falls materially behind. Without this, a slow
        # frame can be followed by a burst of frames as the loop tries to catch
        # up to the old wall-clock anchor.
        if (
            self.max_lag_seconds > 0.0
            and lag > self.max_lag_seconds
        ):
            self._wall_anchor += lag
            target = now
            self._rebases += 1

        delay = max(0.0, target - now)
        self._last_media_seconds = media
        self._last_delay_seconds = delay
        return delay

    def wait_for_frame(
        self,
        pts_seconds: Any,
        *,
        stop_event: threading.Event | None = None,
    ) -> float:
        delay = self.delay_seconds(pts_seconds)
        if delay <= 0.0:
            return 0.0

        if stop_event is not None:
            stop_event.wait(delay)
        else:
            time.sleep(delay)
        return delay

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "playback_rate": self.playback_rate,
            "fallback_fps": self.fallback_fps,
            "last_media_seconds": self._last_media_seconds,
            "last_delay_seconds": self._last_delay_seconds,
            "rebases": self._rebases,
        }
