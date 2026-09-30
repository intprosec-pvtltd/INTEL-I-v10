"""PTS-first media timeline and loop/hard-cut discontinuity detection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import os
import threading
from typing import Any

import numpy as np


class SourcePTSRequiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimelineObservation:
    analytics_seconds: float
    normalized_timestamp: datetime
    pts_seconds: float | None
    timestamp_source: str
    timestamp_quality: str
    discontinuity: bool
    discontinuity_reason: str | None
    scene_change_score: float
    observed_fps: float | None
    discontinuity_count: int


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class MediaTimeline:
    """Turn decoder PTS into monotonic analytics time and evidence timestamps.

    A new segment begins after a backward PTS jump, an excessive PTS gap, or a
    visual hard cut. Callers reset camera-local trackers whenever
    ``discontinuity`` is true. PTS is never replaced with a fixed-FPS counter.
    """

    def __init__(
        self,
        camera_id: str,
        *,
        strict_pts: bool = True,
        gap_threshold_seconds: float = 3.0,
        backward_tolerance_seconds: float = 0.10,
        scene_cut_threshold: float = 0.42,
        scene_cut_cooldown_seconds: float = 1.0,
    ):
        self.camera_id = str(camera_id)
        self.strict_pts = bool(strict_pts)
        self.gap_threshold_seconds = max(0.25, float(gap_threshold_seconds))
        self.backward_tolerance_seconds = max(0.0, float(backward_tolerance_seconds))
        self.scene_cut_threshold = max(0.10, min(1.0, float(scene_cut_threshold)))
        self.scene_cut_cooldown_seconds = max(0.0, float(scene_cut_cooldown_seconds))
        self._lock = threading.RLock()
        self._segment_pts_origin: float | None = None
        self._segment_monotonic_origin: float | None = None
        self._segment_wall_origin: datetime | None = None
        self._last_pts: float | None = None
        self._last_analytics = 0.0
        self._last_signature: np.ndarray | None = None
        self._last_cut_analytics = -1e9
        self._fps_ema: float | None = None
        self._discontinuity_count = 0
        self._forced_reason: str | None = None

    @staticmethod
    def _signature(frame: Any) -> np.ndarray | None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        array = frame.astype(np.float32, copy=False)
        if array.ndim == 3:
            array = array[..., :3].mean(axis=2)
        elif array.ndim != 2:
            return None
        height, width = array.shape[:2]
        if height < 2 or width < 2:
            return None
        y = np.linspace(0, height - 1, num=min(36, height), dtype=np.intp)
        x = np.linspace(0, width - 1, num=min(64, width), dtype=np.intp)
        return array[np.ix_(y, x)] / 255.0

    def force_discontinuity(self, reason: str) -> None:
        with self._lock:
            self._forced_reason = str(reason or "external_discontinuity")[:80]

    def _new_segment(self, pts: float | None, received_at: datetime, received_monotonic: float) -> None:
        self._segment_pts_origin = pts
        self._segment_monotonic_origin = received_monotonic
        self._segment_wall_origin = received_at
        self._last_analytics = 0.0
        self._last_cut_analytics = -1e9

    def observe(
        self,
        *,
        pts_seconds: float | None,
        received_at: datetime,
        received_monotonic: float,
        frame: Any = None,
    ) -> TimelineObservation:
        received = received_at if received_at.tzinfo else received_at.replace(tzinfo=timezone.utc)
        received = received.astimezone(timezone.utc)
        pts = _finite(pts_seconds)
        monotonic_value = _finite(received_monotonic)
        if monotonic_value is None:
            raise ValueError("received_monotonic must be finite")

        with self._lock:
            if pts is None and self.strict_pts:
                raise SourcePTSRequiredError(f"Decoder PTS unavailable for camera {self.camera_id}")

            first = self._segment_wall_origin is None
            reason = self._forced_reason
            self._forced_reason = None
            signature = self._signature(frame)
            scene_score = 0.0

            if pts is not None and self._last_pts is not None:
                delta = pts - self._last_pts
                if delta < -self.backward_tolerance_seconds:
                    reason = reason or "PTS_BACKWARD_LOOP"
                elif delta > self.gap_threshold_seconds:
                    reason = reason or "PTS_GAP"
                elif delta > 1e-6:
                    instant_fps = min(240.0, 1.0 / delta)
                    self._fps_ema = instant_fps if self._fps_ema is None else 0.15 * instant_fps + 0.85 * self._fps_ema

            if signature is not None and self._last_signature is not None and self._last_signature.shape == signature.shape:
                scene_score = float(np.mean(np.abs(signature - self._last_signature)))
                if (
                    scene_score >= self.scene_cut_threshold
                    and self._last_analytics - self._last_cut_analytics >= self.scene_cut_cooldown_seconds
                ):
                    reason = reason or "SCENE_HARD_CUT"

            discontinuity = bool(reason and not first)
            if first or discontinuity:
                self._new_segment(pts, received, monotonic_value)
                if discontinuity:
                    self._discontinuity_count += 1
                    self._last_cut_analytics = 0.0

            if pts is not None:
                if self._segment_pts_origin is None:
                    self._segment_pts_origin = pts
                analytics = max(0.0, pts - self._segment_pts_origin)
                timestamp_source = "PTS_RELATIVE"
                timestamp_quality = "SOURCE_PTS"
            else:
                analytics = max(0.0, monotonic_value - float(self._segment_monotonic_origin or monotonic_value))
                timestamp_source = "ARRIVAL_MONOTONIC"
                timestamp_quality = "ESTIMATED"

            normalized = (self._segment_wall_origin or received) + timedelta(seconds=analytics)
            self._last_pts = pts
            self._last_analytics = analytics
            self._last_signature = signature
            return TimelineObservation(
                analytics_seconds=analytics,
                normalized_timestamp=normalized,
                pts_seconds=pts,
                timestamp_source=timestamp_source,
                timestamp_quality=timestamp_quality,
                discontinuity=discontinuity,
                discontinuity_reason=reason if discontinuity else None,
                scene_change_score=scene_score,
                observed_fps=self._fps_ema,
                discontinuity_count=self._discontinuity_count,
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "strict_pts": self.strict_pts,
                "last_pts_seconds": self._last_pts,
                "last_analytics_seconds": self._last_analytics,
                "observed_fps": self._fps_ema,
                "discontinuity_count": self._discontinuity_count,
            }


def timeline_from_environment(camera_id: str, *, strict_pts: bool) -> MediaTimeline:
    def number(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default
    return MediaTimeline(
        camera_id,
        strict_pts=strict_pts,
        gap_threshold_seconds=number("PTS_GAP_THRESHOLD_SECONDS", 3.0),
        backward_tolerance_seconds=number("PTS_BACKWARD_TOLERANCE_SECONDS", 0.10),
        scene_cut_threshold=number("SCENE_HARD_CUT_THRESHOLD", 0.42),
        scene_cut_cooldown_seconds=number("SCENE_HARD_CUT_COOLDOWN_SECONDS", 1.0),
    )

