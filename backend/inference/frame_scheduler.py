from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .policy import CAMERA_POLICIES
from .settings import SETTINGS


@dataclass
class CameraScheduleState:
    last_submit: float = 0.0
    active_until: float = 0.0
    incident_until: float = 0.0
    critical_until: float = 0.0
    last_priority: int = 5
    last_reason: str = "idle"


class FrameScheduler:
    """Adaptive camera scheduler with queue-pressure degradation."""

    def __init__(self) -> None:
        self._states: dict[str, CameraScheduleState] = {}
        self._lock = threading.RLock()
        self.load_factor = 0.0

    def clear(self, camera_id: str) -> None:
        with self._lock:
            self._states.pop(str(camera_id), None)

    def _state(self, camera_id: str) -> CameraScheduleState:
        return self._states.setdefault(str(camera_id), CameraScheduleState())

    def mark_active(self, camera_id: str, seconds: float = 3.0, reason: str = "active_track") -> None:
        with self._lock:
            state = self._state(camera_id)
            state.active_until = max(state.active_until, time.monotonic() + max(0.1, seconds))
            state.last_reason = reason
            state.last_priority = min(state.last_priority, 4)

    def mark_incident(self, camera_id: str, seconds: float = 5.0, reason: str = "incident") -> None:
        with self._lock:
            state = self._state(camera_id)
            state.incident_until = max(state.incident_until, time.monotonic() + max(0.1, seconds))
            state.last_reason = reason
            state.last_priority = min(state.last_priority, 2)

    def mark_critical(self, camera_id: str, seconds: float = 5.0, reason: str = "critical_incident") -> None:
        with self._lock:
            state = self._state(camera_id)
            state.critical_until = max(state.critical_until, time.monotonic() + max(0.1, seconds))
            state.last_reason = reason
            state.last_priority = 0

    def update_hint(
        self,
        camera_id: str,
        *,
        mode: str | None = None,
        priority: int | None = None,
        reason: str | None = None,
        load_factor: float | None = None,
    ) -> None:
        if load_factor is not None:
            self.load_factor = max(0.0, min(1.0, float(load_factor)))
        normalized = str(mode or "").strip().lower()
        if normalized == "critical":
            self.mark_critical(camera_id, reason=reason or "critical")
        elif normalized == "incident":
            self.mark_incident(camera_id, reason=reason or "incident")
        elif normalized == "active":
            self.mark_active(camera_id, reason=reason or "active")
        with self._lock:
            state = self._state(camera_id)
            if priority is not None:
                state.last_priority = max(0, min(5, int(priority)))
            if reason:
                state.last_reason = str(reason)[:80]

    def mode(self, camera_id: str) -> str:
        now = time.monotonic()
        with self._lock:
            state = self._state(camera_id)
            if state.critical_until > now:
                return "critical"
            if state.incident_until > now:
                return "incident"
            if state.active_until > now:
                return "active"
            return "idle"

    def target_fps(self, camera_id: str, priority: int | None = None) -> float:
        policy = CAMERA_POLICIES.get(str(camera_id))
        mode = self.mode(camera_id)
        if not SETTINGS.adaptive_fps:
            fps = policy.active_fps
        elif mode == "critical":
            fps = max(policy.incident_fps, SETTINGS.critical_fps)
        elif mode == "incident":
            fps = policy.incident_fps
        elif mode == "active":
            fps = policy.active_fps
        else:
            fps = policy.idle_fps

        effective_priority = 5 if priority is None else max(0, min(5, int(priority)))
        if SETTINGS.backpressure_enabled:
            # Critical/watchlist work keeps useful sampling while lower priority
            # surveillance degrades first.
            if self.load_factor >= 0.95:
                if effective_priority >= 5:
                    fps = min(fps, 0.5)
                elif effective_priority >= 4:
                    fps = min(fps, 1.0)
                elif effective_priority >= 3:
                    fps = min(fps, 2.0)
            elif self.load_factor >= 0.80:
                if effective_priority >= 5:
                    fps = min(fps, 1.0)
                elif effective_priority >= 4:
                    fps = min(fps, 2.0)
            elif self.load_factor >= 0.65 and effective_priority >= 5:
                fps = min(fps, 1.5)
        return max(0.1, float(fps))

    def priority(self, camera_id: str) -> int:
        mode = self.mode(camera_id)
        with self._lock:
            state = self._state(camera_id)
            hinted = state.last_priority
        if mode == "critical":
            return 0
        if mode == "incident":
            return min(2, hinted)
        if mode == "active":
            return min(4, hinted)
        return max(4, hinted)

    def should_submit(self, camera_id: str, priority: int | None = None) -> bool:
        now = time.monotonic()
        with self._lock:
            state = self._state(camera_id)
            effective_priority = self.priority(camera_id) if priority is None else priority
            interval = 1.0 / self.target_fps(camera_id, effective_priority)
            if now - state.last_submit < interval:
                return False
            state.last_submit = now
            return True

    def set_load(self, depth: int, capacity: int) -> None:
        self.load_factor = 0.0 if capacity <= 0 else min(1.0, max(0.0, depth / capacity))

    def status(self, camera_id: str | None = None) -> dict:
        if camera_id is not None:
            return {
                "mode": self.mode(camera_id),
                "target_fps": self.target_fps(camera_id, self.priority(camera_id)),
                "priority": self.priority(camera_id),
                "load_factor": round(self.load_factor, 4),
            }
        with self._lock:
            ids = list(self._states)
        return {
            "load_factor": round(self.load_factor, 4),
            "cameras": {camera_id: self.status(camera_id) for camera_id in ids},
        }

