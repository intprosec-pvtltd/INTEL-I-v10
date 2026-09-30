"""Per-camera/per-zone AI execution policy and behavioural trigger decisions."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import threading
import time
from typing import Any

from .settings import SETTINGS


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CameraAIPolicy:
    object_detection: bool = True
    person_tracking: bool = True
    vehicle_tracking: bool = True
    anpr: bool = True
    vehicle_reid: bool = True
    frs: bool = True
    person_reid: bool = True
    behaviour: bool = True
    pose: bool = True
    mask: bool = False
    darkir: bool = True
    edsr: bool = True
    evidence: bool = True
    idle_fps: float = SETTINGS.idle_fps
    active_fps: float = SETTINGS.active_fps
    incident_fps: float = SETTINGS.incident_fps


class CameraPolicyStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._policies: dict[str, CameraAIPolicy] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def _default(self) -> CameraAIPolicy:
        return CameraAIPolicy(
            anpr=_env_bool("ANPR_ENABLED", True),
            vehicle_reid=_env_bool("VEHICLE_REID_ENABLED", True),
            frs=_env_bool("FRS_ENABLED", True),
            person_reid=_env_bool("PERSON_REID_ENABLED", True),
            behaviour=_env_bool("BEHAVIOUR_ENABLED", True),
            pose=_env_bool("POSE_ENABLED", True),
            mask=_env_bool("FACE_MASK_ENABLED", False),
            darkir=_env_bool("DARKIR_ENABLED", True),
            edsr=_env_bool("EDSR_ENABLED", True),
        )

    @staticmethod
    def _pick(configuration: dict[str, Any], *names: str, default: Any) -> Any:
        lowered = {str(k).strip().lower(): v for k, v in configuration.items()}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        return default

    @staticmethod
    def _bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}

    @staticmethod
    def _fps(value: Any, default: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.1, min(30.0, value))

    def configure(
        self,
        camera_id: str,
        configuration: dict[str, Any] | None,
        *,
        processing_fps: float | None = None,
        evidence_enabled: bool | None = None,
        zone: str | None = None,
        profile_name: str | None = None,
    ) -> CameraAIPolicy:
        base = self._default()
        cfg = configuration if isinstance(configuration, dict) else {}
        # Allow either flat configuration or {features:{...}, fps:{...}}.
        features = cfg.get("features") if isinstance(cfg.get("features"), dict) else cfg
        fps_cfg = cfg.get("fps") if isinstance(cfg.get("fps"), dict) else cfg
        kwargs = asdict(base)
        aliases = {
            "object_detection": ("object_detection", "detection"),
            "person_tracking": ("person_tracking",),
            "vehicle_tracking": ("vehicle_tracking",),
            "anpr": ("anpr", "plate_recognition"),
            "vehicle_reid": ("vehicle_reid", "vehicle_re_id"),
            "frs": ("frs", "face_recognition", "watchlist"),
            "person_reid": ("person_reid", "person_re_id"),
            "behaviour": ("behaviour", "behavior", "behavior_intelligence"),
            "pose": ("pose", "pose_estimation"),
            "mask": ("mask", "face_mask"),
            "darkir": ("darkir", "low_light_enhancement"),
            "edsr": ("edsr", "super_resolution"),
            "evidence": ("evidence",),
        }
        for field, names in aliases.items():
            current = bool(kwargs[field])
            kwargs[field] = self._bool(self._pick(features, *names, default=current), current)
        if evidence_enabled is not None:
            kwargs["evidence"] = bool(evidence_enabled)
        kwargs["idle_fps"] = self._fps(
            self._pick(fps_cfg, "idle_fps", "idle_detection_fps", default=base.idle_fps),
            base.idle_fps,
        )
        active_default = processing_fps if processing_fps is not None else base.active_fps
        kwargs["active_fps"] = self._fps(
            self._pick(fps_cfg, "active_fps", "active_detection_fps", default=active_default),
            active_default,
        )
        kwargs["incident_fps"] = self._fps(
            self._pick(fps_cfg, "incident_fps", "incident_detection_fps", default=base.incident_fps),
            base.incident_fps,
        )
        # Conservative profile presets only apply when not explicitly specified.
        zone_name = str(zone or "").strip().lower()
        if zone_name:
            if any(token in zone_name for token in ("road", "traffic", "highway")):
                if "frs" not in {str(k).lower() for k in features}:
                    kwargs["frs"] = False
                if "behaviour" not in {str(k).lower() for k in features}:
                    kwargs["behaviour"] = False
            elif any(token in zone_name for token in ("entry", "gate", "checkpoint")):
                kwargs["frs"] = self._bool(features.get("frs", True), True)
                kwargs["anpr"] = self._bool(features.get("anpr", True), True)
        policy = CameraAIPolicy(**kwargs)
        with self._lock:
            self._policies[str(camera_id)] = policy
            self._meta[str(camera_id)] = {
                "profile_name": profile_name,
                "zone": zone,
                "configured_at": time.time(),
            }
        return policy

    def get(self, camera_id: str) -> CameraAIPolicy:
        with self._lock:
            return self._policies.get(str(camera_id), self._default())

    def clear(self, camera_id: str) -> None:
        with self._lock:
            self._policies.pop(str(camera_id), None)
            self._meta.pop(str(camera_id), None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                camera_id: {**asdict(policy), **self._meta.get(camera_id, {})}
                for camera_id, policy in self._policies.items()
            }


@dataclass(frozen=True)
class BehaviourDecision:
    run_behaviour: bool
    run_pose: bool
    reason: str
    motion_score: float
    interaction: bool


class BehaviourGate:
    """Cheap cue gate used before expensive behaviour/pose models."""
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._centres: dict[tuple[str, str], tuple[float, float, float]] = {}
        self._last_run: dict[str, float] = {}
        self.stationary_refresh = max(
            0.5, float(os.getenv("BEHAVIOUR_STATIONARY_REFRESH_SECONDS", "3.0"))
        )
        self.motion_threshold = max(
            0.001, float(os.getenv("BEHAVIOUR_MOTION_THRESHOLD", "0.012"))
        )

    def decide(
        self,
        camera_id: str,
        tracked_persons: Any,
        *,
        frame_width: int,
        frame_height: int,
        restricted: bool,
        active_alert: bool,
        policy: CameraAIPolicy,
    ) -> BehaviourDecision:
        if not policy.behaviour:
            return BehaviourDecision(False, False, "policy_disabled", 0.0, False)
        count = len(tracked_persons) if tracked_persons is not None else 0
        if count <= 0:
            return BehaviourDecision(False, False, "no_person", 0.0, False)
        ids = getattr(tracked_persons, "tracker_id", None)
        boxes = getattr(tracked_persons, "xyxy", None)
        now = time.monotonic()
        diagonal = max(1.0, (frame_width ** 2 + frame_height ** 2) ** 0.5)
        max_motion = 0.0
        new_track = False
        if ids is not None and boxes is not None:
            with self._lock:
                for idx in range(count):
                    try:
                        tid = str(ids[idx])
                        x1, y1, x2, y2 = [float(v) for v in boxes[idx][:4]]
                    except Exception:
                        continue
                    centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    key = (str(camera_id), tid)
                    previous = self._centres.get(key)
                    if previous is None:
                        new_track = True
                    else:
                        dx = centre[0] - previous[0]
                        dy = centre[1] - previous[1]
                        max_motion = max(max_motion, (dx * dx + dy * dy) ** 0.5 / diagonal)
                    self._centres[key] = (centre[0], centre[1], now)
                stale = [key for key, value in self._centres.items() if now - value[2] > 20.0]
                for key in stale:
                    self._centres.pop(key, None)
        interaction = count >= 2
        strong_cue = bool(restricted or active_alert or interaction or new_track or max_motion >= self.motion_threshold)
        with self._lock:
            last = self._last_run.get(str(camera_id), 0.0)
            due = now - last >= self.stationary_refresh
            if strong_cue or due:
                self._last_run[str(camera_id)] = now
                run = True
            else:
                run = False
        run_pose = bool(policy.pose and run and (strong_cue or interaction))
        reason = (
            "active_alert" if active_alert else
            "restricted_zone" if restricted else
            "interaction" if interaction else
            "new_track" if new_track else
            "motion" if max_motion >= self.motion_threshold else
            "stationary_refresh" if run else
            "stationary_skip"
        )
        return BehaviourDecision(run, run_pose, reason, max_motion, interaction)

    def clear_camera(self, camera_id: str) -> None:
        camera_id = str(camera_id)
        with self._lock:
            self._last_run.pop(camera_id, None)
            for key in [key for key in self._centres if key[0] == camera_id]:
                self._centres.pop(key, None)


CAMERA_POLICIES = CameraPolicyStore()
BEHAVIOUR_GATE = BehaviourGate()
