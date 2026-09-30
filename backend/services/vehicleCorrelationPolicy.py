"""Deterministic evidence-fusion primitive used by acceptance tests and correlation policy."""
from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger("vehicle-correlation")


def _c(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(minimum, min(maximum, value))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evidence_fusion(
    *,
    anpr_score,
    reid_score,
    temporal_score,
    location_score,
    direction_score,
    attribute_score,
    track_quality,
    anpr_available,
    reid_available,
    metadata_available,
):
    """
    Preserve the existing deterministic INTEL-I vehicle evidence fusion.

    PostGIS is deliberately NOT mixed into this mathematical primitive.
    Geographic feasibility is a hard candidate gate that should be applied
    before evidence_fusion() for cross-camera candidates. Keeping the gate
    separate preserves acceptance-test compatibility and prevents an
    impossible journey from receiving a high score merely because ANPR/Re-ID
    evidence is strong.
    """
    branches = []
    scores = []

    if anpr_available:
        branches.append("anpr")
        scores.append(0.45 * _c(anpr_score))

    if reid_available:
        branches.append("reid")
        scores.append(0.30 * _c(reid_score))

    if metadata_available:
        branches.append("metadata")
        scores.append(
            0.25
            * (
                _c(temporal_score) * 0.35
                + _c(location_score) * 0.30
                + _c(direction_score) * 0.20
                + _c(attribute_score) * 0.15
            )
        )

    score = sum(scores) * _c(track_quality)

    return {
        "score": round(score, 6),
        "available_branches": branches,
        "anpr_score": _c(anpr_score),
        "reid_score": _c(reid_score),
        "track_quality": _c(track_quality),
    }


class VehiclePostGISGate:
    """
    PostGIS feasibility gate for cross-camera vehicle correlation.

    This class augments the existing evidence-fusion primitive without
    changing its public contract.

    Camera identifiers are resolved tenant-safely against both cameras.cam_id
    and cameras.id. Distance is obtained through the already validated
    PostGISSpatialService.

    Default rollout behavior is fail-open when PostGIS/camera geometry is
    unavailable so existing vehicle correlation is not silently disabled.
    Set VEHICLE_CORRELATION_POSTGIS_FAIL_CLOSED=true after production camera
    GIS coverage is complete if strict enforcement is desired.
    """

    def __init__(self) -> None:
        self.enabled = _env_bool(
            "VEHICLE_CORRELATION_POSTGIS_ENABLED",
            True,
        )
        self.fail_closed = _env_bool(
            "VEHICLE_CORRELATION_POSTGIS_FAIL_CLOSED",
            False,
        )
        self.require_location = _env_bool(
            "VEHICLE_CORRELATION_POSTGIS_REQUIRE_LOCATION",
            False,
        )
        self.max_distance_km = _env_float(
            "GIS_MAX_CAMERA_DISTANCE_KM",
            100.0,
            0.001,
            20000.0,
        )
        self.max_speed_kmh = _env_float(
            "GIS_MAX_REASONABLE_SPEED_KMH",
            180.0,
            0.001,
            5000.0,
        )
        self.max_transition_seconds = _env_float(
            "GIS_MAX_TRANSITION_SECONDS",
            3600.0,
            0.001,
            86400.0,
        )

    def evaluate(
        self,
        *,
        user_id: int,
        source_camera_id: str | int,
        destination_camera_id: str | int,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        if not self.enabled:
            return self._result(
                applicable=False,
                feasible=True,
                reason="postgis_disabled",
            )

        elapsed = _optional_float(elapsed_seconds)
        if elapsed is None:
            return self._result(
                applicable=False,
                feasible=not self.fail_closed,
                reason="invalid_elapsed_time",
            )

        if elapsed < 0.0:
            return self._result(
                applicable=True,
                feasible=False,
                reason="destination_precedes_source",
            )

        if elapsed > self.max_transition_seconds:
            return self._result(
                applicable=True,
                feasible=False,
                reason="transition_time_exceeded",
            )

        try:
            source_db_id = self._resolve_camera_db_id(
                user_id=int(user_id),
                camera_identifier=source_camera_id,
            )
            destination_db_id = self._resolve_camera_db_id(
                user_id=int(user_id),
                camera_identifier=destination_camera_id,
            )

            if source_db_id is None or destination_db_id is None:
                return self._result(
                    applicable=False,
                    feasible=not (
                        self.require_location or self.fail_closed
                    ),
                    reason="camera_not_resolved",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            if source_db_id == destination_db_id:
                return self._result(
                    applicable=True,
                    feasible=True,
                    distance_m=0.0,
                    required_speed_kmh=0.0,
                    reason="same_camera",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            distance_m = self._camera_distance_m(
                source_db_id,
                destination_db_id,
            )

            if distance_m is None:
                return self._result(
                    applicable=False,
                    feasible=not (
                        self.require_location or self.fail_closed
                    ),
                    reason="camera_location_unavailable",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            distance_km = distance_m / 1000.0

            if distance_km > self.max_distance_km:
                return self._result(
                    applicable=True,
                    feasible=False,
                    distance_m=distance_m,
                    reason="camera_distance_exceeded",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            if elapsed == 0.0:
                if distance_m <= 0.001:
                    required_speed = 0.0
                else:
                    return self._result(
                        applicable=True,
                        feasible=False,
                        distance_m=distance_m,
                        reason="zero_time_nonzero_distance",
                        source_camera_db_id=source_db_id,
                        destination_camera_db_id=destination_db_id,
                    )
            else:
                required_speed = distance_km / (elapsed / 3600.0)

            if required_speed > self.max_speed_kmh:
                return self._result(
                    applicable=True,
                    feasible=False,
                    distance_m=distance_m,
                    required_speed_kmh=required_speed,
                    reason="required_speed_exceeded",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            return self._result(
                applicable=True,
                feasible=True,
                distance_m=distance_m,
                required_speed_kmh=required_speed,
                reason="feasible",
                source_camera_db_id=source_db_id,
                destination_camera_db_id=destination_db_id,
            )

        except Exception:
            logger.exception(
                "Vehicle PostGIS gate failed for user=%s source_camera=%s "
                "destination_camera=%s",
                user_id,
                source_camera_id,
                destination_camera_id,
            )
            return self._result(
                applicable=False,
                feasible=not self.fail_closed,
                reason=(
                    "postgis_error_fail_closed"
                    if self.fail_closed
                    else "postgis_error_fail_open"
                ),
            )

    def candidate_allowed(
        self,
        *,
        user_id: int,
        source_camera_id: str | int,
        destination_camera_id: str | int,
        elapsed_seconds: float,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Convenience API for the vehicle candidate-selection pipeline.

        Usage:
            allowed, spatial = vehicle_postgis_gate.candidate_allowed(...)
            if not allowed:
                continue
            fused = evidence_fusion(...)
        """
        result = self.evaluate(
            user_id=user_id,
            source_camera_id=source_camera_id,
            destination_camera_id=destination_camera_id,
            elapsed_seconds=elapsed_seconds,
        )
        return bool(result["feasible"]), result

    @staticmethod
    def _resolve_camera_db_id(
        *,
        user_id: int,
        camera_identifier: str | int,
    ) -> int | None:
        identifier = str(camera_identifier or "").strip()
        if not identifier:
            return None

        from sqlalchemy import text
        from db.database import SessionLocal

        db = SessionLocal()
        try:
            row = db.execute(
                text(
                    """
                    SELECT id
                    FROM public.cameras
                    WHERE user_id = :user_id
                      AND (
                            cam_id = :camera_identifier
                            OR CAST(id AS TEXT) = :camera_identifier
                      )
                    ORDER BY
                        CASE
                            WHEN cam_id = :camera_identifier THEN 0
                            ELSE 1
                        END,
                        id ASC
                    LIMIT 1
                    """
                ),
                {
                    "user_id": int(user_id),
                    "camera_identifier": identifier,
                },
            ).first()

            if row is None or row[0] is None:
                return None
            return int(row[0])
        finally:
            db.close()

    @staticmethod
    def _camera_distance_m(
        source_camera_db_id: int,
        destination_camera_db_id: int,
    ) -> float | None:
        from db.database import SessionLocal
        from services.postgisSpatial import PostGISSpatialService

        db = SessionLocal()
        try:
            spatial = PostGISSpatialService(db)
            result = spatial.camera_distance(
                int(source_camera_db_id),
                int(destination_camera_db_id),
            )
            if result is None:
                return None
            return _optional_float(result.distance_m)
        finally:
            db.close()

    @staticmethod
    def _result(
        *,
        applicable: bool,
        feasible: bool,
        reason: str,
        distance_m: float | None = None,
        required_speed_kmh: float | None = None,
        source_camera_db_id: int | None = None,
        destination_camera_db_id: int | None = None,
    ) -> dict[str, Any]:
        distance = _optional_float(distance_m)
        speed = _optional_float(required_speed_kmh)

        return {
            "applicable": bool(applicable),
            "feasible": bool(feasible),
            "reason": str(reason),
            "distance_m": distance,
            "distance_km": (
                distance / 1000.0
                if distance is not None
                else None
            ),
            "required_speed_kmh": speed,
            "source_camera_db_id": source_camera_db_id,
            "destination_camera_db_id": destination_camera_db_id,
        }


vehicle_postgis_gate = VehiclePostGISGate()
