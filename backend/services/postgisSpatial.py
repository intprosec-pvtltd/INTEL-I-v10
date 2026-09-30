from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.orm import Session


logger = logging.getLogger("postgis-spatial")


# =============================================================================
# Configuration
# =============================================================================

EARTH_RADIUS_M = 6_371_008.8

DEFAULT_MAX_CAMERA_DISTANCE_KM = 100.0
DEFAULT_MAX_REASONABLE_SPEED_KMH = 180.0
DEFAULT_MAX_TRANSITION_SECONDS = 3600.0
DEFAULT_NEAREST_CAMERA_LIMIT = 25
DEFAULT_CANDIDATE_LIMIT = 100

MAX_QUERY_DISTANCE_KM = 20_000.0
MAX_RESULT_LIMIT = 1000

POSTGIS_SRID = 4326


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r; using default %s",
            name,
            raw,
            default,
        )
        return default

    if not math.isfinite(value):
        logger.warning(
            "Non-finite %s=%r; using default %s",
            name,
            raw,
            default,
        )
        return default

    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r; using default %s",
            name,
            raw,
            default,
        )
        return default

    return value


GIS_MAX_CAMERA_DISTANCE_KM = max(
    0.001,
    _env_float(
        "GIS_MAX_CAMERA_DISTANCE_KM",
        DEFAULT_MAX_CAMERA_DISTANCE_KM,
    ),
)

GIS_MAX_REASONABLE_SPEED_KMH = max(
    0.001,
    _env_float(
        "GIS_MAX_REASONABLE_SPEED_KMH",
        DEFAULT_MAX_REASONABLE_SPEED_KMH,
    ),
)

GIS_MAX_TRANSITION_SECONDS = max(
    0.001,
    _env_float(
        "GIS_MAX_TRANSITION_SECONDS",
        DEFAULT_MAX_TRANSITION_SECONDS,
    ),
)

GIS_NEAREST_CAMERA_LIMIT = min(
    MAX_RESULT_LIMIT,
    max(
        1,
        _env_int(
            "GIS_NEAREST_CAMERA_LIMIT",
            DEFAULT_NEAREST_CAMERA_LIMIT,
        ),
    ),
)

GIS_CANDIDATE_LIMIT = min(
    MAX_RESULT_LIMIT,
    max(
        1,
        _env_int(
            "GIS_CANDIDATE_LIMIT",
            DEFAULT_CANDIDATE_LIMIT,
        ),
    ),
)


# =============================================================================
# Exceptions
# =============================================================================


class SpatialError(RuntimeError):
    """Base exception for INTEL-I spatial-query failures."""


class SpatialValidationError(SpatialError):
    """Raised when invalid coordinates or query parameters are supplied."""


class SpatialUnavailableError(SpatialError):
    """Raised when PostGIS or the required schema is unavailable."""


# =============================================================================
# Data models
# =============================================================================


@dataclass(frozen=True, slots=True)
class SpatialReadiness:
    available: bool
    postgis_version: Optional[str]
    location_column: bool
    spatial_index: bool
    sync_trigger: bool
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "postgis_version": self.postgis_version,
            "location_column": self.location_column,
            "spatial_index": self.spatial_index,
            "sync_trigger": self.sync_trigger,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CameraSpatialResult:
    camera_id: int
    camera_name: Optional[str]
    latitude: float
    longitude: float
    distance_m: float

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "distance_m": self.distance_m,
            "distance_km": self.distance_km,
        }


@dataclass(frozen=True, slots=True)
class CameraDistance:
    source_camera_id: int
    destination_camera_id: int
    distance_m: float

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_camera_id": self.source_camera_id,
            "destination_camera_id": self.destination_camera_id,
            "distance_m": self.distance_m,
            "distance_km": self.distance_km,
        }


@dataclass(frozen=True, slots=True)
class TransitionFeasibility:
    source_camera_id: int
    destination_camera_id: int
    distance_m: Optional[float]
    elapsed_seconds: Optional[float]
    required_speed_kmh: Optional[float]
    feasible: bool
    reason: str

    @property
    def distance_km(self) -> Optional[float]:
        if self.distance_m is None:
            return None
        return self.distance_m / 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_camera_id": self.source_camera_id,
            "destination_camera_id": self.destination_camera_id,
            "distance_m": self.distance_m,
            "distance_km": self.distance_km,
            "elapsed_seconds": self.elapsed_seconds,
            "required_speed_kmh": self.required_speed_kmh,
            "feasible": self.feasible,
            "reason": self.reason,
        }


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_latitude(value: Any) -> float:
    try:
        latitude = float(value)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError("Latitude must be numeric.") from exc

    if not math.isfinite(latitude):
        raise SpatialValidationError("Latitude must be finite.")

    if latitude < -90.0 or latitude > 90.0:
        raise SpatialValidationError(
            "Latitude must be between -90 and 90 degrees."
        )

    return latitude


def _validate_longitude(value: Any) -> float:
    try:
        longitude = float(value)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError("Longitude must be numeric.") from exc

    if not math.isfinite(longitude):
        raise SpatialValidationError("Longitude must be finite.")

    if longitude < -180.0 or longitude > 180.0:
        raise SpatialValidationError(
            "Longitude must be between -180 and 180 degrees."
        )

    return longitude


def _validate_distance_km(value: Any) -> float:
    try:
        distance = float(value)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError(
            "Distance must be numeric."
        ) from exc

    if not math.isfinite(distance):
        raise SpatialValidationError(
            "Distance must be finite."
        )

    if distance <= 0.0:
        raise SpatialValidationError(
            "Distance must be greater than zero."
        )

    if distance > MAX_QUERY_DISTANCE_KM:
        raise SpatialValidationError(
            f"Distance exceeds maximum supported query radius "
            f"of {MAX_QUERY_DISTANCE_KM:g} km."
        )

    return distance


def _validate_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError(
            "Result limit must be an integer."
        ) from exc

    if limit < 1:
        raise SpatialValidationError(
            "Result limit must be at least 1."
        )

    return min(limit, MAX_RESULT_LIMIT)


def _validate_camera_id(value: Any) -> int:
    if isinstance(value, bool):
        raise SpatialValidationError(
            "Camera ID must be a positive integer."
        )

    try:
        camera_id = int(value)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError(
            "Camera ID must be a positive integer."
        ) from exc

    if camera_id <= 0:
        raise SpatialValidationError(
            "Camera ID must be a positive integer."
        )

    return camera_id


def _normalize_camera_ids(
    camera_ids: Optional[Sequence[int]],
) -> Optional[list[int]]:
    if camera_ids is None:
        return None

    normalized: list[int] = []
    seen: set[int] = set()

    for raw_id in camera_ids:
        camera_id = _validate_camera_id(raw_id)

        if camera_id not in seen:
            normalized.append(camera_id)
            seen.add(camera_id)

    if not normalized:
        return None

    return normalized


def _validate_timestamp(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise SpatialValidationError(
            f"{name} must be a datetime."
        )

    return value


# =============================================================================
# Fallback calculations
# =============================================================================


def haversine_distance_m(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    """
    Pure-Python WGS84-style great-circle fallback.

    This is intentionally retained as a validation/fallback helper.
    Production candidate filtering should use PostGIS.
    """

    lat1 = math.radians(_validate_latitude(latitude1))
    lon1 = math.radians(_validate_longitude(longitude1))
    lat2 = math.radians(_validate_latitude(latitude2))
    lon2 = math.radians(_validate_longitude(longitude2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    a = min(1.0, max(0.0, a))

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(1.0 - a),
    )

    return EARTH_RADIUS_M * c


def required_speed_kmh(
    distance_m: float,
    elapsed_seconds: float,
) -> Optional[float]:
    try:
        distance = float(distance_m)
        elapsed = float(elapsed_seconds)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError(
            "Distance and elapsed time must be numeric."
        ) from exc

    if not math.isfinite(distance) or distance < 0:
        raise SpatialValidationError(
            "Distance must be finite and non-negative."
        )

    if not math.isfinite(elapsed):
        raise SpatialValidationError(
            "Elapsed time must be finite."
        )

    if elapsed < 0:
        raise SpatialValidationError(
            "Elapsed time cannot be negative."
        )

    if elapsed == 0:
        if distance == 0:
            return 0.0
        return None

    return (distance / 1000.0) / (elapsed / 3600.0)


# =============================================================================
# SQL helpers
# =============================================================================


def _database_error(exc: Exception, operation: str) -> SpatialError:
    logger.exception(
        "PostGIS operation failed: %s",
        operation,
    )

    if isinstance(exc, DBAPIError):
        original = getattr(exc, "orig", None)

        if original is not None:
            message = str(original).lower()

            if (
                "postgis" in message
                or 'type "geometry" does not exist' in message
                or "function st_" in message
                or "extension postgis" in message
            ):
                return SpatialUnavailableError(
                    f"PostGIS spatial infrastructure is unavailable "
                    f"during {operation}."
                )

    return SpatialError(
        f"Spatial database operation failed during {operation}."
    )


def _camera_filter_sql(
    *,
    camera_ids: Optional[list[int]],
    exclude_camera_id: Optional[int],
    table_alias: str = "c",
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if camera_ids:
        placeholders: list[str] = []

        for index, camera_id in enumerate(camera_ids):
            key = f"camera_filter_{index}"
            placeholders.append(f":{key}")
            params[key] = camera_id

        clauses.append(
            f"{table_alias}.id IN ({', '.join(placeholders)})"
        )

    if exclude_camera_id is not None:
        clauses.append(
            f"{table_alias}.id <> :exclude_camera_id"
        )
        params["exclude_camera_id"] = exclude_camera_id

    if not clauses:
        return "", params

    return " AND " + " AND ".join(clauses), params


# =============================================================================
# PostGIS service
# =============================================================================


class PostGISSpatialService:
    """
    Production spatial-query layer for INTEL-I.

    PostGIS is authoritative for database-side candidate filtering and
    distance calculations. Existing latitude/longitude columns remain
    available for compatibility and fallback behavior.
    """

    def __init__(self, db: Session):
        if db is None:
            raise SpatialValidationError(
                "A SQLAlchemy database session is required."
            )

        self.db = db

    # ---------------------------------------------------------------------
    # Readiness
    # ---------------------------------------------------------------------

    def readiness(self) -> SpatialReadiness:
        postgis_version: Optional[str] = None
        location_column = False
        spatial_index = False
        sync_trigger = False

        try:
            postgis_version = self.db.execute(
                text(
                    """
                    SELECT PostGIS_Version()
                    """
                )
            ).scalar_one()

            location_column = bool(
                self.db.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                              AND table_name = 'cameras'
                              AND column_name = 'location'
                              AND udt_name = 'geometry'
                        )
                        """
                    )
                ).scalar_one()
            )

            spatial_index = bool(
                self.db.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_indexes
                            WHERE schemaname = 'public'
                              AND tablename = 'cameras'
                              AND indexname =
                                  'idx_cameras_location_gist'
                        )
                        """
                    )
                ).scalar_one()
            )

            sync_trigger = bool(
                self.db.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_trigger
                            WHERE tgrelid =
                                  'public.cameras'::regclass
                              AND tgname =
                                  'trg_cameras_sync_location'
                              AND NOT tgisinternal
                        )
                        """
                    )
                ).scalar_one()
            )

            available = bool(
                postgis_version
                and location_column
                and spatial_index
                and sync_trigger
            )

            error = None

            if not available:
                missing: list[str] = []

                if not postgis_version:
                    missing.append("PostGIS extension")

                if not location_column:
                    missing.append("cameras.location")

                if not spatial_index:
                    missing.append(
                        "idx_cameras_location_gist"
                    )

                if not sync_trigger:
                    missing.append(
                        "trg_cameras_sync_location"
                    )

                error = (
                    "Missing spatial infrastructure: "
                    + ", ".join(missing)
                )

            return SpatialReadiness(
                available=available,
                postgis_version=postgis_version,
                location_column=location_column,
                spatial_index=spatial_index,
                sync_trigger=sync_trigger,
                error=error,
            )

        except SQLAlchemyError as exc:
            logger.exception(
                "PostGIS readiness check failed."
            )

            return SpatialReadiness(
                available=False,
                postgis_version=postgis_version,
                location_column=location_column,
                spatial_index=spatial_index,
                sync_trigger=sync_trigger,
                error="PostGIS readiness check failed.",
            )

    def require_ready(self) -> SpatialReadiness:
        readiness = self.readiness()

        if not readiness.available:
            raise SpatialUnavailableError(
                readiness.error
                or "PostGIS spatial infrastructure is unavailable."
            )

        return readiness

    # ---------------------------------------------------------------------
    # Point distance
    # ---------------------------------------------------------------------

    def point_distance_m(
        self,
        *,
        latitude1: float,
        longitude1: float,
        latitude2: float,
        longitude2: float,
    ) -> float:
        lat1 = _validate_latitude(latitude1)
        lon1 = _validate_longitude(longitude1)
        lat2 = _validate_latitude(latitude2)
        lon2 = _validate_longitude(longitude2)

        try:
            value = self.db.execute(
                text(
                    """
                    SELECT ST_Distance(
                        ST_SetSRID(
                            ST_MakePoint(:lon1, :lat1),
                            4326
                        )::geography,
                        ST_SetSRID(
                            ST_MakePoint(:lon2, :lat2),
                            4326
                        )::geography
                    )
                    """
                ),
                {
                    "lat1": lat1,
                    "lon1": lon1,
                    "lat2": lat2,
                    "lon2": lon2,
                },
            ).scalar_one()

            return float(value)

        except SQLAlchemyError as exc:
            raise _database_error(
                exc,
                "point distance calculation",
            ) from exc

    # ---------------------------------------------------------------------
    # Camera lookup
    # ---------------------------------------------------------------------

    def camera_coordinates(
        self,
        camera_id: int,
    ) -> Optional[tuple[float, float]]:
        camera_id = _validate_camera_id(camera_id)

        try:
            row = self.db.execute(
                text(
                    """
                    SELECT
                        ST_Y(location) AS latitude,
                        ST_X(location) AS longitude
                    FROM public.cameras
                    WHERE id = :camera_id
                      AND location IS NOT NULL
                    """
                ),
                {"camera_id": camera_id},
            ).mappings().first()

            if row is None:
                return None

            return (
                float(row["latitude"]),
                float(row["longitude"]),
            )

        except SQLAlchemyError as exc:
            raise _database_error(
                exc,
                "camera coordinate lookup",
            ) from exc

    # ---------------------------------------------------------------------
    # Camera-to-camera distance
    # ---------------------------------------------------------------------

    def camera_distance(
        self,
        source_camera_id: int,
        destination_camera_id: int,
    ) -> Optional[CameraDistance]:
        source_id = _validate_camera_id(
            source_camera_id
        )
        destination_id = _validate_camera_id(
            destination_camera_id
        )

        if source_id == destination_id:
            coordinates = self.camera_coordinates(source_id)

            if coordinates is None:
                return None

            return CameraDistance(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=0.0,
            )

        try:
            row = self.db.execute(
                text(
                    """
                    SELECT
                        ST_Distance(
                            source.location::geography,
                            destination.location::geography
                        ) AS distance_m
                    FROM public.cameras AS source
                    JOIN public.cameras AS destination
                      ON destination.id = :destination_camera_id
                    WHERE source.id = :source_camera_id
                      AND source.location IS NOT NULL
                      AND destination.location IS NOT NULL
                    """
                ),
                {
                    "source_camera_id": source_id,
                    "destination_camera_id": destination_id,
                },
            ).mappings().first()

            if row is None:
                return None

            return CameraDistance(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=float(row["distance_m"]),
            )

        except SQLAlchemyError as exc:
            raise _database_error(
                exc,
                "camera distance calculation",
            ) from exc

    # ---------------------------------------------------------------------
    # Nearby cameras
    # ---------------------------------------------------------------------

    def nearby_cameras(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: float = GIS_MAX_CAMERA_DISTANCE_KM,
        limit: int = GIS_CANDIDATE_LIMIT,
        camera_ids: Optional[Sequence[int]] = None,
        exclude_camera_id: Optional[int] = None,
    ) -> list[CameraSpatialResult]:
        latitude = _validate_latitude(latitude)
        longitude = _validate_longitude(longitude)
        radius_km = _validate_distance_km(radius_km)
        limit = _validate_limit(limit)

        normalized_ids = _normalize_camera_ids(camera_ids)

        excluded_id: Optional[int] = None

        if exclude_camera_id is not None:
            excluded_id = _validate_camera_id(
                exclude_camera_id
            )

        extra_filter, filter_params = _camera_filter_sql(
            camera_ids=normalized_ids,
            exclude_camera_id=excluded_id,
            table_alias="c",
        )

        radius_m = radius_km * 1000.0

        query = f"""
            WITH query_point AS (
                SELECT ST_SetSRID(
                    ST_MakePoint(:longitude, :latitude),
                    4326
                )::geography AS geog
            )
            SELECT
                c.id AS camera_id,
                c.camera_name AS camera_name,
                ST_Y(c.location) AS latitude,
                ST_X(c.location) AS longitude,
                ST_Distance(
                    c.location::geography,
                    qp.geog
                ) AS distance_m
            FROM public.cameras AS c
            CROSS JOIN query_point AS qp
            WHERE c.location IS NOT NULL
              AND ST_DWithin(
                    c.location::geography,
                    qp.geog,
                    :radius_m
              )
              {extra_filter}
            ORDER BY
                distance_m ASC,
                c.id ASC
            LIMIT :result_limit
        """

        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "radius_m": radius_m,
            "result_limit": limit,
        }

        params.update(filter_params)

        try:
            rows = self.db.execute(
                text(query),
                params,
            ).mappings().all()

            return [
                CameraSpatialResult(
                    camera_id=int(row["camera_id"]),
                    camera_name=(
                        str(row["camera_name"])
                        if row["camera_name"] is not None
                        else None
                    ),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    distance_m=float(row["distance_m"]),
                )
                for row in rows
            ]

        except SQLAlchemyError as exc:
            raise _database_error(
                exc,
                "nearby camera query",
            ) from exc

    # ---------------------------------------------------------------------
    # Nearest cameras
    # ---------------------------------------------------------------------

    def nearest_cameras(
        self,
        *,
        latitude: float,
        longitude: float,
        limit: int = GIS_NEAREST_CAMERA_LIMIT,
        max_distance_km: Optional[float] = None,
        camera_ids: Optional[Sequence[int]] = None,
        exclude_camera_id: Optional[int] = None,
    ) -> list[CameraSpatialResult]:
        latitude = _validate_latitude(latitude)
        longitude = _validate_longitude(longitude)
        limit = _validate_limit(limit)

        if max_distance_km is not None:
            return self.nearby_cameras(
                latitude=latitude,
                longitude=longitude,
                radius_km=max_distance_km,
                limit=limit,
                camera_ids=camera_ids,
                exclude_camera_id=exclude_camera_id,
            )

        normalized_ids = _normalize_camera_ids(camera_ids)

        excluded_id: Optional[int] = None

        if exclude_camera_id is not None:
            excluded_id = _validate_camera_id(
                exclude_camera_id
            )

        extra_filter, filter_params = _camera_filter_sql(
            camera_ids=normalized_ids,
            exclude_camera_id=excluded_id,
            table_alias="c",
        )

        query = f"""
            WITH query_point AS (
                SELECT ST_SetSRID(
                    ST_MakePoint(:longitude, :latitude),
                    4326
                ) AS geom
            )
            SELECT
                c.id AS camera_id,
                c.camera_name AS camera_name,
                ST_Y(c.location) AS latitude,
                ST_X(c.location) AS longitude,
                ST_Distance(
                    c.location::geography,
                    qp.geom::geography
                ) AS distance_m
            FROM public.cameras AS c
            CROSS JOIN query_point AS qp
            WHERE c.location IS NOT NULL
              {extra_filter}
            ORDER BY
                c.location <-> qp.geom,
                c.id ASC
            LIMIT :result_limit
        """

        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "result_limit": limit,
        }

        params.update(filter_params)

        try:
            rows = self.db.execute(
                text(query),
                params,
            ).mappings().all()

            return [
                CameraSpatialResult(
                    camera_id=int(row["camera_id"]),
                    camera_name=(
                        str(row["camera_name"])
                        if row["camera_name"] is not None
                        else None
                    ),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    distance_m=float(row["distance_m"]),
                )
                for row in rows
            ]

        except SQLAlchemyError as exc:
            raise _database_error(
                exc,
                "nearest camera query",
            ) from exc

    # ---------------------------------------------------------------------
    # Nearby cameras from another camera
    # ---------------------------------------------------------------------

    def nearby_cameras_from_camera(
        self,
        source_camera_id: int,
        *,
        radius_km: float = GIS_MAX_CAMERA_DISTANCE_KM,
        limit: int = GIS_CANDIDATE_LIMIT,
        camera_ids: Optional[Sequence[int]] = None,
        include_source: bool = False,
    ) -> list[CameraSpatialResult]:
        source_id = _validate_camera_id(
            source_camera_id
        )

        coordinates = self.camera_coordinates(
            source_id
        )

        if coordinates is None:
            return []

        latitude, longitude = coordinates

        return self.nearby_cameras(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            limit=limit,
            camera_ids=camera_ids,
            exclude_camera_id=(
                None if include_source else source_id
            ),
        )

    # ---------------------------------------------------------------------
    # Transition feasibility
    # ---------------------------------------------------------------------

    def transition_feasibility(
        self,
        *,
        source_camera_id: int,
        destination_camera_id: int,
        source_timestamp: datetime,
        destination_timestamp: datetime,
        max_speed_kmh: float = GIS_MAX_REASONABLE_SPEED_KMH,
        max_transition_seconds: float = GIS_MAX_TRANSITION_SECONDS,
        max_distance_km: float = GIS_MAX_CAMERA_DISTANCE_KM,
    ) -> TransitionFeasibility:
        source_id = _validate_camera_id(
            source_camera_id
        )
        destination_id = _validate_camera_id(
            destination_camera_id
        )

        source_time = _validate_timestamp(
            source_timestamp,
            "source_timestamp",
        )
        destination_time = _validate_timestamp(
            destination_timestamp,
            "destination_timestamp",
        )

        max_speed = _validate_positive_finite(
            max_speed_kmh,
            "max_speed_kmh",
        )

        max_transition = _validate_positive_finite(
            max_transition_seconds,
            "max_transition_seconds",
        )

        max_distance = _validate_distance_km(
            max_distance_km
        )

        elapsed_seconds = (
            destination_time - source_time
        ).total_seconds()

        if elapsed_seconds < 0:
            return TransitionFeasibility(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=None,
                elapsed_seconds=elapsed_seconds,
                required_speed_kmh=None,
                feasible=False,
                reason="destination_precedes_source",
            )

        if elapsed_seconds > max_transition:
            return TransitionFeasibility(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=None,
                elapsed_seconds=elapsed_seconds,
                required_speed_kmh=None,
                feasible=False,
                reason="transition_time_exceeded",
            )

        distance = self.camera_distance(
            source_id,
            destination_id,
        )

        if distance is None:
            return TransitionFeasibility(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=None,
                elapsed_seconds=elapsed_seconds,
                required_speed_kmh=None,
                feasible=False,
                reason="camera_location_unavailable",
            )

        if distance.distance_km > max_distance:
            return TransitionFeasibility(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=distance.distance_m,
                elapsed_seconds=elapsed_seconds,
                required_speed_kmh=None,
                feasible=False,
                reason="camera_distance_exceeded",
            )

        speed = required_speed_kmh(
            distance.distance_m,
            elapsed_seconds,
        )

        if speed is None:
            return TransitionFeasibility(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=distance.distance_m,
                elapsed_seconds=elapsed_seconds,
                required_speed_kmh=None,
                feasible=False,
                reason="zero_time_nonzero_distance",
            )

        if speed > max_speed:
            return TransitionFeasibility(
                source_camera_id=source_id,
                destination_camera_id=destination_id,
                distance_m=distance.distance_m,
                elapsed_seconds=elapsed_seconds,
                required_speed_kmh=speed,
                feasible=False,
                reason="required_speed_exceeded",
            )

        return TransitionFeasibility(
            source_camera_id=source_id,
            destination_camera_id=destination_id,
            distance_m=distance.distance_m,
            elapsed_seconds=elapsed_seconds,
            required_speed_kmh=speed,
            feasible=True,
            reason="feasible",
        )

    # ---------------------------------------------------------------------
    # Candidate camera filtering
    # ---------------------------------------------------------------------

    def feasible_destination_cameras(
        self,
        *,
        source_camera_id: int,
        source_timestamp: datetime,
        destination_timestamp: datetime,
        candidate_camera_ids: Optional[Sequence[int]] = None,
        max_speed_kmh: float = GIS_MAX_REASONABLE_SPEED_KMH,
        max_transition_seconds: float = GIS_MAX_TRANSITION_SECONDS,
        max_distance_km: float = GIS_MAX_CAMERA_DISTANCE_KM,
        limit: int = GIS_CANDIDATE_LIMIT,
    ) -> list[TransitionFeasibility]:
        """
        Filter destination cameras for person/vehicle correlation.

        This method first uses PostGIS ST_DWithin to reduce the candidate
        set and then applies temporal/speed feasibility checks.

        This prevents expensive Re-ID/correlation work from being performed
        against geographically impossible cameras.
        """

        source_id = _validate_camera_id(
            source_camera_id
        )

        source_time = _validate_timestamp(
            source_timestamp,
            "source_timestamp",
        )

        destination_time = _validate_timestamp(
            destination_timestamp,
            "destination_timestamp",
        )

        limit = _validate_limit(limit)

        max_speed = _validate_positive_finite(
            max_speed_kmh,
            "max_speed_kmh",
        )

        max_transition = _validate_positive_finite(
            max_transition_seconds,
            "max_transition_seconds",
        )

        max_distance = _validate_distance_km(
            max_distance_km
        )

        elapsed_seconds = (
            destination_time - source_time
        ).total_seconds()

        if elapsed_seconds < 0:
            return []

        if elapsed_seconds > max_transition:
            return []

        source_coordinates = self.camera_coordinates(
            source_id
        )

        if source_coordinates is None:
            return []

        source_latitude, source_longitude = (
            source_coordinates
        )

        # Physical speed limit can produce a tighter radius than the
        # configured global camera-distance limit.
        speed_limited_distance_km = (
            max_speed * (elapsed_seconds / 3600.0)
        )

        effective_radius_km = min(
            max_distance,
            max(
                0.001,
                speed_limited_distance_km,
            ),
        )

        nearby = self.nearby_cameras(
            latitude=source_latitude,
            longitude=source_longitude,
            radius_km=effective_radius_km,
            limit=limit,
            camera_ids=candidate_camera_ids,
            exclude_camera_id=None,
        )

        results: list[TransitionFeasibility] = []

        for candidate in nearby:
            speed = required_speed_kmh(
                candidate.distance_m,
                elapsed_seconds,
            )

            if speed is None:
                feasible = (
                    candidate.distance_m == 0.0
                )

                reason = (
                    "feasible"
                    if feasible
                    else "zero_time_nonzero_distance"
                )

            elif speed > max_speed:
                feasible = False
                reason = "required_speed_exceeded"

            else:
                feasible = True
                reason = "feasible"

            results.append(
                TransitionFeasibility(
                    source_camera_id=source_id,
                    destination_camera_id=(
                        candidate.camera_id
                    ),
                    distance_m=candidate.distance_m,
                    elapsed_seconds=elapsed_seconds,
                    required_speed_kmh=speed,
                    feasible=feasible,
                    reason=reason,
                )
            )

        results.sort(
            key=lambda item: (
                not item.feasible,
                (
                    item.distance_m
                    if item.distance_m is not None
                    else float("inf")
                ),
                item.destination_camera_id,
            )
        )

        return results


# =============================================================================
# Additional validation helper
# =============================================================================


def _validate_positive_finite(
    value: Any,
    name: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(result):
        raise SpatialValidationError(
            f"{name} must be finite."
        )

    if result <= 0:
        raise SpatialValidationError(
            f"{name} must be greater than zero."
        )

    return result


# =============================================================================
# Convenience API
# =============================================================================


def get_postgis_spatial_service(
    db: Session,
) -> PostGISSpatialService:
    return PostGISSpatialService(db)


def postgis_readiness(
    db: Session,
) -> dict[str, Any]:
    return PostGISSpatialService(
        db
    ).readiness().as_dict()


def nearby_cameras(
    db: Session,
    *,
    latitude: float,
    longitude: float,
    radius_km: float = GIS_MAX_CAMERA_DISTANCE_KM,
    limit: int = GIS_CANDIDATE_LIMIT,
    camera_ids: Optional[Sequence[int]] = None,
    exclude_camera_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    results = PostGISSpatialService(
        db
    ).nearby_cameras(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        limit=limit,
        camera_ids=camera_ids,
        exclude_camera_id=exclude_camera_id,
    )

    return [
        result.as_dict()
        for result in results
    ]


def nearest_cameras(
    db: Session,
    *,
    latitude: float,
    longitude: float,
    limit: int = GIS_NEAREST_CAMERA_LIMIT,
    max_distance_km: Optional[float] = None,
    camera_ids: Optional[Sequence[int]] = None,
    exclude_camera_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    results = PostGISSpatialService(
        db
    ).nearest_cameras(
        latitude=latitude,
        longitude=longitude,
        limit=limit,
        max_distance_km=max_distance_km,
        camera_ids=camera_ids,
        exclude_camera_id=exclude_camera_id,
    )

    return [
        result.as_dict()
        for result in results
    ]


def camera_distance(
    db: Session,
    source_camera_id: int,
    destination_camera_id: int,
) -> Optional[dict[str, Any]]:
    result = PostGISSpatialService(
        db
    ).camera_distance(
        source_camera_id,
        destination_camera_id,
    )

    if result is None:
        return None

    return result.as_dict()


def transition_feasibility(
    db: Session,
    *,
    source_camera_id: int,
    destination_camera_id: int,
    source_timestamp: datetime,
    destination_timestamp: datetime,
    max_speed_kmh: float = GIS_MAX_REASONABLE_SPEED_KMH,
    max_transition_seconds: float = GIS_MAX_TRANSITION_SECONDS,
    max_distance_km: float = GIS_MAX_CAMERA_DISTANCE_KM,
) -> dict[str, Any]:
    result = PostGISSpatialService(
        db
    ).transition_feasibility(
        source_camera_id=source_camera_id,
        destination_camera_id=destination_camera_id,
        source_timestamp=source_timestamp,
        destination_timestamp=destination_timestamp,
        max_speed_kmh=max_speed_kmh,
        max_transition_seconds=max_transition_seconds,
        max_distance_km=max_distance_km,
    )

    return result.as_dict()