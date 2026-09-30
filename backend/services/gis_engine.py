from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional

EARTH_RADIUS_METERS = 6_371_008.8
DEFAULT_MIN_SPEED_KMH = 2.0
DEFAULT_MAX_SPEED_KMH = 180.0
MIN_MEANINGFUL_DISTANCE_METERS = 5.0
DIRECTION_FULL_MATCH_DEG = 20.0
DIRECTION_PARTIAL_MATCH_DEG = 90.0
DIRECTION_OPPOSITE_DEG = 150.0
TRAVEL_TIME_TOLERANCE_RATIO = 0.35
MIN_TRAVEL_SECONDS = 0.25
MAX_TRAVEL_SECONDS = 15 * 60
DISTANCE_WEIGHT = 0.20
TRAVEL_TIME_WEIGHT = 0.45
DIRECTION_WEIGHT = 0.35


@dataclass(frozen=True)
class CameraPoint:
    camera_id: str
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    heading: Optional[float] = None
    fov: Optional[float] = None
    road_name: Optional[str] = None
    location_name: Optional[str] = None


@dataclass(frozen=True)
class GISScore:
    available: bool
    distance_meters: Optional[float]
    bearing_degrees: Optional[float]
    elapsed_seconds: Optional[float]
    implied_speed_kmh: Optional[float]
    travel_time_score: float
    direction_score: float
    distance_score: float
    gis_score: float
    feasible: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "gis_available": self.available,
            "distance_meters": self.distance_meters,
            "bearing_degrees": self.bearing_degrees,
            "elapsed_seconds": self.elapsed_seconds,
            "implied_speed_kmh": self.implied_speed_kmh,
            "travel_time_score": self.travel_time_score,
            "direction_score": self.direction_score,
            "distance_score": self.distance_score,
            "gis_score": self.gis_score,
            "feasible": self.feasible,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------
# Generic safety helpers
# ---------------------------------------------------------------------

def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _clamp01(value: Any) -> float:
    result = _finite_float(value)
    if result is None:
        return 0.0
    return max(0.0, min(1.0, result))


def _normalize_heading(degrees: Any) -> Optional[float]:
    value = _finite_float(degrees)
    if value is None:
        return None
    return value % 360.0


def _angular_difference(a: float, b: float) -> float:
    """
    Smallest absolute difference between two compass bearings.
    Result is always in [0, 180].
    """
    a = a % 360.0
    b = b % 360.0
    return abs((a - b + 180.0) % 360.0 - 180.0)



def _local_xy_to_latlon(
    x_m: float,
    y_m: float,
    origin_lat: float,
    origin_lon: float,
) -> Optional[tuple[float, float]]:
    """Convert a local tangent/equirectangular ground coordinate to WGS84."""
    try:
        meters_per_lat = 111_320.0
        meters_per_lon = meters_per_lat * max(0.1, math.cos(math.radians(origin_lat)))
        lat = origin_lat + float(y_m) / meters_per_lat
        lon = origin_lon + float(x_m) / meters_per_lon
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon
    except Exception:
        return None


def project_pixel_to_geo(
    pixel_x: float,
    pixel_y: float,
    homography: Any = None,
    calibration: Optional[dict[str, Any]] = None,
) -> Optional[tuple[float, float]]:
    """Project a processed-frame pixel to latitude/longitude.

    Production calibration format (version 2) stores the homography in a
    locally scaled metric ground plane. This is numerically more stable than
    fitting a homography directly against raw degree-valued lat/lon. Version 1
    direct lat/lon matrices remain supported for backward compatibility.
    """
    try:
        import numpy as np

        if calibration is not None and isinstance(calibration, dict):
            version = int(calibration.get("version", 1) or 1)
            matrix_value = calibration.get("homography_xy") or calibration.get("homography")
            matrix = np.asarray(matrix_value, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                return None

            x = float(pixel_x)
            y = float(pixel_y)
            if not (math.isfinite(x) and math.isfinite(y)):
                return None

            q = matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
            if not np.isfinite(q).all() or abs(float(q[2])) < 1e-12:
                return None

            projected_x = float(q[0] / q[2])
            projected_y = float(q[1] / q[2])

            if version >= 2 and calibration.get("origin"):
                origin = calibration.get("origin") or {}
                origin_lat = float(origin.get("latitude"))
                origin_lon = float(origin.get("longitude"))
                return _local_xy_to_latlon(
                    projected_x, projected_y, origin_lat, origin_lon
                )

            lon = projected_x
            lat = projected_y
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                return None
            return lat, lon

        matrix = np.asarray(homography, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return None
        x = float(pixel_x)
        y = float(pixel_y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        q = matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
        if not np.isfinite(q).all() or abs(float(q[2])) < 1e-12:
            return None
        lon = float(q[0] / q[2])
        lat = float(q[1] / q[2])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon
    except Exception:
        return None


# ---------------------------------------------------------------------
# Camera metadata validation
# ---------------------------------------------------------------------

def validate_camera_point(
    camera_id: Any,
    latitude: Any,
    longitude: Any,
    *,
    altitude: Any = None,
    heading: Any = None,
    fov: Any = None,
    road_name: Any = None,
    location_name: Any = None,
) -> Optional[CameraPoint]:
    """
    Validate and normalize camera GIS metadata.

    Returns None for invalid coordinates.
    """
    if not isinstance(camera_id, str):
        return None

    camera_id = camera_id.strip()

    if not camera_id or len(camera_id) > 128:
        return None

    if any(ord(char) < 32 for char in camera_id):
        return None

    lat = _finite_float(latitude)
    lon = _finite_float(longitude)

    if lat is None or lon is None:
        return None

    if not -90.0 <= lat <= 90.0:
        return None

    if not -180.0 <= lon <= 180.0:
        return None

    alt = _finite_float(altitude)
    normalized_heading = _normalize_heading(heading)
    normalized_fov = _finite_float(fov)

    if normalized_fov is not None:
        normalized_fov = max(0.0, min(360.0, normalized_fov))

    return CameraPoint(
        camera_id=camera_id,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        heading=normalized_heading,
        fov=normalized_fov,
        road_name=(
            road_name.strip()
            if isinstance(road_name, str) and road_name.strip()
            else None
        ),
        location_name=(
            location_name.strip()
            if isinstance(location_name, str) and location_name.strip()
            else None
        ),
    )


def camera_from_dict(camera_id: str, metadata: Any) -> Optional[CameraPoint]:
    if not isinstance(metadata, dict):
        return None

    return validate_camera_point(
        camera_id=camera_id,
        latitude=metadata.get("latitude"),
        longitude=metadata.get("longitude"),
        altitude=metadata.get("altitude"),
        heading=metadata.get("heading"),
        fov=metadata.get("fov"),
        road_name=metadata.get("road_name"),
        location_name=metadata.get("location_name"),
    )


# ---------------------------------------------------------------------
# Distance / bearing
# ---------------------------------------------------------------------

def haversine_distance_meters(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> Optional[float]:
    """Return great-circle WGS84 distance in meters."""
    lat1 = _finite_float(latitude1)
    lon1 = _finite_float(longitude1)
    lat2 = _finite_float(latitude2)
    lon2 = _finite_float(longitude2)

    if None in (lat1, lon1, lat2, lon2):
        return None

    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return None

    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        return None

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(longitude2 - longitude1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(d_lambda / 2.0) ** 2
    )

    a = max(0.0, min(1.0, a))

    return 2.0 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


def initial_bearing_degrees(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> Optional[float]:
    """
    Return the initial great-circle bearing from point 1 to point 2.

    0 = North, 90 = East, 180 = South, 270 = West.
    """
    lat1 = _finite_float(latitude1)
    lon1 = _finite_float(longitude1)
    lat2 = _finite_float(latitude2)
    lon2 = _finite_float(longitude2)

    if None in (lat1, lon1, lat2, lon2):
        return None

    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return None

    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        return None

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    x = math.sin(delta_lambda) * math.cos(phi2)

    y = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1)
        * math.cos(phi2)
        * math.cos(delta_lambda)
    )

    if abs(x) < 1e-15 and abs(y) < 1e-15:
        return None

    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360.0


# ---------------------------------------------------------------------
# Travel-time / direction scoring
# ---------------------------------------------------------------------

def implied_speed_kmh(
    distance_meters: Any,
    elapsed_seconds: Any,
) -> Optional[float]:
    distance = _finite_float(distance_meters)
    elapsed = _finite_float(elapsed_seconds)

    if distance is None or elapsed is None:
        return None

    if elapsed <= 0:
        return None

    return (distance / 1000.0) / (elapsed / 3600.0)


def direction_score(
    observed_heading: Any,
    expected_bearing: Any,
) -> float:
    """
    Convert heading-vs-route bearing into [0,1].

    Unknown heading -> neutral 0.5 rather than false confidence.
    """
    observed = _normalize_heading(observed_heading)
    expected = _normalize_heading(expected_bearing)

    if observed is None or expected is None:
        return 0.5

    difference = _angular_difference(observed, expected)

    if difference <= DIRECTION_FULL_MATCH_DEG:
        return 1.0

    if difference >= DIRECTION_OPPOSITE_DEG:
        return 0.0

    if difference >= DIRECTION_PARTIAL_MATCH_DEG:
        # 90 -> 0.25, 150 -> 0.0
        span = DIRECTION_OPPOSITE_DEG - DIRECTION_PARTIAL_MATCH_DEG
        return max(
            0.0,
            0.25 * (DIRECTION_OPPOSITE_DEG - difference) / span,
        )

    # Linear decay from 1.0 at 20 degrees to 0.25 at 90 degrees.
    return max(
        0.25,
        1.0
        - 0.75
        * (
            difference - DIRECTION_FULL_MATCH_DEG
        )
        / (
            DIRECTION_PARTIAL_MATCH_DEG
            - DIRECTION_FULL_MATCH_DEG
        ),
    )


def distance_score(distance_meters: Any) -> float:
    """
    Distance itself is not identity evidence.

    A bounded score is used only as a weak GIS component. Very close
    cameras are neutral because their spatial separation contributes
    little information.
    """
    distance = _finite_float(distance_meters)

    if distance is None:
        return 0.0

    if distance <= MIN_MEANINGFUL_DISTANCE_METERS:
        return 0.5

    # Soft decay; 1 km still has useful geographic relationship,
    # while very distant cameras receive progressively less support.
    return max(
        0.0,
        min(
            1.0,
            math.exp(-distance / 5000.0),
        ),
    )


def travel_time_score(
    distance_meters: Any,
    elapsed_seconds: Any,
    *,
    min_speed_kmh: float = DEFAULT_MIN_SPEED_KMH,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
) -> tuple[float, bool, Optional[float], str]:
    """
    Evaluate whether the observed camera transition is physically
    plausible.

    This is a plausibility check, not a road-routing engine.
    """
    distance = _finite_float(distance_meters)
    elapsed = _finite_float(elapsed_seconds)

    if distance is None or elapsed is None:
        return 0.0, False, None, "missing_distance_or_time"

    if elapsed < MIN_TRAVEL_SECONDS:
        return 0.0, False, None, "non_positive_travel_time"

    if elapsed > MAX_TRAVEL_SECONDS:
        return 0.0, False, None, "travel_time_window_exceeded"

    speed = implied_speed_kmh(distance, elapsed)

    if speed is None:
        return 0.0, False, None, "unable_to_calculate_speed"

    min_speed = max(0.0, float(min_speed_kmh))
    max_speed = max(min_speed + 1.0, float(max_speed_kmh))

    if distance <= MIN_MEANINGFUL_DISTANCE_METERS:
        return 0.5, True, speed, "nearby_cameras"

    if speed > max_speed:
        return 0.0, False, speed, "impossible_implied_speed"

    if speed < min_speed:
        # A low speed can still be legitimate because a vehicle may
        # stop between cameras. Penalize instead of hard rejecting.
        penalty_span = max(min_speed, 1.0)
        score = max(
            0.15,
            min(0.5, speed / penalty_span),
        )
        return score, True, speed, "slow_but_plausible"

    # Inside the practical speed envelope.
    midpoint = (min_speed + max_speed) / 2.0
    if speed <= midpoint:
        score = 0.85 + 0.15 * (
            speed - min_speed
        ) / max(midpoint - min_speed, 1.0)
    else:
        score = 1.0 - 0.15 * (
            speed - midpoint
        ) / max(max_speed - midpoint, 1.0)

    return _clamp01(score), True, speed, "travel_time_plausible"


# ---------------------------------------------------------------------
# Camera-to-camera evaluation
# ---------------------------------------------------------------------

def evaluate_camera_transition(
    source_camera: CameraPoint,
    destination_camera: CameraPoint,
    source_timestamp: Any,
    destination_timestamp: Any,
    *,
    source_heading: Any = None,
    destination_heading: Any = None,
    min_speed_kmh: float = DEFAULT_MIN_SPEED_KMH,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
) -> GISScore:
    """
    Evaluate a possible vehicle transition between two cameras.
    """
    distance = haversine_distance_meters(
        source_camera.latitude,
        source_camera.longitude,
        destination_camera.latitude,
        destination_camera.longitude,
    )

    bearing = initial_bearing_degrees(
        source_camera.latitude,
        source_camera.longitude,
        destination_camera.latitude,
        destination_camera.longitude,
    )

    source_ts = _finite_float(source_timestamp)
    destination_ts = _finite_float(destination_timestamp)

    if distance is None or bearing is None:
        return GISScore(
            available=False,
            distance_meters=distance,
            bearing_degrees=bearing,
            elapsed_seconds=None,
            implied_speed_kmh=None,
            travel_time_score=0.0,
            direction_score=0.5,
            distance_score=0.0,
            gis_score=0.0,
            feasible=False,
            reason="invalid_camera_geometry",
        )

    if source_ts is None or destination_ts is None:
        return GISScore(
            available=True,
            distance_meters=distance,
            bearing_degrees=bearing,
            elapsed_seconds=None,
            implied_speed_kmh=None,
            travel_time_score=0.5,
            direction_score=direction_score(
                destination_heading
                if destination_heading is not None
                else source_heading,
                bearing,
            ),
            distance_score=distance_score(distance),
            gis_score=0.5,
            feasible=True,
            reason="geometry_available_time_missing",
        )

    elapsed = destination_ts - source_ts

    if elapsed < 0:
        return GISScore(
            available=True,
            distance_meters=distance,
            bearing_degrees=bearing,
            elapsed_seconds=elapsed,
            implied_speed_kmh=None,
            travel_time_score=0.0,
            direction_score=0.0,
            distance_score=distance_score(distance),
            gis_score=0.0,
            feasible=False,
            reason="out_of_order_timestamps",
        )

    observed_heading = (
        destination_heading
        if destination_heading is not None
        else source_heading
    )

    dir_score = direction_score(
        observed_heading,
        bearing,
    )

    time_score, feasible, speed, reason = travel_time_score(
        distance,
        elapsed,
        min_speed_kmh=min_speed_kmh,
        max_speed_kmh=max_speed_kmh,
    )

    dist_score = distance_score(distance)

    # Do not let unknown heading become an unjustified positive signal.
    # 0.5 is neutral.
    total = (
        DISTANCE_WEIGHT * dist_score
        + TRAVEL_TIME_WEIGHT * time_score
        + DIRECTION_WEIGHT * dir_score
    )

    # Impossible travel should never become a positive match.
    if not feasible and reason == "impossible_implied_speed":
        total = 0.0

    return GISScore(
        available=True,
        distance_meters=distance,
        bearing_degrees=bearing,
        elapsed_seconds=elapsed,
        implied_speed_kmh=speed,
        travel_time_score=_clamp01(time_score),
        direction_score=_clamp01(dir_score),
        distance_score=_clamp01(dist_score),
        gis_score=_clamp01(total),
        feasible=feasible,
        reason=reason,
    )


# ---------------------------------------------------------------------
# Vehicle-record helpers
# ---------------------------------------------------------------------

def _camera_from_vehicle_record(
    vehicle: dict[str, Any],
    metadata_loader=None,
) -> Optional[CameraPoint]:
    """
    Resolve GIS from a vehicle record.

    Supported vehicle formats:
      vehicle["camera_gis"] = {"latitude": ..., "longitude": ...}
      vehicle["gis"]        = {"latitude": ..., "longitude": ...}

    If metadata_loader is supplied it is called as:
      metadata_loader(camera_id) -> dict
    """
    if not isinstance(vehicle, dict):
        return None

    camera_id = vehicle.get("camera_id")
    if not isinstance(camera_id, str):
        return None

    metadata = vehicle.get("camera_gis")

    if not isinstance(metadata, dict):
        metadata = vehicle.get("gis")

    if not isinstance(metadata, dict) and metadata_loader is not None:
        try:
            metadata = metadata_loader(camera_id)
        except Exception:
            metadata = None

    return camera_from_dict(camera_id, metadata)


def evaluate_vehicle_transition(
    source_vehicle: dict[str, Any],
    destination_vehicle: dict[str, Any],
    *,
    metadata_loader=None,
) -> GISScore:
    """
    Convenience wrapper for vehicleCorrelation.py.

    Expected fields:
      camera_id
      timestamp

    Optional:
      heading
      direction
      camera_gis / gis
    """
    source_camera = _camera_from_vehicle_record(
        source_vehicle,
        metadata_loader=metadata_loader,
    )
    destination_camera = _camera_from_vehicle_record(
        destination_vehicle,
        metadata_loader=metadata_loader,
    )

    if source_camera is None or destination_camera is None:
        return GISScore(
            available=False,
            distance_meters=None,
            bearing_degrees=None,
            elapsed_seconds=None,
            implied_speed_kmh=None,
            travel_time_score=0.0,
            direction_score=0.5,
            distance_score=0.0,
            gis_score=0.0,
            feasible=False,
            reason="camera_gis_unavailable",
        )

    source_heading = source_vehicle.get(
        "heading",
        source_vehicle.get("direction"),
    )
    destination_heading = destination_vehicle.get(
        "heading",
        destination_vehicle.get("direction"),
    )

    return evaluate_camera_transition(
        source_camera=source_camera,
        destination_camera=destination_camera,
        source_timestamp=source_vehicle.get("timestamp"),
        destination_timestamp=destination_vehicle.get("timestamp"),
        source_heading=source_heading,
        destination_heading=destination_heading,
    )


# ---------------------------------------------------------------------
# Persistence-friendly result
# ---------------------------------------------------------------------

def build_correlation_gis_fields(
    source_vehicle: dict[str, Any],
    destination_vehicle: dict[str, Any],
    *,
    metadata_loader=None,
) -> dict[str, Any]:
    """
    Return exactly the GIS fields needed by the existing
    CorrelationDecision persistence layer.
    """
    result = evaluate_vehicle_transition(
        source_vehicle,
        destination_vehicle,
        metadata_loader=metadata_loader,
    )

    return {
        "gis_score": result.gis_score,
        "direction_score": result.direction_score,
        "distance_meters": result.distance_meters,
        "bearing_degrees": result.bearing_degrees,
        "elapsed_seconds": result.elapsed_seconds,
        "implied_speed_kmh": result.implied_speed_kmh,
        "travel_time_score": result.travel_time_score,
        "feasible": result.feasible,
        "gis_reason": result.reason,
    }


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "CameraPoint",
    "GISScore",
    "validate_camera_point",
    "camera_from_dict",
    "haversine_distance_meters",
    "initial_bearing_degrees",
    "implied_speed_kmh",
    "direction_score",
    "distance_score",
    "travel_time_score",
    "evaluate_camera_transition",
    "evaluate_vehicle_transition",
    "build_correlation_gis_fields",
]
