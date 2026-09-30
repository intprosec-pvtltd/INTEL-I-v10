from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.camera_state import CameraState

SUPPORTED_SOURCE_TYPES = frozenset({
    "rtsp", "onvif", "nvr", "vms", "recorded_video",
    "http", "https", "live", "hls", "webcam", "upload",
    "vendor_api", "vendor_sdk",
})

NETWORK_SOURCE_TYPES = frozenset({"rtsp", "http", "https", "live", "hls"})
CONNECTOR_SOURCE_TYPES = frozenset({"onvif", "nvr", "vms", "vendor_api", "vendor_sdk"})
RECORDED_SOURCE_TYPES = frozenset({"upload", "recorded_video"})

SOURCE_ALIASES = {
    "recorded": "recorded_video",
    "recorded-video": "recorded_video",
    "recorded_video": "recorded_video",
    "nvr_stream": "nvr",
    "vms_stream": "vms",
}


def normalize_source_type(value: Any) -> str:
    source_type = str(value or "").strip().lower().replace(" ", "_")
    source_type = SOURCE_ALIASES.get(source_type, source_type)
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"Unsupported camera source type: {source_type or '<empty>'}")
    return source_type


def normalize_direction(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip().upper()
    if not value:
        return None
    allowed = {"NORTH", "NORTHEAST", "EAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "WEST", "NORTHWEST", "UNKNOWN"}
    if value not in allowed:
        raise ValueError("Camera direction must be a standard compass direction or UNKNOWN")
    return value


def _finite_float(value: Any, *, minimum: float, maximum: float, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _positive_int(value: Any, *, minimum: int, maximum: int, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


@dataclass(frozen=True)
class NormalizedCameraConfig:
    camera_id: str
    name: str
    source_type: str
    resolved_source_type: str | None = None
    connection_state: CameraState = CameraState.OFFLINE
    zone: str | None = None
    direction: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    heading: float | None = None
    fov: float | None = None
    location_name: str | None = None
    road_name: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    stream_fps: float | None = None
    stream_width: int | None = None
    stream_height: int | None = None
    transport: str | None = None
    codec: str | None = None
    connector_type: str | None = None
    vendor: str | None = None
    vendor_device_id: str | None = None
    stream_profile: str | None = None
    analytics: dict[str, bool] = field(default_factory=lambda: {
        "vehicle_detection": True,
        "vehicle_tracking": True,
        "anpr": True,
        "vehicle_reid": True,
    })

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["connection_state"] = self.connection_state.value
        # The public contract must never contain a plaintext source or secret.
        return payload


def normalize_camera_config(
    *,
    camera_id: Any,
    name: Any,
    source_type: Any,
    resolved_source_type: Any = None,
    connection_state: Any = CameraState.OFFLINE,
    zone: Any = None,
    direction: Any = None,
    latitude: Any = None,
    longitude: Any = None,
    altitude: Any = None,
    heading: Any = None,
    fov: Any = None,
    location_name: Any = None,
    road_name: Any = None,
    city: Any = None,
    state: Any = None,
    country: Any = None,
    stream_fps: Any = None,
    stream_width: Any = None,
    stream_height: Any = None,
    transport: Any = None,
    codec: Any = None,
    connector_type: Any = None,
    vendor: Any = None,
    vendor_device_id: Any = None,
    stream_profile: Any = None,
    analytics: dict[str, Any] | None = None,
) -> NormalizedCameraConfig:
    camera_id = str(camera_id or "").strip()
    name = str(name or camera_id).strip()
    if not camera_id or len(camera_id) > 100:
        raise ValueError("Camera ID is required and must be <= 100 characters")
    if not name or len(name) > 150:
        raise ValueError("Camera name is required and must be <= 150 characters")

    source = normalize_source_type(source_type)
    resolved = normalize_source_type(resolved_source_type) if resolved_source_type else None
    state_value = connection_state if isinstance(connection_state, CameraState) else CameraState(str(connection_state or "OFFLINE").strip().upper())

    direction_value = normalize_direction(direction)
    lat = _finite_float(latitude, minimum=-90, maximum=90, name="latitude")
    lon = _finite_float(longitude, minimum=-180, maximum=180, name="longitude")
    altitude_value = _finite_float(altitude, minimum=-1000, maximum=10000, name="altitude")
    heading_value = _finite_float(heading, minimum=0, maximum=360, name="heading")
    fov_value = _finite_float(fov, minimum=1, maximum=360, name="fov")
    fps = _finite_float(stream_fps, minimum=0.1, maximum=120, name="stream_fps")
    width = _positive_int(stream_width, minimum=160, maximum=16384, name="stream_width")
    height = _positive_int(stream_height, minimum=120, maximum=16384, name="stream_height")

    transport_value = str(transport or "").strip().lower() or None
    if source == "rtsp":
        transport_value = "tcp"
    elif transport_value and transport_value not in {"tcp", "udp", "http", "https"}:
        raise ValueError("Unsupported camera transport")

    codec_value = str(codec or "").strip().lower() or None
    if codec_value and len(codec_value) > 32:
        raise ValueError("Camera codec is too long")

    def text(value, max_len):
        if value is None:
            return None
        value = str(value).strip()
        return value[:max_len] if value else None

    configured_analytics = {
        "vehicle_detection": True,
        "vehicle_tracking": True,
        "anpr": True,
        "vehicle_reid": True,
    }
    if isinstance(analytics, dict):
        for key in configured_analytics:
            if key in analytics:
                configured_analytics[key] = bool(analytics[key])

    connector = text(connector_type, 50)
    if source in {"onvif", "nvr", "vms"} and not connector:
        connector = source

    return NormalizedCameraConfig(
        camera_id=camera_id,
        name=name,
        source_type=source,
        resolved_source_type=resolved,
        connection_state=state_value,
        zone=text(zone, 150),
        direction=direction_value,
        latitude=lat,
        longitude=lon,
        altitude=altitude_value,
        heading=heading_value,
        fov=fov_value,
        location_name=text(location_name, 255),
        road_name=text(road_name, 255),
        city=text(city, 100),
        state=text(state, 100),
        country=text(country, 100),
        stream_fps=fps,
        stream_width=width,
        stream_height=height,
        transport=transport_value,
        codec=codec_value,
        connector_type=connector,
        vendor=text(vendor, 100),
        vendor_device_id=text(vendor_device_id, 150),
        stream_profile=text(stream_profile, 256),
        analytics=configured_analytics,
    )
