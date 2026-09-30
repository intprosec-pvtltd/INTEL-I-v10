"""Hardened HTTP utilities and schema mapping for camera catalogues."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from connectors.catalogue import CatalogueCamera, StreamDescriptor


MAX_CATALOGUE_BYTES = 16 * 1024 * 1024
MAX_CATALOGUE_PAGES = 100
MAX_CAMERAS_PER_SYNC = 10000


def nested_get(data: Any, path: str | None, default: Any = None) -> Any:
    if not path:
        return data
    current = data
    for part in str(path).split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part, default)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else default
        else:
            return default
    return current


def first_value(data: Any, paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        value = nested_get(data, path)
        if value is not None and value != "":
            return value
    return default


def clean_text(value: Any, limit: int, default: str | None = None) -> str | None:
    if value is None:
        return default
    result = str(value).strip()
    if not result:
        return default
    return result[:limit]


def finite_float(value: Any, minimum: float, maximum: float) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        return None
    return result


def positive_int(value: Any, maximum: int = 16384) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if 1 <= result <= maximum else None


def source_type_for_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    scheme = parsed.scheme.lower()
    if scheme == "rtsp" and parsed.hostname:
        return "rtsp"
    if scheme in {"http", "https"} and parsed.hostname:
        if parsed.path.lower().endswith(".m3u8"):
            return "hls"
        return scheme
    return None


def validate_base_url(value: Any) -> str:
    url = str(value or "").strip().rstrip("/")
    if not url or len(url) > 2048 or any(char in url for char in "\r\n"):
        raise ValueError("Catalogue base_url is invalid")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Catalogue base_url must use http:// or https://")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Catalogue base_url must be an origin or base path without credentials")
    return url


def normalize_live_status(value: Any) -> str:
    if isinstance(value, bool):
        return "ONLINE" if value else "OFFLINE"
    normalized = str(value or "UNKNOWN").strip().upper().replace(" ", "_")
    if normalized in {"UP", "ACTIVE", "LIVE", "CONNECTED", "AVAILABLE", "RUNNING", "1"}:
        return "ONLINE"
    if normalized in {"DOWN", "INACTIVE", "DISCONNECTED", "UNAVAILABLE", "STOPPED", "0"}:
        return "OFFLINE"
    if normalized in {"DEGRADED", "UNSTABLE", "WARNING"}:
        return "DEGRADED"
    return "UNKNOWN"


def safe_endpoint(base_url: str, path: str) -> str:
    value = str(path or "/api/ingest").strip()
    if not value.startswith("/") or any(char in value for char in "\r\n"):
        raise ValueError("Catalogue path must be an absolute path")
    endpoint = urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))
    if urlparse(endpoint).netloc != urlparse(base_url).netloc:
        raise ValueError("Catalogue endpoint must remain on the configured host")
    return endpoint


def auth_headers(config: dict[str, Any]) -> tuple[dict[str, str], Any]:
    headers = {"Accept": "application/json", "User-Agent": "INTEL-I/1.0 CameraCatalogue"}
    supplied = config.get("headers") or {}
    if isinstance(supplied, dict):
        for key, value in supplied.items():
            key_text = str(key).strip()
            value_text = str(value).strip()
            if key_text and len(key_text) <= 128 and len(value_text) <= 4096:
                headers[key_text] = value_text

    auth = None
    auth_type = str(config.get("auth_type") or "none").lower()
    if auth_type == "bearer":
        token = str(config.get("token") or "").strip()
        if not token:
            raise ValueError("Bearer token is required")
        headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "api_key":
        api_key = str(config.get("api_key") or "").strip()
        header_name = str(config.get("api_key_header") or "X-API-Key").strip()
        if not api_key or not header_name or len(header_name) > 128:
            raise ValueError("API key and a valid API key header are required")
        headers[header_name] = api_key
    elif auth_type == "basic":
        username = str(config.get("username") or "").strip()
        if not username:
            raise ValueError("Username is required for basic authentication")
        auth = httpx.BasicAuth(username, str(config.get("password") or ""))
    elif auth_type != "none":
        raise ValueError("Unsupported catalogue authentication type")
    return headers, auth


def http_client(config: dict[str, Any], *, transport=None) -> httpx.Client:
    timeout_seconds = finite_float(config.get("timeout_seconds", 12), 1.0, 60.0) or 12.0
    verify_tls = bool(config.get("verify_tls", True))
    return httpx.Client(
        timeout=httpx.Timeout(timeout_seconds, connect=timeout_seconds),
        follow_redirects=False,
        verify=verify_tls,
        transport=transport,
    )


def parse_json_response(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        raise RuntimeError(f"Camera catalogue returned HTTP {response.status_code}")
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_CATALOGUE_BYTES:
        raise RuntimeError("Camera catalogue response is too large")
    if len(response.content) > MAX_CATALOGUE_BYTES:
        raise RuntimeError("Camera catalogue response is too large")
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Camera catalogue did not return valid JSON") from exc


def stream_from_mapping(raw: Any, *, default_profile: str | None = None) -> StreamDescriptor | None:
    if isinstance(raw, str):
        url = raw.strip()
        source_type = source_type_for_url(url)
        if not source_type:
            return None
        return StreamDescriptor(
            url=url,
            source_type=source_type,
            profile=default_profile,
            transport="tcp" if source_type == "rtsp" else None,
            priority={"rtsp": 10, "hls": 20, "https": 30, "http": 40}.get(source_type, 100),
        )
    if not isinstance(raw, dict):
        return None
    url = first_value(raw, ("url", "uri", "stream_url", "playback_url", "src"))
    source_type = source_type_for_url(str(url or ""))
    if not source_type:
        return None
    codec = clean_text(first_value(raw, ("codec", "video_codec", "encoding")), 32)
    profile = clean_text(first_value(raw, ("profile", "name", "id")), 128, default_profile)
    width = positive_int(first_value(raw, ("width", "resolution.width")))
    height = positive_int(first_value(raw, ("height", "resolution.height")))
    fps = finite_float(first_value(raw, ("fps", "frame_rate", "framerate")), 0.1, 240.0)
    priority = positive_int(raw.get("priority"), 10000)
    if priority is None:
        priority = {"rtsp": 10, "hls": 20, "https": 30, "http": 40}.get(source_type, 100)
    return StreamDescriptor(
        url=str(url).strip(),
        source_type=source_type,
        profile=profile,
        codec=codec,
        width=width,
        height=height,
        fps=fps,
        transport="tcp" if source_type == "rtsp" else clean_text(raw.get("transport"), 20),
        priority=priority,
    )


def generic_camera_from_mapping(raw: Any, mapping: dict[str, Any] | None = None) -> CatalogueCamera | None:
    if not isinstance(raw, dict):
        return None
    mapping = mapping or {}

    def mapped(name: str, defaults: tuple[str, ...], default=None):
        custom = mapping.get(name)
        paths = (str(custom),) if custom else defaults
        return first_value(raw, paths, default)

    external_id = clean_text(mapped("id", ("camera_id", "cam_id", "id", "device_id", "uuid")), 150)
    if not external_id:
        return None
    name = clean_text(mapped("name", ("camera_name", "name", "display_name", "title")), 150, external_id) or external_id
    streams: list[StreamDescriptor] = []

    streams_value = mapped("streams", ("streams", "stream_urls", "media.streams"))
    if isinstance(streams_value, dict):
        for profile, item in streams_value.items():
            if isinstance(item, list):
                for entry in item:
                    parsed = stream_from_mapping(entry, default_profile=str(profile))
                    if parsed:
                        streams.append(parsed)
            else:
                parsed = stream_from_mapping(item, default_profile=str(profile))
                if parsed:
                    streams.append(parsed)
    elif isinstance(streams_value, list):
        for item in streams_value:
            parsed = stream_from_mapping(item)
            if parsed:
                streams.append(parsed)

    known_stream_fields = (
        ("rtsp_url", "rtsp"), ("rtsp", "rtsp"),
        ("hls_url", "hls"), ("hls", "hls"),
        ("stream_url", "primary"), ("url", "primary"),
        ("http_url", "http"), ("https_url", "https"),
    )
    for field_name, profile in known_stream_fields:
        parsed = stream_from_mapping(raw.get(field_name), default_profile=profile)
        if parsed:
            streams.append(parsed)

    mapped_primary = mapped("stream_url", ("stream_url", "url", "playback_url"))
    parsed_primary = stream_from_mapping(mapped_primary, default_profile="primary")
    if parsed_primary:
        streams.append(parsed_primary)

    unique: dict[str, StreamDescriptor] = {}
    for stream in streams:
        unique.setdefault(stream.url, stream)
    streams = list(unique.values())
    default_codec = clean_text(mapped("codec", ("codec", "video_codec", "encoding")), 32)
    default_width = positive_int(mapped("width", ("width", "resolution.width")))
    default_height = positive_int(mapped("height", ("height", "resolution.height")))
    default_fps = finite_float(mapped("fps", ("fps", "frame_rate", "framerate")), 0.1, 240.0)
    streams = [
        replace(
            stream,
            codec=stream.codec or default_codec,
            width=stream.width or default_width,
            height=stream.height or default_height,
            fps=stream.fps or default_fps,
        )
        for stream in streams
    ]

    location = raw.get("location") if isinstance(raw.get("location"), dict) else {}
    latitude = finite_float(mapped("latitude", ("latitude", "lat", "location.latitude", "location.lat")), -90, 90)
    longitude = finite_float(mapped("longitude", ("longitude", "lng", "lon", "location.longitude", "location.lng", "location.lon")), -180, 180)

    metadata = {
        "raw_keys": sorted(str(key)[:80] for key in raw.keys())[:100],
        "stream_count": len(streams),
        "provider_metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
    }
    return CatalogueCamera(
        external_id=external_id,
        name=name,
        live_status=normalize_live_status(mapped("status", ("live_status", "status", "online", "is_live", "connected"))),
        streams=tuple(streams),
        location_name=clean_text(mapped("location_name", ("location_name", "address", "location.name", "location.address")), 255),
        latitude=latitude,
        longitude=longitude,
        altitude=finite_float(mapped("altitude", ("altitude", "location.altitude")), -1000, 10000),
        heading=finite_float(mapped("heading", ("heading", "bearing", "location.heading")), 0, 360),
        fov=finite_float(mapped("fov", ("fov", "field_of_view")), 1, 360),
        direction=clean_text(mapped("direction", ("direction", "orientation")), 20),
        road_name=clean_text(mapped("road_name", ("road_name", "road", "location.road")), 255),
        city=clean_text(mapped("city", ("city", "location.city")), 100),
        state=clean_text(mapped("state", ("state", "region", "location.state")), 100),
        country=clean_text(mapped("country", ("country", "location.country")), 100),
        vendor=clean_text(mapped("vendor", ("vendor", "manufacturer", "make")), 100),
        metadata=metadata,
    )
