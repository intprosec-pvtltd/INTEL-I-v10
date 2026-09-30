from __future__ import annotations

import json
from typing import Any

from connectors.base import CameraConnector, ConnectorResult
from connectors.onvif.connector import ONVIFConnector
from connectors.vendor_api.connector import VendorAPIConnector
from connectors.vendor_sdk.loader import VendorSDKConnector


CONNECTOR_TYPES = {"onvif", "nvr", "vms", "vendor_api", "vendor_sdk"}
DIRECT_TYPES = {"rtsp", "http", "https", "live", "hls"}


def validate_connector_config(source_type: str, config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = str(source_type or "").strip().lower()
    if normalized not in CONNECTOR_TYPES:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"{normalized} connector configuration is required")

    encoded = json.dumps(config, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > 12000:
        raise ValueError("Connector configuration is too large")

    if normalized in {"nvr", "vms"}:
        # NVR/VMS sources use the vendor-API connector contract.
        normalized = "vendor_api"

    if normalized == "onvif":
        if not config.get("host"):
            raise ValueError("ONVIF host is required")
        if not config.get("username"):
            raise ValueError("ONVIF username is required")

    elif normalized == "vendor_api":
        if not config.get("stream_url") and not config.get("base_url"):
            raise ValueError("Vendor API base_url or stream_url is required")

    elif normalized == "vendor_sdk":
        if not config.get("module") or not config.get("class_name"):
            raise ValueError("Vendor SDK module and class_name are required")

    return config


def create_connector(source_type: str, config: dict[str, Any]) -> CameraConnector:
    normalized = str(source_type or "").strip().lower()
    config = validate_connector_config(normalized, config)

    if normalized in {"nvr", "vms"}:
        # NVR/VMS sources use the vendor-API connector contract.
        normalized = "vendor_api"

    if normalized == "onvif":
        return ONVIFConnector(config)
    if normalized == "vendor_api":
        return VendorAPIConnector(config)
    if normalized == "vendor_sdk":
        return VendorSDKConnector(config)

    raise ValueError(f"Unsupported connector type: {normalized}")


def resolve_connector(source_type: str, config: dict[str, Any]) -> ConnectorResult:
    connector = create_connector(source_type, config)
    try:
        return connector.resolve()
    finally:
        connector.close()


def open_camera_source(source_type: str, config: dict[str, Any]):
    """Return a capture-like object and normalized decoder type.

    The returned object always implements read() and release(), allowing the
    existing INTEL-I worker to remain connector-agnostic.
    """
    result = resolve_connector(source_type, config)

    if result.source_type in {"rtsp", "http", "https", "hls", "live"}:
        import cv2
        from main import _open_video_capture
        capture = _open_video_capture(result.source, result.source_type)
        if capture is None:
            raise RuntimeError("Resolved camera stream could not be opened")
        return capture, result.source_type, result.metadata

    # Native SDK frame sources must implement the same capture-like contract.
    source = result.source
    if not hasattr(source, "read") or not hasattr(source, "release"):
        raise RuntimeError("Vendor SDK returned an invalid frame source")
    return source, result.source_type, result.metadata
