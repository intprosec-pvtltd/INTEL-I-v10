"""Resolve encrypted Camera rows into in-memory capture inputs for workers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from connectors.manager import validate_connector_config
from db.database import SessionLocal
from db.crud import get_uploaded_video_by_cam
from security.cameraSource import decrypt_camera_source
from security.connectorConfig import decrypt_connector_config
from services.objectStorage import (
    ObjectNotFoundError,
    ObjectStorageError,
    get_object_storage,
)


DIRECT_SOURCE_TYPES = {"rtsp", "http", "https", "live", "hls"}
CONNECTOR_SOURCE_TYPES = {"onvif", "nvr", "vms", "vendor_api", "vendor_sdk"}
RECORDED_SOURCE_TYPES = {"upload", "recorded_video"}

SUPPORTED_SOURCE_TYPES = (
    DIRECT_SOURCE_TYPES
    | CONNECTOR_SOURCE_TYPES
    | RECORDED_SOURCE_TYPES
    | {"webcam"}
)


@dataclass(frozen=True)
class ResolvedCameraSource:
    source: Any
    source_type: str


def _validate_direct_source(source: str, source_type: str) -> None:
    value = str(source or "").strip()

    if not value:
        raise ValueError("Camera source is empty")

    if source_type == "rtsp":
        if not value.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("RTSP source must use rtsp:// or rtsps://")
        return

    if source_type in {"http", "https", "live", "hls"}:
        parsed = urlparse(value)

        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "HTTP/HLS source must use a valid http:// or https:// URL"
            )

        if source_type == "hls" and not parsed.path.lower().endswith(".m3u8"):
            raise ValueError("HLS source must point to a .m3u8 playlist")


def _resolve_recorded_source(camera, source_type: str) -> ResolvedCameraSource:
    """
    Resolve upload/recorded-video Camera rows to a real local media path.

    Camera.source intentionally stores the logical camera identifier, not a
    filesystem path. The durable video object is referenced by
    UploadedVideo.storage_key and lives in private object storage.

    Worker flow:
        Camera.cam_id
            -> UploadedVideo.storage_key
            -> object storage / MinIO
            -> verified local cache path
            -> PyAV/OpenCV
    """
    cam_id = str(getattr(camera, "cam_id", "") or "").strip()
    user_id = getattr(camera, "user_id", None)

    if not cam_id:
        raise RuntimeError("Recorded camera ID is not configured")

    try:
        normalized_user_id = int(user_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Recorded source owner is invalid for camera {cam_id}"
        ) from exc

    with SessionLocal() as db:
        uploaded = get_uploaded_video_by_cam(
            db=db,
            cam_id=cam_id,
            user_id=normalized_user_id,
        )

        if uploaded is None:
            raise FileNotFoundError(
                f"Uploaded video metadata not found for camera {cam_id}"
            )

        storage_key = str(getattr(uploaded, "storage_key", "") or "").strip()

        if not storage_key:
            raise RuntimeError(
                f"Uploaded video storage key is missing for camera {cam_id}"
            )

    try:
        local_path = get_object_storage().materialize_to_cache(storage_key)
    except ObjectNotFoundError as exc:
        raise FileNotFoundError(
            f"Uploaded video object not found for camera {cam_id}"
        ) from exc
    except ObjectStorageError as exc:
        raise RuntimeError(
            f"Uploaded video storage is unavailable for camera {cam_id}"
        ) from exc

    local_path = Path(local_path).resolve()

    if not local_path.is_file():
        raise FileNotFoundError(
            f"Uploaded video could not be materialized for camera {cam_id}"
        )

    try:
        size_bytes = local_path.stat().st_size
    except OSError as exc:
        raise RuntimeError(
            f"Unable to inspect materialized video for camera {cam_id}"
        ) from exc

    if size_bytes <= 0:
        raise RuntimeError(
            f"Materialized video is empty for camera {cam_id}"
        )

    return ResolvedCameraSource(
        source=str(local_path),
        source_type=source_type,
    )


def resolve_camera_row(camera) -> ResolvedCameraSource:
    source_type = str(
        getattr(camera, "source_type", "") or ""
    ).strip().lower()

    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unsupported camera source type: {source_type}"
        )

    # ============================================================
    # Direct network sources
    # ============================================================
    if source_type in DIRECT_SOURCE_TYPES:
        encrypted = getattr(camera, "encrypted_source", None)

        if not encrypted:
            raise RuntimeError(
                "Encrypted camera source is not configured"
            )

        source = decrypt_camera_source(encrypted)
        _validate_direct_source(source, source_type)

        return ResolvedCameraSource(
            source=source,
            source_type=source_type,
        )

    # ============================================================
    # Local webcam
    # ============================================================
    if source_type == "webcam":
        encrypted = getattr(camera, "encrypted_source", None)

        if not encrypted:
            raise RuntimeError(
                "Encrypted webcam source is not configured"
            )

        try:
            device = int(
                str(decrypt_camera_source(encrypted)).strip()
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid webcam device ID"
            ) from exc

        if device < 0:
            raise ValueError(
                "Webcam device ID cannot be negative"
            )

        return ResolvedCameraSource(
            source=device,
            source_type="webcam",
        )

    # ============================================================
    # Connector-backed cameras
    # ============================================================
    if source_type in CONNECTOR_SOURCE_TYPES:
        encrypted_config = getattr(
            camera,
            "connector_config_encrypted",
            None,
        )

        if not encrypted_config:
            raise RuntimeError(
                "Camera connector configuration is not configured"
            )

        config = decrypt_connector_config(
            encrypted_config
        )

        config = validate_connector_config(
            source_type,
            config,
        )

        # open_timestamped_capture resolves connector-backed
        # sources itself.
        return ResolvedCameraSource(
            source=config,
            source_type=source_type,
        )

    # ============================================================
    # Uploaded / recorded video
    # ============================================================
    if source_type in RECORDED_SOURCE_TYPES:
        return _resolve_recorded_source(
            camera,
            source_type,
        )

    # Defensive guard. SUPPORTED_SOURCE_TYPES above should make
    # this path unreachable.
    raise ValueError(
        f"Unsupported camera source type: {source_type}"
    )
