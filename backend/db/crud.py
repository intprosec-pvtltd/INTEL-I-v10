import hashlib
from datetime import timedelta
import json

from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from core.camera_contract import normalize_source_type
from db.model import (
    Camera,
    Alert,
    Snapshot,
    Incident,
    indian_time,
    UploadedVideo,
)

# Register advanced SQLAlchemy models before mapper configuration.
# Incident may reference IncidentEvidence by string relationship(), so the
# class must be present in SQLAlchemy's registry before worker queries run.
from db.advanced_intelligence_model import IncidentEvidence as _IncidentEvidence

from security.cameraSource import encrypt_camera_source
from security.connectorConfig import encrypt_connector_config
from connectors.manager import validate_connector_config


# ============================================================
# CONSTANTS
# ============================================================

# Camera source types that may contain a network URL.
NETWORK_SOURCE_TYPES = {
    "rtsp",
    "http",
    "https",
    "live",
    "hls",
    "onvif",
    "vendor_api",
    "vendor_sdk",
}


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _normalize_source_type(source_type: str | None) -> str:
    """
    Normalize and validate camera source type.

    Only known source types are accepted.
    """
    value = str(source_type or "").strip().lower()

    if not value:
        raise ValueError("Camera source type is required")

    allowed_types = {
        "rtsp", "http", "https", "live", "upload", "webcam", "hls",
        "onvif", "nvr", "vms", "recorded_video", "vendor_api", "vendor_sdk",
    }

    if value not in allowed_types:
        raise ValueError(
            f"Unsupported camera source type: {value}"
        )

    return value


def _validate_camera_source(
    source: str | None,
    source_type: str,
) -> str:
    """
    Validate the incoming camera source.

    IMPORTANT:
    The plaintext source is never returned by a camera API.
    It is only passed to the encryption helper here.
    """
    normalized_type = _normalize_source_type(source_type)

    if source is None:
        raise ValueError("Camera source is required")

    value = str(source).strip()

    if not value:
        raise ValueError("Camera source cannot be empty")

    # Prevent accidental excessively large values.
    if len(value) > 4096:
        raise ValueError("Camera source is too long")

    if normalized_type == "rtsp":
        if not value.lower().startswith("rtsp://"):
            raise ValueError(
                "RTSP camera source must start with rtsp://"
            )

    elif normalized_type in {"http", "https", "live"}:
        if not (
            value.lower().startswith("http://")
            or value.lower().startswith("https://")
        ):
            raise ValueError(
                "HTTP camera source must use http:// or https://"
            )

    elif normalized_type == "webcam":
        # Webcam sources should normally be numeric IDs.
        try:
            webcam_id = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Webcam source must be a numeric device ID"
            )

        if webcam_id < 0:
            raise ValueError(
                "Webcam device ID cannot be negative"
            )

    elif normalized_type == "upload":
        if not value:
            raise ValueError(
                "Upload source cannot be empty"
            )

    elif normalized_type == "hls":
        if not (
            value.lower().startswith("http://")
            or value.lower().startswith("https://")
        ) or not value.lower().split("?", 1)[0].endswith(".m3u8"):
            raise ValueError("HLS source must point to a .m3u8 playlist")

    elif normalized_type == "onvif":
        if not (
            value.lower().startswith("onvif://")
            or value.lower().startswith("http://")
            or value.lower().startswith("https://")
        ):
            raise ValueError("ONVIF source must use onvif://, http://, or https://")

    elif normalized_type == "vendor_api":
        if not (
            value.lower().startswith("http://")
            or value.lower().startswith("https://")
        ):
            raise ValueError("Vendor API source must use http:// or https://")

    elif normalized_type == "vendor_sdk":
        if not value.lower().startswith("sdk://"):
            raise ValueError("Vendor SDK source must use sdk://")

    return value


def _validate_coordinates(
    latitude,
    longitude,
):
    """
    Validate optional GIS coordinates.

    Returns normalized float values.
    """
    if latitude is None and longitude is None:
        return None, None

    if latitude is None or longitude is None:
        raise ValueError(
            "Both latitude and longitude are required"
        )

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        raise ValueError(
            "Latitude and longitude must be valid numbers"
        )

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(
            "Latitude must be between -90 and 90"
        )

    if not (-180.0 <= lon <= 180.0):
        raise ValueError(
            "Longitude must be between -180 and 180"
        )

    return lat, lon


def _validate_cam_id(cam_id: str) -> str:
    """
    Validate camera identifier.
    """
    value = str(cam_id or "").strip()

    if not value:
        raise ValueError("Camera ID is required")

    if len(value) > 100:
        raise ValueError("Camera ID is too long")

    return value


def _validate_camera_name(
    camera_name: str | None,
    fallback: str,
) -> str:
    """
    Normalize camera name.
    """
    value = str(camera_name or "").strip()

    if not value:
        value = fallback

    if len(value) > 150:
        raise ValueError(
            "Camera name is too long"
        )

    return value


def _validate_zone(zone: str | None) -> str | None:
    """
    Normalize optional zone.
    """
    if zone is None:
        return None

    value = str(zone).strip()

    if not value:
        return None

    if len(value) > 150:
        raise ValueError(
            "Zone name is too long"
        )

    return value

def _validate_location_name(
    location_name: str | None,
) -> str | None:


    if location_name is None:
        return None

    value = str(
        location_name
    ).strip()

    if not value:
        return None

    if len(value) > 255:
        raise ValueError(
            "Location name is too long"
        )

    return value


# ============================================================
# CAMERA CRUD
# ============================================================

def create_camera(
    db: Session,
    user_id: int,
    cam_id: str,
    source,
    source_type: str,
    camera_name: str = "",
    zone: str = "",
    latitude=None,
    longitude=None,
    location_name: str | None = None,
    connector_config: dict | None = None,
    vendor: str | None = None,
    vendor_device_id: str | None = None,
    stream_profile: str | None = None,
    direction: str | None = None,
    stream_fps: float | None = None,
    stream_width: int | None = None,
    stream_height: int | None = None,
    transport: str | None = None,
    codec: str | None = None,
    altitude=None,
    heading=None,
    fov=None,
    road_name: str | None = None,
    city: str | None = None,
    state: str | None = None,
    country: str | None = None,
    commit: bool = True,
):
    normalized_cam_id = _validate_cam_id(cam_id)

    normalized_source_type = normalize_source_type(source_type)

    normalized_source = _validate_camera_source(
        source,
        normalized_source_type,
    )

    normalized_name = _validate_camera_name(
        camera_name,
        normalized_cam_id,
    )

    normalized_zone = _validate_zone(zone)

    latitude_value, longitude_value = _validate_coordinates(
        latitude,
        longitude,
    )

    normalized_location_name = (
    _validate_location_name(
        location_name
    )
)

    encrypted_source = encrypt_camera_source(
        normalized_source
    )

    connector_type = (
        "vendor_api" if normalized_source_type in {"nvr", "vms"}
        else normalized_source_type
        if normalized_source_type in {"onvif", "vendor_api", "vendor_sdk"}
        else None
    )

    connector_config = validate_connector_config(
        normalized_source_type,
        connector_config,
    )

    encrypted_connector_config = encrypt_connector_config(
        connector_config
    )

    existing = get_camera(
        db,
        normalized_cam_id,
        user_id,
    )

    # ========================================================
    # UPDATE EXISTING CAMERA
    # ========================================================

    if existing:
        existing.source = normalized_cam_id

        existing.encrypted_source = encrypted_source

        existing.source_type = normalized_source_type
        existing.connector_type = connector_type
        existing.vendor = (
            str(vendor).strip()[:100] if vendor else None
        )
        existing.vendor_device_id = (
            str(vendor_device_id).strip()[:150] if vendor_device_id else None
        )
        existing.stream_profile = (
            str(stream_profile).strip()[:256] if stream_profile else None
        )
        existing.connector_config_encrypted = encrypted_connector_config
        existing.resolved_source_type = (
            normalized_source_type
            if normalized_source_type not in {"onvif", "nvr", "vms", "vendor_api", "vendor_sdk"}
            else None
        )
        existing.connection_state = "OFFLINE"
        existing.last_health_check = None
        existing.direction = direction
        existing.stream_fps = stream_fps
        existing.stream_width = stream_width
        existing.stream_height = stream_height
        existing.transport = "tcp" if normalized_source_type == "rtsp" else (str(transport).strip().lower() if transport else None)
        existing.codec = str(codec).strip().lower()[:32] if codec else None
        existing.altitude = altitude
        existing.heading = heading
        existing.fov = fov
        existing.road_name = str(road_name).strip()[:255] if road_name else None
        existing.city = str(city).strip()[:100] if city else None
        existing.state = str(state).strip()[:100] if state else None
        existing.country = str(country).strip()[:100] if country else None

        existing.camera_name = normalized_name

        existing.zone = (
            normalized_zone
            if normalized_source_type == "rtsp"
            else None
        )

        existing.latitude = latitude_value
        existing.longitude = longitude_value
        existing.location_name = normalized_location_name
        existing.is_active = True

        if commit:
            db.commit()
            db.refresh(existing)
        else:
            db.flush()

        return existing

    # ========================================================
    # CREATE NEW CAMERA
    # ========================================================

    camera = Camera(
        user_id=user_id,

        cam_id=normalized_cam_id,

        camera_name=normalized_name,

        zone=(
            normalized_zone
            if normalized_source_type == "rtsp"
            else None
        ),

        source=normalized_cam_id,

        encrypted_source=encrypted_source,

        source_type=normalized_source_type,
        connector_type=connector_type,
        vendor=(str(vendor).strip()[:100] if vendor else None),
        vendor_device_id=(
            str(vendor_device_id).strip()[:150]
            if vendor_device_id else None
        ),
        stream_profile=(
            str(stream_profile).strip()[:256]
            if stream_profile else None
        ),
        connector_config_encrypted=encrypted_connector_config,
        resolved_source_type=(
            normalized_source_type
            if normalized_source_type not in {"onvif", "nvr", "vms", "vendor_api", "vendor_sdk"}
            else None
        ),
        connection_state="OFFLINE",
        direction=direction,
        stream_fps=stream_fps,
        stream_width=stream_width,
        stream_height=stream_height,
        transport=("tcp" if normalized_source_type == "rtsp" else (str(transport).strip().lower() if transport else None)),
        codec=(str(codec).strip().lower()[:32] if codec else None),
        altitude=altitude,
        heading=heading,
        fov=fov,
        road_name=(str(road_name).strip()[:255] if road_name else None),
        city=(str(city).strip()[:100] if city else None),
        state=(str(state).strip()[:100] if state else None),
        country=(str(country).strip()[:100] if country else None),

        latitude=latitude_value,

        longitude=longitude_value,

        location_name=normalized_location_name,

        is_active=True,
        desired_state="STOPPED",
        desired_state_updated_at=None,
    )

    db.add(camera)
    if commit:
        db.commit()
        db.refresh(camera)
    else:
        db.flush()

    return camera


def get_cameras(
    db: Session,
    user_id: int,
):
    """
    Return cameras belonging to the authenticated user.

    NOTE:
    encrypted_source is intentionally NOT exposed by this
    function's API response layer.

    The model object contains the encrypted value internally,
    but the API schema should exclude it.
    """
    return (
        db.query(Camera)
        .filter(
            Camera.user_id == user_id
        )
        .order_by(
            Camera.created_at.desc()
        )
        .all()
    )


def get_requested_running_cameras(
    db: Session,
    limit: int = 200,
):
    """Return active cameras requested for distributed analytics.

    This is an internal worker query, not a user-facing authorization path.

    Live cameras, uploaded videos, and recorded-video sources all use the same
    distributed-worker ownership model.  The API persists desired_state only;
    a camera worker claims each RUNNING source, resolves/open its media source,
    executes analytics, and publishes runtime/preview state.

    ``is_active`` remains the administrative enable/disable gate.  A camera
    does not need to already be ONLINE before it is returned here because the
    worker is responsible for establishing the media connection and updating
    runtime connection state.
    """
    safe_limit = max(1, min(int(limit), 1000))

    return (
        db.query(Camera)
        .filter(
            Camera.desired_state == "RUNNING",
            Camera.is_active.is_(True),
        )
        .order_by(Camera.id.asc())
        .limit(safe_limit)
        .all()
    )


def get_camera_for_worker(
    db: Session,
    camera_pk: int,
):
    """Fetch one camera by database primary key for an internal worker."""
    try:
        camera_pk = int(camera_pk)
    except (TypeError, ValueError):
        return None
    return db.query(Camera).filter(Camera.id == camera_pk).first()


def set_camera_desired_state(
    db: Session,
    camera: Camera,
    desired_state: str,
    *,
    commit: bool = True,
):
    value = str(desired_state or "").strip().upper()
    if value not in {"RUNNING", "STOPPED"}:
        raise ValueError("Camera desired_state must be RUNNING or STOPPED")
    camera.desired_state = value
    camera.desired_state_updated_at = indian_time()
    if commit:
        db.commit()
        db.refresh(camera)
    else:
        db.flush()
    return camera


def get_camera(
    db: Session,
    cam_id: str,
    user_id: int,
):
    """
    Retrieve a camera belonging to the specified user.
    """
    normalized_cam_id = _validate_cam_id(
        cam_id
    )

    return (
        db.query(Camera)
        .filter(
            Camera.cam_id == normalized_cam_id,
            Camera.user_id == user_id,
        )
        .first()
    )


def delete_camera(
    db: Session,
    cam_id: str,
    user_id: int,
):
    """
    Delete a camera belonging to the specified user.
    """
    camera = get_camera(
        db,
        cam_id,
        user_id,
    )

    if not camera:
        return None

    db.delete(camera)
    db.commit()

    return camera


def set_camera_status(
    db: Session,
    cam_id: str,
    user_id: int,
    is_active: bool,
):
    """
    Update persistent camera active state.

    Worker/runtime state should still be maintained separately
    in Redis.
    """
    camera = get_camera(
        db,
        cam_id,
        user_id,
    )

    if not camera:
        return None

    camera.is_active = bool(is_active)

    db.commit()
    db.refresh(camera)

    return camera


# ============================================================
# ALERT CRUD
# ============================================================

def save_alert(
    db: Session,
    user_id: int,
    alert_type: str,
    alert_rule: str,
    track_id: str | None,
    cam_id: str,
    source_type: str | None = None,
    zone: str | None = None,
    level: str | None = None,
    cooldown_seconds: int = 180,
    source_timestamp=None,
    source_pts_seconds: float | None = None,
    timestamp_source: str | None = None,
    timestamp_quality: str | None = None,
    confidence_score: float | None = None,
):
    """
    Save an alert while preventing duplicate alerts during
    the configured cooldown period.
    """

    clean_cam_id = _validate_cam_id(
        cam_id
    )

    clean_rule = str(
        alert_rule or ""
    ).strip()

    if not clean_rule:
        raise ValueError(
            "Alert rule is required"
        )

    if len(clean_rule) > 100:
        raise ValueError(
            "Alert rule is too long"
        )

    final_level = str(
        level or alert_type or "LOW"
    ).strip()

    if len(final_level) > 50:
        raise ValueError(
            "Alert level is too long"
        )

    clean_track_id = (
        str(track_id)
        if track_id is not None
        else "global"
    )

    if len(clean_track_id) > 100:
        clean_track_id = clean_track_id[:100]

    cooldown_seconds = max(
        0,
        min(
            int(cooldown_seconds),
            86400,
        ),
    )

    normalized_confidence = None

    if confidence_score is not None:
        try:
            normalized_confidence = float(confidence_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Alert confidence must be a valid number"
            ) from exc

        if not 0.0 <= normalized_confidence <= 1.0:
            raise ValueError(
                "Alert confidence must be between 0.0 and 1.0"
            )

    since_time = (
        indian_time()
        - timedelta(
            seconds=cooldown_seconds
        )
    )

    existing_alert = (
        db.query(Alert)
        .filter(
            Alert.user_id == user_id,
            Alert.cam_id == clean_cam_id,
            Alert.rule == clean_rule,
            Alert.track_id == clean_track_id,
            Alert.created_at >= since_time,
        )
        .order_by(
            Alert.created_at.desc()
        )
        .first()
    )

    if existing_alert:
        return None, False

    alert = Alert(
        user_id=user_id,
        cam_id=clean_cam_id,
        alert_type=final_level,
        level=final_level,
        confidence_score=normalized_confidence,
        track_id=clean_track_id,
        rule=clean_rule,
        source_type=(
            str(source_type).strip()
            if source_type
            else None
        ),
        zone=(
            str(zone).strip()
            if zone
            else None
        ),
        source_timestamp=source_timestamp,
        source_pts_seconds=source_pts_seconds,
        timestamp_source=(str(timestamp_source)[:40] if timestamp_source else None),
        timestamp_quality=(str(timestamp_quality)[:30] if timestamp_quality else None),
    )

    db.add(alert)
    db.commit()
    db.refresh(alert)

    return alert, True


# ============================================================
# SNAPSHOT CRUD
# ============================================================

def save_snapshot_to_db(
    db,
    alert_id: int,
    snapshot_data,
    *,
    is_original: bool = False,
    metadata: dict | None = None,
):
    """
    Save alert snapshot.

    Snapshot image bytes are stored as binary data.
    """
    if not snapshot_data:
        return None

    if isinstance(
        snapshot_data,
        bytes,
    ):
        image_data = snapshot_data
        image_type = "image/jpeg"

    elif isinstance(
        snapshot_data,
        dict,
    ):
        image_data = snapshot_data.get(
            "image_data"
        )

        image_type = (
            snapshot_data.get(
                "image_type",
                "image/jpeg",
            )
        )

    else:
        raise ValueError(
            "Invalid snapshot data format"
        )

    if not image_data:
        return None

    snapshot = Snapshot(
        alert_id=alert_id,
        image_data=image_data,
        image_type=image_type,
        sha256=hashlib.sha256(image_data).hexdigest(),
        is_original=bool(is_original),
        metadata_json=metadata if isinstance(metadata, dict) else None,
    )

    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    return snapshot


def get_alerts(
    db: Session,
    user_id: int,
    limit: int = 100,
    cam_id: str | None = None,
    include_all_users: bool = False,
    visible_camera_ids: list[str] | tuple[str, ...] | set[str] | None = None,
):
    """
    Get alerts visible to the authenticated user.

    ``include_all_users`` must only be enabled by an already-authorized
    super-administrator route. It defaults to False so every existing caller
    remains tenant/user scoped.
    """
    safe_limit = max(
        1,
        min(int(limit), 500),
    )

    query = (
        db.query(Alert)
        .options(
            selectinload(
                Alert.snapshots
            )
        )
    )

    if not include_all_users:
        # Alerts created before the current camera-ownership contract can have
        # a legacy user_id.  Keep tenant isolation while allowing an operator
        # to see alerts belonging to a camera they are authorized to operate.
        # Never use this to grant a global alert view.
        allowed_cameras = [
            _validate_cam_id(camera_id)
            for camera_id in (visible_camera_ids or [])
            if str(camera_id or "").strip()
        ]
        if allowed_cameras:
            query = query.filter(
                or_(
                    Alert.user_id == user_id,
                    Alert.cam_id.in_(allowed_cameras),
                )
            )
        else:
            query = query.filter(Alert.user_id == user_id)

    if cam_id:
        query = query.filter(
            Alert.cam_id
            == _validate_cam_id(cam_id)
        )

    return (
        query
        .order_by(
            Alert.created_at.desc()
        )
        .limit(safe_limit)
        .all()
    )


def get_alerts_after_id(
    db: Session,
    user_id: int,
    after_id: int,
    limit: int = 100,
    include_all_users: bool = False,
    visible_camera_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    created_after=None,
):
    """Return durable alerts newer than the client's last seen alert ID."""
    safe_after = max(0, int(after_id))
    safe_limit = max(1, min(int(limit), 500))

    query = (
        db.query(Alert)
        .options(selectinload(Alert.snapshots))
        .filter(Alert.id > safe_after)
    )

    if created_after is not None:
        query = query.filter(Alert.created_at >= created_after)

    if not include_all_users:
        allowed_cameras = [
            _validate_cam_id(camera_id)
            for camera_id in (visible_camera_ids or [])
            if str(camera_id or "").strip()
        ]
        if allowed_cameras:
            query = query.filter(
                or_(
                    Alert.user_id == user_id,
                    Alert.cam_id.in_(allowed_cameras),
                )
            )
        else:
            query = query.filter(Alert.user_id == user_id)

    return (
        query
        .order_by(Alert.id.asc())
        .limit(safe_limit)
        .all()
    )


def get_alert_by_id(
    db: Session,
    alert_id: int,
):
    return (
        db.query(Alert)
        .options(
            selectinload(
                Alert.snapshots
            )
        )
        .filter(
            Alert.id == alert_id
        )
        .first()
    )


def get_snapshot(
    db: Session,
    snapshot_id: int,
):
    return (
        db.query(Snapshot)
        .filter(
            Snapshot.id == snapshot_id
        )
        .first()
    )


def get_snapshot_for_user(
    db: Session,
    snapshot_id: int,
    user_id: int,
):
    """
    Secure snapshot lookup.

    Ensures the snapshot belongs to an alert owned by
    the authenticated user.
    """
    return (
        db.query(Snapshot)
        .join(
            Alert,
            Snapshot.alert_id == Alert.id,
        )
        .filter(
            Snapshot.id == snapshot_id,
            Alert.user_id == user_id,
        )
        .first()
    )


# ============================================================
# INCIDENT CRUD
# ============================================================

def create_incident(
    db: Session,
    user_id: int,
    cam_id: str | None,
    primary_track_id,
    incident_type: str,
    evidence=None,
):
    """
    Create a new incident.
    """

    clean_cam_id = (
        _validate_cam_id(cam_id)
        if cam_id
        else None
    )

    clean_incident_type = str(
        incident_type or ""
    ).strip()

    if not clean_incident_type:
        raise ValueError(
            "Incident type is required"
        )

    if len(clean_incident_type) > 100:
        raise ValueError(
            "Incident type is too long"
        )

    clean_track_id = (
        str(primary_track_id)
        if primary_track_id is not None
        else None
    )

    if clean_track_id and len(
        clean_track_id
    ) > 100:
        clean_track_id = clean_track_id[:100]

    if isinstance(
        evidence,
        dict,
    ):
        safe_evidence = json.dumps(
            evidence
        )
    else:
        safe_evidence = evidence

    incident = Incident(
        user_id=user_id,
        cam_id=clean_cam_id,
        primary_track_id=clean_track_id,
        incident_type=clean_incident_type,
        evidence=safe_evidence,
        status="OPEN",
    )

    db.add(incident)
    db.commit()
    db.refresh(incident)

    return incident


def get_open_incidents(
    db: Session,
    user_id: int,
    cam_id: str | None = None,
    incident_type: str | None = None,
):
    query = (
        db.query(Incident)
        .filter(
            Incident.user_id == user_id,
            Incident.status == "OPEN",
        )
    )

    if cam_id:
        query = query.filter(
            Incident.cam_id
            == _validate_cam_id(cam_id)
        )

    if incident_type:
        clean_type = str(
            incident_type
        ).strip()

        query = query.filter(
            Incident.incident_type
            == clean_type
        )

    return (
        query
        .order_by(
            Incident.started_at.desc()
        )
        .all()
    )


def get_incidents(
    db: Session,
    user_id: int,
    limit: int = 100,
):
    safe_limit = max(
        1,
        min(int(limit), 500),
    )

    return (
        db.query(Incident)
        .filter(
            Incident.user_id == user_id
        )
        .order_by(
            Incident.started_at.desc()
        )
        .limit(safe_limit)
        .all()
    )


# ============================================================
# CAMERA ZONES
# ============================================================

def get_camera_zones(
    db: Session,
    cam_id: str,
    user_id: int,
):
    camera = get_camera(
        db,
        cam_id,
        user_id,
    )

    if not camera:
        return None

    return camera.zones or []


def update_camera_zones(
    db: Session,
    cam_id: str,
    user_id: int,
    zones: list,
):
    camera = get_camera(
        db,
        cam_id,
        user_id,
    )

    if not camera:
        return None

    if not isinstance(
        zones,
        list,
    ):
        raise ValueError(
            "Zones must be a list"
        )

    camera.zones = zones

    db.commit()
    db.refresh(camera)

    return camera


# ============================================================
# UPLOADED VIDEO CRUD
# ============================================================

def create_uploaded_video(
    db,
    user_id: int,
    cam_id: str,
    stream_id: str,
    storage_key: str,
    original_filename: str | None,
    mime_type: str | None,
    size_bytes: int,
):
    """
    Create uploaded-video metadata.

    Only storage_key is persisted here; actual filesystem
    locations should not be exposed through API responses.
    """

    clean_cam_id = _validate_cam_id(
        cam_id
    )

    clean_stream_id = str(
        stream_id or ""
    ).strip()

    if not clean_stream_id:
        raise ValueError(
            "Stream ID is required"
        )

    if len(clean_stream_id) > 120:
        raise ValueError(
            "Stream ID is too long"
        )

    clean_storage_key = str(
        storage_key or ""
    ).strip()

    if not clean_storage_key:
        raise ValueError(
            "Storage key is required"
        )

    if len(clean_storage_key) > 255:
        raise ValueError(
            "Storage key is too long"
        )

    safe_size = int(size_bytes)

    if safe_size < 0:
        raise ValueError(
            "File size cannot be negative"
        )

    uploaded = UploadedVideo(
        user_id=user_id,
        cam_id=clean_cam_id,
        stream_id=clean_stream_id,
        storage_key=clean_storage_key,
        original_filename=(
            str(original_filename).strip()
            if original_filename
            else None
        ),
        mime_type=(
            str(mime_type).strip()
            if mime_type
            else None
        ),
        size_bytes=safe_size,
        status="READY",
    )

    db.add(uploaded)
    db.commit()
    db.refresh(uploaded)

    return uploaded


def get_uploaded_video_by_stream(
    db,
    stream_id: str,
    user_id: int,
):
    clean_stream_id = str(
        stream_id or ""
    ).strip()

    if not clean_stream_id:
        return None

    return (
        db.query(UploadedVideo)
        .filter(
            UploadedVideo.stream_id
            == clean_stream_id,
            UploadedVideo.user_id
            == user_id,
        )
        .first()
    )


def get_uploaded_video_by_cam(
    db,
    cam_id: str,
    user_id: int,
):
    clean_cam_id = _validate_cam_id(
        cam_id
    )

    return (
        db.query(UploadedVideo)
        .filter(
            UploadedVideo.cam_id
            == clean_cam_id,
            UploadedVideo.user_id
            == user_id,
        )
        .first()
    )


def delete_uploaded_video_by_cam(
    db,
    cam_id: str,
    user_id: int,
):
    uploaded = get_uploaded_video_by_cam(
        db,
        cam_id,
        user_id,
    )

    if uploaded:
        db.delete(uploaded)
        db.commit()

    return uploaded
