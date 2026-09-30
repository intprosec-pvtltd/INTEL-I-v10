from sqlalchemy import (Column,Integer,BigInteger,String,DateTime,Boolean,Text,ForeignKey,LargeBinary,Index,Float,JSON,UniqueConstraint )
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from datetime import datetime, timezone
from db.database import Base
def indian_time():
    """Backward-compatible name; database timestamps are canonical UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    cam_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    camera_name = Column(
        String(150),
        nullable=False,
    )

    zone = Column(
        String(150),
        nullable=True,
    )

    zones = Column(
        JSON,
        nullable=True,
        default=list,
    )

    source = Column(
        Text,
        nullable=False,
    )
    encrypted_source = Column(
        Text,
        nullable=True,
    )

    source_type = Column(
        String(50),
        nullable=False,
        index=True,
    )

    # Connector metadata. Secrets are encrypted at rest in
    # connector_config_encrypted and are never exposed by camera APIs.
    connector_type = Column(
        String(50),
        nullable=True,
        index=True,
    )

    vendor = Column(
        String(100),
        nullable=True,
    )

    vendor_device_id = Column(
        String(150),
        nullable=True,
    )

    stream_profile = Column(
        String(256),
        nullable=True,
    )

    connector_config_encrypted = Column(
        Text,
        nullable=True,
    )

    resolved_source_type = Column(
        String(50),
        nullable=True,
    )

    connection_state = Column(
        String(30),
        nullable=True,
        default="OFFLINE",
    )

    last_health_check = Column(
        DateTime,
        nullable=True,
    )
    last_seen_at = Column(DateTime, nullable=True, index=True)
    last_error_code = Column(String(100), nullable=True)
    last_error_message = Column(String(500), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    timestamp_status = Column(String(30), nullable=True)
    district = Column(String(150), nullable=True, index=True)
    jurisdiction = Column(String(150), nullable=True)
    hostname = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True, index=True)
    onvif_uuid = Column(String(255), nullable=True, index=True)
    serial_number = Column(String(255), nullable=True, index=True)
    mac_address = Column(String(32), nullable=True)
    firmware_version = Column(String(100), nullable=True)
    main_stream_encrypted = Column(Text, nullable=True)
    sub_stream_encrypted = Column(Text, nullable=True)
    group_id = Column(Integer, ForeignKey("camera_groups.id", ondelete="SET NULL"), nullable=True, index=True)
    credential_profile_id = Column(Integer, ForeignKey("camera_credentials.id", ondelete="SET NULL"), nullable=True, index=True)
    ai_profile_id = Column(Integer, ForeignKey("camera_ai_profiles.id", ondelete="SET NULL"), nullable=True, index=True)
    processing_enabled = Column(Boolean, nullable=False, default=False, index=True)
    analytics_enabled = Column(Boolean, nullable=False, default=False)
    preferred_worker_pool = Column(String(100), nullable=True, index=True)
    assigned_worker_id = Column(String(150), nullable=True, index=True)
    worker_status = Column(String(30), nullable=True)
    assigned_at = Column(DateTime, nullable=True)

    # Canonical normalized camera contract (Phase 3).
    direction = Column(String(20), nullable=True)
    stream_fps = Column(Float, nullable=True)
    stream_width = Column(Integer, nullable=True)
    stream_height = Column(Integer, nullable=True)
    transport = Column(String(20), nullable=True)
    codec = Column(String(32), nullable=True)

    # External catalogue ownership and synchronization provenance. Stream
    # URLs remain encrypted in encrypted_source/connector_config_encrypted.
    integration_id = Column(
        Integer,
        ForeignKey("camera_integrations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    external_provider = Column(String(50), nullable=True, index=True)
    external_camera_id = Column(String(150), nullable=True, index=True)
    external_live_status = Column(String(30), nullable=True, index=True)
    external_metadata = Column(JSON, nullable=True)
    external_metadata_hash = Column(String(64), nullable=True, index=True)
    external_last_seen_at = Column(DateTime, nullable=True)
    external_last_synced_at = Column(DateTime, nullable=True)
    externally_managed = Column(Boolean, nullable=False, default=False)


    # ----------------------------------------------------------------
    # Backward-compatible WGS84 coordinate fields.
    #
    # Existing REST responses, frontend maps and correlation code continue
    # reading latitude/longitude.  PostGIS `location` is the canonical spatial
    # query column and is kept in sync by the database migration/triggers.
    # ----------------------------------------------------------------
    latitude = Column(
        Float,
        nullable=True,
    )

    longitude = Column(
        Float,
        nullable=True,
    )

    # Canonical PostGIS camera position.
    #
    # IMPORTANT:
    # - Geometry uses X=longitude and Y=latitude.
    # - SRID 4326 is WGS84, matching browser/GIS latitude-longitude inputs.
    # - spatial_index=False because INTEL-I declares one explicit, stable GiST
    #   index name below.  This avoids duplicate indexes during Alembic
    #   autogeneration and keeps schema ownership explicit.
    # - nullable=True preserves cameras whose location is not yet known.
    location = Column(
        Geometry(
            geometry_type="POINT",
            srid=4326,
            dimension=2,
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="geometry",
        ),
        nullable=True,
    )

    # ============================================================
    # GIS / GEOSPATIAL METADATA
    # ============================================================

    altitude = Column(
        Float,
        nullable=True,
    )

    heading = Column(
        Float,
        nullable=True,
    )

    fov = Column(
        Float,
        nullable=True,
    )

    location_name = Column(
        String(255),
        nullable=True,
    )

    road_name = Column(
        String(255),
        nullable=True,
    )

    gis_calibration = Column(
        JSON,
        nullable=True,
    )

    city = Column(
        String(100),
        nullable=True,
    )

    state = Column(
        String(100),
        nullable=True,
    )

    country = Column(
        String(100),
        nullable=True,
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
    )

    # Distributed analytics intent.  is_active is retained for backward
    # compatibility with the existing camera catalogue/UI; desired_state is
    # the authoritative worker command used by the 60-camera architecture.
    desired_state = Column(
        String(20),
        default="STOPPED",
        nullable=False,
        index=True,
    )

    desired_state_updated_at = Column(
        DateTime,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="cameras",
    )

    integration = relationship(
        "CameraIntegration",
        back_populates="cameras",
    )

    __table_args__ = (
        Index(
            "uq_camera_user_cam",
            "user_id",
            "cam_id",
            unique=True,
        ),
        Index(
            "idx_camera_user",
            "user_id",
        ),
        Index(
            "idx_camera_cam_type",
            "cam_id",
            "source_type",
        ),
        Index(
            "uq_camera_external_identity",
            "user_id",
            "external_provider",
            "external_camera_id",
            unique=True,
        ),
        # Production PostGIS spatial index used by ST_DWithin, bounding-box
        # filtering and nearest-camera/correlation candidate queries.
        Index(
            "idx_cameras_location_gist",
            "location",
            postgresql_using="gist",
        ),
    )


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    cam_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    alert_type = Column(
        String(50),
        nullable=False,
        index=True,
    )

    level = Column(
        String(50),
        nullable=False,
        index=True,
    )

    # Canonical confidence used by Alert History and evidence exports.
    # Stored as a normalized ratio in the inclusive range 0.0..1.0.
    # Nullable preserves compatibility with legacy alerts whose source
    # confidence was not persisted.
    confidence_score = Column(
        Float,
        nullable=True,
    )

    watchlist_entry_id = Column(
        BigInteger,
        nullable=True,
        index=True,
    )

    plate = Column(
        String(50),
        nullable=True,
        index=True,
    )

    watchlist_category = Column(
        String(50),
        nullable=True,
    )

    watchlist_status = Column(
        String(30),
        nullable=True,
    )

    watchlist_match_type = Column(
        String(30),
        nullable=True,
    )

    watchlist_match_confidence = Column(
        Float,
        nullable=True,
    )

    watchlist_version = Column(
        Integer,
        nullable=True,
    )

    track_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    rule = Column(
        String(100),
        nullable=False,
        index=True,
    )

    source_type = Column(
        String(50),
        nullable=True,
        index=True,
    )

    zone = Column(
        String(150),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
        index=True,
    )

    # Evidence time is distinct from database insertion time. The source
    # timestamp is derived from decoder PTS when available.
    source_timestamp = Column(DateTime, nullable=True, index=True)
    source_pts_seconds = Column(Float, nullable=True)
    timestamp_source = Column(String(40), nullable=True)
    timestamp_quality = Column(String(30), nullable=True, index=True)

    user = relationship(
        "User",
        back_populates="alerts",
    )

    snapshots = relationship(
        "Snapshot",
        back_populates="alert",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    __table_args__ = (
        Index(
            "idx_alert_user_time",
            "user_id",
            "created_at",
        ),
        Index(
            "idx_alert_user_cam_rule_time",
            "user_id",
            "cam_id",
            "rule",
            "created_at",
        ),
        Index(
            "idx_alert_user_cam_track_rule",
            "user_id",
            "cam_id",
            "track_id",
            "rule",
        ),
    )


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    alert_id = Column(
        Integer,
        ForeignKey("alerts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    image_data = Column(
        LargeBinary,
        nullable=False,
    )

    image_type = Column(
        String(50),
        default="image/jpeg",
        nullable=False,
    )

    sha256 = Column(
        String(64),
        nullable=True,
        index=True,
    )

    is_original = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
        index=True,
    )

    alert = relationship(
        "Alert",
        back_populates="snapshots",
    )

    __table_args__ = (
        Index(
            "idx_snapshot_alert_time",
            "alert_id",
            "created_at",
        ),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    full_name = Column(
        String(150),
        nullable=False,
    )

    email = Column(
        String(150),
        unique=True,
        nullable=False,
        index=True,
    )

    password = Column(
        String(255),
        nullable=False,
    )

    role = Column(
        String(50),
        default="user",
        nullable=False,
    )

    department = Column(String(20), nullable=True, index=True)

    manager_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    token_version = Column(Integer, nullable=False, default=0)

    updated_at = Column(
        DateTime,
        default=indian_time,
        onupdate=indian_time,
        nullable=False,
    )

    manager = relationship(
        "User",
        remote_side=[id],
        back_populates="staff_members",
        foreign_keys=[manager_id],
    )

    staff_members = relationship(
        "User",
        back_populates="manager",
        foreign_keys=[manager_id],
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
    )

    cameras = relationship(
        "Camera",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    camera_integrations = relationship(
        "CameraIntegration",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    uploaded_videos = relationship(
        "UploadedVideo",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    refresh_tokens = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    alerts = relationship(
        "Alert",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    incidents = relationship(
        "Incident",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class CameraIntegration(Base):
    """Encrypted, user-scoped Sentinel/VMS/NVR catalogue configuration."""

    __tablename__ = "camera_integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(150), nullable=False)
    provider_type = Column(String(50), nullable=False, index=True)
    config_encrypted = Column(Text, nullable=False)
    config_fingerprint = Column(String(64), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    auto_sync = Column(Boolean, nullable=False, default=True, index=True)
    sync_interval_seconds = Column(Integer, nullable=False, default=300)
    last_sync_started_at = Column(DateTime, nullable=True)
    last_sync_at = Column(DateTime, nullable=True, index=True)
    last_sync_status = Column(String(30), nullable=True, index=True)
    last_error_code = Column(String(80), nullable=True)
    last_error_message = Column(String(500), nullable=True)
    last_camera_count = Column(Integer, nullable=False, default=0)
    last_created_count = Column(Integer, nullable=False, default=0)
    last_updated_count = Column(Integer, nullable=False, default=0)
    last_offline_count = Column(Integer, nullable=False, default=0)
    source_revision = Column(String(128), nullable=True)
    etag = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=indian_time, nullable=False)
    updated_at = Column(DateTime, default=indian_time, onupdate=indian_time, nullable=False)

    user = relationship("User", back_populates="camera_integrations")
    cameras = relationship("Camera", back_populates="integration")
    sync_runs = relationship(
        "CameraIntegrationSyncRun",
        back_populates="integration",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_camera_integration_user_name"),
        Index("idx_camera_integration_due", "enabled", "auto_sync", "last_sync_at"),
    )


class CameraIntegrationSyncRun(Base):
    __tablename__ = "camera_integration_sync_runs"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, index=True)
    integration_id = Column(
        Integer,
        ForeignKey("camera_integrations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trigger = Column(String(30), nullable=False, default="manual")
    status = Column(String(30), nullable=False, index=True)
    discovered_count = Column(Integer, nullable=False, default=0)
    created_count = Column(Integer, nullable=False, default=0)
    updated_count = Column(Integer, nullable=False, default=0)
    skipped_count = Column(Integer, nullable=False, default=0)
    offline_count = Column(Integer, nullable=False, default=0)
    source_revision = Column(String(128), nullable=True)
    error_code = Column(String(80), nullable=True)
    error_message = Column(String(500), nullable=True)
    started_at = Column(DateTime, default=indian_time, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)

    integration = relationship("CameraIntegration", back_populates="sync_runs")

    __table_args__ = (
        Index("idx_camera_integration_run_time", "integration_id", "started_at"),
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    token_jti = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    expires_at = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    revoked = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    created_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="refresh_tokens",
    )

    __table_args__ = (
        Index(
            "idx_refresh_user_revoked",
            "user_id",
            "revoked",
        ),
        Index(
            "idx_refresh_jti_revoked",
            "token_jti",
            "revoked",
        ),
        Index(
            "idx_refresh_expires",
            "expires_at",
        ),
    )


class UploadedVideo(Base):
    __tablename__ = "uploaded_videos"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    cam_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    stream_id = Column(
        String(120),
        nullable=False,
        unique=True,
        index=True,
    )

    storage_key = Column(
        String(255),
        nullable=False,
    )

    original_filename = Column(
        String(255),
        nullable=True,
    )

    mime_type = Column(
        String(100),
        nullable=True,
    )

    size_bytes = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    status = Column(
        String(50),
        default="READY",
        nullable=False,
        index=True,
    )

    created_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
        index=True,
    )

    deleted_at = Column(
        DateTime,
        nullable=True,
        index=True,
    )

    user = relationship(
        "User",
        back_populates="uploaded_videos",
    )

    __table_args__ = (
        Index(
            "idx_uploaded_video_user",
            "user_id",
        ),
        Index(
            "idx_uploaded_video_user_stream",
            "user_id",
            "stream_id",
        ),
        Index(
            "idx_uploaded_video_user_cam",
            "user_id",
            "cam_id",
        ),
        Index(
            "idx_uploaded_video_status_time",
            "status",
            "created_at",
        ),
    )


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    cam_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    primary_track_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    global_vehicle_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    incident_type = Column(
        String(100),
        nullable=False,
        index=True,
    )

    status = Column(
        String(50),
        default="OPEN",
        nullable=False,
        index=True,
    )

    evidence = Column(
        Text,
        nullable=True,
    )

    started_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
        index=True,
    )

    ended_at = Column(
        DateTime,
        nullable=True,
    )

    # ============================================================
    # INCIDENT EVALUATION
    # ============================================================

    evaluation_status = Column(
        String(50),
        default="PENDING",
        nullable=False,
        index=True,
    )

    evaluation_score = Column(
        Float,
        nullable=True,
    )

    evaluation_confidence = Column(
        Float,
        nullable=True,
    )

    evaluation_severity = Column(
        String(50),
        nullable=True,
    )

    evaluation_reason = Column(
        Text,
        nullable=True,
    )

    evaluation_evidence = Column(
        JSON,
        nullable=True,
    )

    evaluated_at = Column(
        DateTime,
        nullable=True,
    )

    evaluated_by = Column(
        String(100),
        nullable=True,
    )

    user = relationship(
        "User",
        back_populates="incidents",
    )

    evidence_items = relationship(
        "IncidentEvidence",
        back_populates="incident",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    __table_args__ = (
        Index(
            "idx_incident_user_status",
            "user_id",
            "status",
        ),
        Index(
            "idx_incident_user_cam_status",
            "user_id",
            "cam_id",
            "status",
        ),
        Index(
            "idx_incident_user_type_time",
            "user_id",
            "incident_type",
            "started_at",
        ),
    )


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    topic = Column(
        String(255),
        nullable=False,
        index=True,
    )

    event_key = Column(
        String(255),
        nullable=True,
        index=True,
    )

    payload = Column(
        Text,
        nullable=False,
    )

    status = Column(
        String(50),
        default="PENDING",
        nullable=False,
        index=True,
    )

    attempts = Column(
        Integer,
        default=0,
        nullable=False,
    )

    next_attempt_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
        index=True,
    )

    last_error = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
        index=True,
    )

    updated_at = Column(
        DateTime,
        default=indian_time,
        nullable=False,
    )

    __table_args__ = (
        Index(
            "idx_outbox_status_next",
            "status",
            "next_attempt_at",
        ),
        Index(
            "idx_outbox_topic_status",
            "topic",
            "status",
        ),
    )
    
class CameraConnection(Base):
    """
    Persistent GIS relationship between two CCTV cameras.

    Example:
        CAM-001 -> CAM-002

    Used by the GIS engine for:
    - camera topology
    - vehicle transition validation
    - travel-time feasibility
    - direction analysis
    - GIS correlation scoring
    """

    __tablename__ = "camera_connections"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    source_camera_id = Column(
        String(255),
        nullable=False,
        index=True,
    )

    destination_camera_id = Column(
        String(255),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------
    # GEOGRAPHIC INFORMATION
    # ------------------------------------------------------------

    distance_km = Column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------
    # TRAVEL-TIME INFORMATION
    # ------------------------------------------------------------

    expected_min_seconds = Column(
        Float,
        nullable=True,
    )

    expected_max_seconds = Column(
        Float,
        nullable=True,
    )

    expected_seconds = Column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------
    # ROAD / DIRECTION
    # ------------------------------------------------------------

    road_name = Column(
        String(255),
        nullable=True,
    )

    direction = Column(
        Float,
        nullable=True,
    )

    direction_tolerance = Column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------
    # ROAD ROUTING
    # ------------------------------------------------------------

    road_distance_km = Column(
        Float,
        nullable=True,
    )

    road_duration_seconds = Column(
        Float,
        nullable=True,
    )

    route_geometry = Column(
        Text,
        nullable=True,
    )

    routing_provider = Column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------
    # INTELLIGENCE
    # ------------------------------------------------------------

    confidence = Column(
        Float,
        nullable=True,
    )

    active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    # ------------------------------------------------------------
    # AUDIT
    # ------------------------------------------------------------

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # ------------------------------------------------------------
    # CONSTRAINTS / INDEXES
    # ------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "source_camera_id",
            "destination_camera_id",
            name="uq_camera_connection_source_destination",
        ),

        Index(
            "ix_camera_connection_source_destination",
            "source_camera_id",
            "destination_camera_id",
        ),

        Index(
            "ix_camera_connection_active_source",
            "active",
            "source_camera_id",
        ),

        Index(
            "ix_camera_connection_active_destination",
            "active",
            "destination_camera_id",
        ),
    )

    def __repr__(self) -> str:
        return (
            "<CameraConnection("
            f"id={self.id}, "
            f"source={self.source_camera_id!r}, "
            f"destination={self.destination_camera_id!r}, "
            f"distance_km={self.distance_km}, "
            f"active={self.active}"
            ")>"
        )

# ============================================================
# PHASE 0-3 OPERATIONAL AUDIT TABLES
# ============================================================

class CameraHealth(Base):
    """Immutable-ish health observations for camera lifecycle auditing."""
    __tablename__ = "camera_health"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String(100), nullable=False, index=True)
    state = Column(String(30), nullable=False, index=True)
    frame_count = Column(BigInteger, nullable=False, default=0)
    last_frame_at = Column(DateTime, nullable=True)
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)

    __table_args__ = (
        Index("idx_camera_health_camera_time", "camera_id", "created_at"),
        Index("idx_camera_health_state_time", "state", "created_at"),
    )


class CameraCredential(Base):
    __tablename__ = "camera_credentials"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    username = Column(String(255), nullable=True)
    secret_encrypted = Column(Text, nullable=False)
    auth_type = Column(String(30), nullable=False, default="rtsp")
    created_at = Column(DateTime, nullable=False, default=indian_time)
    updated_at = Column(DateTime, nullable=False, default=indian_time, onupdate=indian_time)
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_camera_credential_user_name"),)


class CameraGroup(Base):
    __tablename__ = "camera_groups"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    parent_id = Column(Integer, ForeignKey("camera_groups.id", ondelete="CASCADE"), nullable=True, index=True)
    group_type = Column(String(50), nullable=False, default="logical")
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time)
    updated_at = Column(DateTime, nullable=False, default=indian_time, onupdate=indian_time)
    __table_args__ = (UniqueConstraint("user_id", "parent_id", "name", name="uq_camera_group_parent_name"),)


class CameraAIProfile(Base):
    __tablename__ = "camera_ai_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    configuration = Column(JSON, nullable=False, default=dict)
    processing_fps = Column(Float, nullable=False, default=5.0)
    preferred_stream = Column(String(20), nullable=False, default="substream")
    evidence_enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=indian_time)
    updated_at = Column(DateTime, nullable=False, default=indian_time, onupdate=indian_time)
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_camera_ai_profile_user_name"),)


class CameraOnboardingJob(Base):
    __tablename__ = "camera_onboarding_jobs"
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    job_type = Column(String(30), nullable=False, index=True)
    status = Column(String(30), nullable=False, default="QUEUED", index=True)
    total = Column(Integer, nullable=False, default=0)
    completed = Column(Integer, nullable=False, default=0)
    successful = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    duplicate = Column(Integer, nullable=False, default=0)
    request_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    __table_args__ = (Index("idx_onboarding_jobs_user_time", "user_id", "created_at"),)


class CameraOnboardingItem(Base):
    __tablename__ = "camera_onboarding_items"
    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("camera_onboarding_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    ordinal = Column(Integer, nullable=False)
    camera_id = Column(String(100), nullable=True, index=True)
    camera_name = Column(String(150), nullable=True)
    source_encrypted = Column(Text, nullable=True)
    input_data_encrypted = Column(Text, nullable=True)
    status = Column(String(30), nullable=False, default="QUEUED", index=True)
    error_code = Column(String(100), nullable=True)
    error_message = Column(String(500), nullable=True)
    result_metadata = Column(JSON, nullable=True)
    registered_camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=indian_time)
    updated_at = Column(DateTime, nullable=False, default=indian_time, onupdate=indian_time)
    __table_args__ = (UniqueConstraint("job_id", "ordinal", name="uq_onboarding_item_ordinal"),)


class EvidenceHash(Base):
    """Separate integrity ledger for evidence objects."""
    __tablename__ = "evidence_hashes"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    evidence_key = Column(String(500), nullable=False, unique=True, index=True)
    sha256 = Column(String(64), nullable=False, index=True)
    algorithm = Column(String(20), nullable=False, default="SHA-256")
    is_original = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)


class AuditLog(Base):
    """Security/operations audit trail for Phase 2 persistence."""
    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(150), nullable=True, index=True)
    details = Column(JSON, nullable=True)
    source_ip = Column(String(64), nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)

    __table_args__ = (
        Index("idx_audit_user_time", "user_id", "created_at"),
        Index("idx_audit_resource_time", "resource_type", "resource_id", "created_at"),
    )


class SystemEvent(Base):
    """Persistent lifecycle/system event ledger."""
    __tablename__ = "system_events"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(30), nullable=False, default="INFO", index=True)
    component = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)


class CameraTimeSync(Base):
    """Latest per-camera clock synchronization state."""
    __tablename__ = "camera_time_sync"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String(100), nullable=False, unique=True, index=True)
    camera_timestamp = Column(DateTime, nullable=True)
    server_timestamp = Column(DateTime, nullable=False)
    ingestion_timestamp = Column(DateTime, nullable=False)
    normalized_timestamp = Column(DateTime, nullable=False)
    clock_offset_ms = Column(Float, nullable=True)
    jitter_ms = Column(Float, nullable=True)
    sync_status = Column(String(30), nullable=False, default="ESTIMATED", index=True)
    sync_method = Column(String(50), nullable=False, default="ARRIVAL_TIME")
    sample_count = Column(BigInteger, nullable=False, default=0)
    source_timestamp_available = Column(Boolean, nullable=False, default=False)
    pts_seconds = Column(Float, nullable=True)
    timestamp_quality = Column(String(30), nullable=True)
    discontinuity_count = Column(BigInteger, nullable=False, default=0)
    last_discontinuity_reason = Column(String(80), nullable=True)
    last_sync_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)
    updated_at = Column(DateTime, nullable=False, default=indian_time, index=True)


class StreamMetric(Base):
    """Rate-limited immutable stream telemetry snapshots."""
    __tablename__ = "stream_metrics"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String(100), nullable=False, index=True)
    observed_fps = Column(Float, nullable=True)
    target_fps = Column(Float, nullable=True)
    decoded_frames = Column(BigInteger, nullable=False, default=0)
    processed_frames = Column(BigInteger, nullable=False, default=0)
    dropped_frames = Column(BigInteger, nullable=False, default=0)
    dropped_ratio = Column(Float, nullable=True)
    read_latency_ms = Column(Float, nullable=True)
    queue_depth = Column(Integer, nullable=False, default=0)
    queue_capacity = Column(Integer, nullable=False, default=1)
    reconnect_count = Column(Integer, nullable=False, default=0)
    discontinuity_count = Column(BigInteger, nullable=False, default=0)
    last_frame_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)


class WatchlistVersion(Base):
    """Version ledger for watchlist configuration changes."""
    __tablename__ = "watchlist_versions"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    watchlist_id = Column(BigInteger, ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    snapshot = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)

    __table_args__ = (
        UniqueConstraint("watchlist_id", "version", name="uq_watchlist_version"),
    )
