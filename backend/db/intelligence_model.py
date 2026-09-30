from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    DateTime,
    Float,
    Boolean,
    Text,
    ForeignKey,
    Index,
    JSON,
    LargeBinary,
)

from sqlalchemy.orm import relationship

from db.database import Base


def indian_time():
    """Backward-compatible name; database timestamps are canonical UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ============================================================
# GLOBAL VEHICLE
# ============================================================

class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    global_vehicle_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    vehicle_type = Column(
        String(50),
        nullable=True,
        index=True,
    )

    color = Column(
        String(50),
        nullable=True,
    )

    make = Column(
        String(100),
        nullable=True,
    )

    model = Column(
        String(100),
        nullable=True,
    )

    first_seen_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    last_seen_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    observation_count = Column(
        Integer,
        nullable=False,
        default=0,
    )

    status = Column(
        String(30),
        nullable=False,
        default="ACTIVE",
        index=True,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    observations = relationship(
        "VehicleObservation",
        back_populates="vehicle",
        cascade="all, delete-orphan",
    )

    embeddings = relationship(
        "VehicleEmbedding",
        back_populates="vehicle",
        cascade="all, delete-orphan",
    )

    plates = relationship(
        "PlateObservation",
        back_populates="vehicle",
    )

    __table_args__ = (
        Index(
            "idx_vehicle_type_last_seen",
            "vehicle_type",
            "last_seen_at",
        ),
    )


# ============================================================
# VEHICLE OBSERVATION
# ============================================================

class VehicleObservation(Base):
    __tablename__ = "vehicle_observations"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    vehicle_id = Column(
        BigInteger,
        ForeignKey(
            "vehicles.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    global_vehicle_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    camera_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    local_track_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    vehicle_type = Column(
        String(50),
        nullable=True,
        index=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    frame_timestamp = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    inference_timestamp = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    bbox = Column(
        JSON,
        nullable=True,
    )

    crop_storage_key = Column(
        String(500),
        nullable=True,
    )

    model_name = Column(
        String(150),
        nullable=True,
    )

    model_version = Column(
        String(100),
        nullable=True,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    vehicle = relationship(
        "Vehicle",
        back_populates="observations",
    )

    __table_args__ = (
        Index(
            "idx_vehicle_obs_camera_time",
            "camera_id",
            "frame_timestamp",
        ),
        Index(
            "idx_vehicle_obs_global_time",
            "global_vehicle_id",
            "frame_timestamp",
        ),
    )


# ============================================================
# VEHICLE EMBEDDING
# ============================================================

class VehicleEmbedding(Base):
    __tablename__ = "vehicle_embeddings"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    vehicle_id = Column(
        BigInteger,
        ForeignKey(
            "vehicles.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    observation_id = Column(
        BigInteger,
        ForeignKey(
            "vehicle_observations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    model_name = Column(
        String(150),
        nullable=False,
    )

    model_version = Column(
        String(100),
        nullable=False,
    )

    dimension = Column(
        Integer,
        nullable=False,
    )

    embedding = Column(
        LargeBinary,
        nullable=False,
    )

    quality_score = Column(
        Float,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    vehicle = relationship(
        "Vehicle",
        back_populates="embeddings",
    )


# ============================================================
# PLATE OBSERVATION
# ============================================================

class PlateObservation(Base):
    __tablename__ = "plate_observations"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    vehicle_id = Column(
        BigInteger,
        ForeignKey(
            "vehicles.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    camera_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    local_track_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    plate_text = Column(
        String(50),
        nullable=False,
        index=True,
    )

    normalized_plate = Column(
        String(50),
        nullable=False,
        index=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    frame_timestamp = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    model_name = Column(
        String(150),
        nullable=True,
    )

    model_version = Column(
        String(100),
        nullable=True,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    vehicle = relationship(
        "Vehicle",
        back_populates="plates",
    )

    __table_args__ = (
        Index(
            "idx_plate_camera_time",
            "camera_id",
            "frame_timestamp",
        ),
        Index(
            "idx_plate_normalized_time",
            "normalized_plate",
            "frame_timestamp",
        ),
    )


# ============================================================
# CORRELATION DECISION
# ============================================================

class CorrelationDecision(Base):
    __tablename__ = "correlation_decisions"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    source_camera_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    source_track_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    source_global_vehicle_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    candidate_global_vehicle_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    reid_score = Column(
        Float,
        nullable=True,
    )

    plate_score = Column(
        Float,
        nullable=True,
    )

    vehicle_type_score = Column(
        Float,
        nullable=True,
    )

    track_quality_score = Column(
        Float,
        nullable=True,
    )

    temporal_score = Column(
        Float,
        nullable=True,
    )

    gis_score = Column(
        Float,
        nullable=True,
    )

    direction_score = Column(
        Float,
        nullable=True,
    )

    final_score = Column(
        Float,
        nullable=True,
        index=True,
    )

    decision = Column(
        String(30),
        nullable=False,
        index=True,
    )

    reason = Column(
        Text,
        nullable=True,
    )

    model_name = Column(
        String(150),
        nullable=True,
    )

    model_version = Column(
        String(100),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    __table_args__ = (
        Index(
            "idx_correlation_source_time",
            "source_camera_id",
            "created_at",
        ),
        Index(
            "idx_correlation_decision_time",
            "decision",
            "created_at",
        ),
    )


# ============================================================
# GLOBAL PERSON
# ============================================================

class Person(Base):
    __tablename__ = "persons"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    user_id = Column(BigInteger, nullable=False, index=True)

    global_person_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    first_seen_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    last_seen_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    observation_count = Column(
        Integer,
        nullable=False,
        default=0,
    )

    status = Column(
        String(30),
        nullable=False,
        default="ACTIVE",
        index=True,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
    )

    observations = relationship(
        "PersonObservation",
        back_populates="person",
        cascade="all, delete-orphan",
    )


# ============================================================
# PERSON OBSERVATION
# ============================================================

class PersonObservation(Base):
    __tablename__ = "person_observations"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    user_id = Column(BigInteger, nullable=False, index=True)

    person_id = Column(
        BigInteger,
        ForeignKey(
            "persons.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    global_person_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    camera_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    local_track_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    frame_timestamp = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    inference_timestamp = Column(
        DateTime,
        nullable=False,
        default=indian_time,
    )

    bbox = Column(
        JSON,
        nullable=True,
    )

    pose = Column(
        JSON,
        nullable=True,
    )

    model_name = Column(
        String(150),
        nullable=True,
    )

    model_version = Column(
        String(100),
        nullable=True,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    person = relationship(
        "Person",
        back_populates="observations",
    )

    __table_args__ = (
        Index(
            "idx_person_obs_camera_time",
            "camera_id",
            "frame_timestamp",
        ),
        Index(
            "idx_person_obs_global_time",
            "global_person_id",
            "frame_timestamp",
        ),
    )


class PersonCorrelationDecision(Base):
    __tablename__ = "person_correlation_decisions"

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    camera_id = Column(String(100), nullable=False, index=True)
    local_track_id = Column(String(100), nullable=False, index=True)
    global_person_id = Column(String(100), nullable=False, index=True)
    candidate_global_person_id = Column(String(100), nullable=True, index=True)
    appearance_score = Column(Float, nullable=True)
    temporal_score = Column(Float, nullable=True)
    topology_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=False, index=True)
    decision = Column(String(30), nullable=False, index=True)
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)

    __table_args__ = (
        Index("idx_person_corr_user_time", "user_id", "created_at"),
        Index("idx_person_corr_global_time", "global_person_id", "created_at"),
    )


class InvestigationSummary(Base):
    __tablename__ = "investigation_summaries"

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    incident_id = Column(BigInteger, nullable=True, index=True)
    subject_type = Column(String(30), nullable=False, index=True)
    subject_id = Column(String(100), nullable=False, index=True)
    source_version = Column(String(100), nullable=False)
    model_name = Column(String(150), nullable=False)
    status = Column(String(30), nullable=False, default="PENDING", index=True)
    summary_text = Column(Text, nullable=True)
    structured_summary = Column(JSON, nullable=True)
    error_code = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_summary_subject", "user_id", "subject_type", "subject_id"),
    )


# ============================================================
# ACTIVITY EVENT
# ============================================================

class ActivityEvent(Base):
    __tablename__ = "activity_events"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    event_id = Column(
        String(120),
        unique=True,
        nullable=False,
        index=True,
    )

    camera_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    track_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    global_person_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    global_vehicle_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    rule_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    event_type = Column(
        String(100),
        nullable=False,
        index=True,
    )

    severity = Column(
        String(30),
        nullable=False,
        default="LOW",
        index=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    status = Column(
        String(30),
        nullable=False,
        default="CONFIRMED",
        index=True,
    )

    started_at = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    ended_at = Column(
        DateTime,
        nullable=True,
    )

    evidence = Column(
        JSON,
        nullable=True,
    )

    model_name = Column(
        String(150),
        nullable=True,
    )

    model_version = Column(
        String(100),
        nullable=True,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    __table_args__ = (
        Index(
            "idx_activity_camera_time",
            "camera_id",
            "started_at",
        ),
        Index(
            "idx_activity_rule_time",
            "rule_id",
            "started_at",
        ),
    )


# ============================================================
# MODEL EXECUTION
# ============================================================

class ModelExecution(Base):
    __tablename__ = "model_executions"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    model_name = Column(
        String(150),
        nullable=False,
        index=True,
    )

    model_version = Column(
        String(100),
        nullable=False,
        index=True,
    )

    model_hash = Column(
        String(128),
        nullable=True,
    )

    camera_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    track_id = Column(
        String(100),
        nullable=True,
    )

    inference_timestamp = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    frame_timestamp = Column(
        DateTime,
        nullable=True,
        index=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    latency_ms = Column(
        Float,
        nullable=True,
    )

    device = Column(
        String(50),
        nullable=True,
    )

    metadata_json = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )
