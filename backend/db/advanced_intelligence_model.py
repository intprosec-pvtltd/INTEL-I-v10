from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from db.database import Base


def indian_time():
    """Backward-compatible name; database timestamps are canonical UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PersonWatchlistEntry(Base):
    __tablename__ = "person_watchlist_entries"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    full_name = Column(
        String(150),
        nullable=False,
    )

    category = Column(
        String(30),
        nullable=False,
        index=True,
    )

    status = Column(
        String(30),
        nullable=False,
        default="ACTIVE",
        index=True,
    )

    description = Column(
        Text,
        nullable=True,
    )

    source = Column(
        String(150),
        nullable=True,
    )

    external_reference = Column(
        String(150),
        nullable=True,
    )

    embedding_encrypted = Column(
        LargeBinary,
        nullable=False,
    )

    embedding_dimension = Column(
        Integer,
        nullable=False,
    )

    face_model = Column(
        String(150),
        nullable=False,
    )

    face_model_version = Column(
        String(100),
        nullable=False,
    )

    # ========================================================
    # REFERENCE IMAGE
    # ========================================================

    reference_image_data = Column(
        LargeBinary,
        nullable=True,
    )

    reference_image_content_type = Column(
        String(100),
        nullable=True,
    )

    reference_image_filename = Column(
        String(255),
        nullable=True,
    )

    reference_image_size = Column(
        Integer,
        nullable=True,
    )

    # ========================================================
    # METADATA
    # ========================================================

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
        onupdate=indian_time,
        index=True,
    )

    __table_args__ = (
        Index(
            "idx_person_watchlist_user_status",
            "user_id",
            "status",
        ),

        Index(
            "idx_person_watchlist_user_category",
            "user_id",
            "category",
        ),
    )


class IncidentEvidence(Base):
    __tablename__ = "incident_evidence"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    incident_id = Column(
        Integer,
        ForeignKey(
            "incidents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    alert_id = Column(
        Integer,
        ForeignKey(
            "alerts.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    snapshot_id = Column(
        Integer,
        ForeignKey(
            "snapshots.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    evidence_type = Column(
        String(50),
        nullable=False,
        index=True,
    )

    object_type = Column(
        String(30),
        nullable=True,
        index=True,
    )

    object_reference = Column(
        String(150),
        nullable=True,
        index=True,
    )

    description = Column(
        Text,
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

    incident = relationship(
        "Incident",
        back_populates="evidence_items",
    )

    __table_args__ = (
        Index(
            "idx_incident_evidence_user_incident",
            "user_id",
            "incident_id",
        ),

        Index(
            "idx_incident_evidence_user_type",
            "user_id",
            "evidence_type",
        ),
    )
