from datetime import datetime, timezone
from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Index, Integer, JSON, LargeBinary, String, Text, UniqueConstraint
from db.database import Base


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ExternalWatchlistSource(Base):
    __tablename__ = "external_watchlist_sources"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    owner_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    description = Column(Text)
    provider_type = Column(String(30), nullable=False, default="REST_JSON")
    base_url = Column(String(2048), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    read_only = Column(Boolean, nullable=False, default=True)
    auth_type = Column(String(30), nullable=False, default="NONE")
    credential_ciphertext = Column(LargeBinary)
    api_key_header = Column(String(100))
    entity_types = Column(JSON, nullable=False, default=list)
    field_mapping = Column(JSON, nullable=False, default=dict)
    records_path = Column(String(250), nullable=False, default="records")
    sync_mode = Column(String(20), nullable=False, default="INTERVAL")
    sync_interval_seconds = Column(Integer, nullable=False, default=300)
    pagination = Column(JSON, nullable=False, default=dict)
    tls_verify = Column(Boolean, nullable=False, default=True)
    status = Column(String(30), nullable=False, default="NEVER_SYNCED")
    last_sync_at = Column(DateTime)
    last_success_at = Column(DateTime)
    last_error = Column(String(500))
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    __table_args__ = (Index("ix_external_watchlist_due", "enabled", "last_sync_at"),)


class ExternalWatchlistRecord(Base):
    __tablename__ = "external_watchlist_records"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(BigInteger, ForeignKey("external_watchlist_sources.id", ondelete="CASCADE"), nullable=False)
    owner_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    external_id = Column(String(200), nullable=False)
    entity_type = Column(String(20), nullable=False)
    category = Column(String(50), nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    priority = Column(String(20))
    case_reference = Column(String(200))
    name = Column(String(150))
    image_available = Column(Boolean, nullable=False, default=False)
    plate_display = Column(String(50))
    plate_normalized = Column(String(50), index=True)
    make = Column(String(100))
    model = Column(String(100))
    color = Column(String(60))
    source_name = Column(String(150), nullable=False)
    raw_metadata = Column(JSON, nullable=False, default=dict)
    local_watchlist_entry_id = Column(BigInteger)
    last_synced_at = Column(DateTime, nullable=False, default=utcnow)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    __table_args__ = (UniqueConstraint("source_id", "external_id", name="uq_external_watchlist_source_record"), Index("ix_external_watchlist_owner_type_active", "owner_user_id", "entity_type", "active"))


class ExternalWatchlistSyncHistory(Base):
    __tablename__ = "external_watchlist_sync_history"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(BigInteger, ForeignKey("external_watchlist_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    actor_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    action = Column(String(30), nullable=False)
    result = Column(String(30), nullable=False)
    records_received = Column(Integer, nullable=False, default=0)
    records_created = Column(Integer, nullable=False, default=0)
    records_updated = Column(Integer, nullable=False, default=0)
    records_unchanged = Column(Integer, nullable=False, default=0)
    records_rejected = Column(Integer, nullable=False, default=0)
    duration_ms = Column(Integer, nullable=False, default=0)
    message = Column(String(500))
    created_at = Column(DateTime, nullable=False, default=utcnow)
