from datetime import datetime, timezone
from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Boolean, Text, ForeignKey, Float, Index, JSON
from sqlalchemy.orm import relationship
from db.database import Base

def indian_time():
    """Backward-compatible name; database timestamps are canonical UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Watchlist(Base):
    __tablename__ = 'watchlists'

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)
    updated_at = Column(DateTime, nullable=False, default=indian_time, index=True)

    entries = relationship('WatchlistEntry', back_populates='watchlist', cascade='all, delete-orphan')

    __table_args__ = (Index('idx_watchlist_user_active', 'user_id', 'is_active'),)

class WatchlistEntry(Base):
    __tablename__ = 'watchlist_entries'

    id = Column(BigInteger, primary_key=True, index=True)
    watchlist_id = Column(BigInteger, ForeignKey('watchlists.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    plate_normalized = Column(String(50), nullable=False, index=True)
    plate_display = Column(String(50), nullable=False)
    category = Column(String(50), nullable=False, default='SUSPICIOUS', index=True)
    status = Column(String(30), nullable=False, default='ACTIVE', index=True)
    priority = Column(String(20), nullable=False, default='HIGH', index=True)
    description = Column(Text, nullable=True)
    source = Column(String(150), nullable=True)
    effective_from = Column(DateTime, nullable=True, index=True)
    effective_until = Column(DateTime, nullable=True, index=True)
    version = Column(Integer, nullable=False, default=1)
    metadata_json = Column('metadata', JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=indian_time, index=True)
    updated_at = Column(DateTime, nullable=False, default=indian_time, index=True)

    watchlist = relationship('Watchlist', back_populates='entries')

    __table_args__ = (
        Index('idx_watchlist_entry_user_plate', 'user_id', 'plate_normalized'),
        Index('idx_watchlist_entry_user_category', 'user_id', 'category'),
        Index('idx_watchlist_entry_active_effective', 'user_id', 'status', 'effective_from', 'effective_until'),
    )
