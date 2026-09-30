from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator

CATEGORIES = {'STOLEN', 'BLACKLISTED', 'SUSPICIOUS', 'WANTED', 'OTHER'}
STATUSES = {'ACTIVE', 'CLEARED', 'EXPIRED', 'DISABLED'}
PRIORITIES = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}

class WatchlistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: str | None = Field(default=None, max_length=4000)
    is_active: bool = True

class WatchlistUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    description: str | None = Field(default=None, max_length=4000)
    is_active: bool | None = None

class WatchlistEntryCreate(BaseModel):
    plate: str = Field(min_length=2, max_length=50)
    category: str = 'SUSPICIOUS'
    status: str = 'ACTIVE'
    priority: str = 'HIGH'
    description: str | None = Field(default=None, max_length=4000)
    source: str | None = Field(default=None, max_length=150)
    effective_from: datetime | None = None
    effective_until: datetime | None = None
    metadata: dict | None = None

    @field_validator('category')
    @classmethod
    def validate_category(cls, value):
        value = value.strip().upper()
        if value not in CATEGORIES:
            raise ValueError(f'category must be one of: {", ".join(sorted(CATEGORIES))}')
        return value

    @field_validator('status')
    @classmethod
    def validate_status(cls, value):
        value = value.strip().upper()
        if value not in STATUSES:
            raise ValueError(f'status must be one of: {", ".join(sorted(STATUSES))}')
        return value

    @field_validator('priority')
    @classmethod
    def validate_priority(cls, value):
        value = value.strip().upper()
        if value not in PRIORITIES:
            raise ValueError(f'priority must be one of: {", ".join(sorted(PRIORITIES))}')
        return value

class WatchlistEntryUpdate(BaseModel):
    plate: str | None = Field(default=None, min_length=2, max_length=50)
    category: str | None = None
    status: str | None = None
    priority: str | None = None
    description: str | None = Field(default=None, max_length=4000)
    source: str | None = Field(default=None, max_length=150)
    effective_from: datetime | None = None
    effective_until: datetime | None = None
    metadata: dict | None = None

class WatchlistMatchRequest(BaseModel):
    plate: str = Field(min_length=2, max_length=50)
    include_possible: bool = True

class WatchlistEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    watchlist_id: int
    plate: str
    plate_normalized: str
    category: str
    status: str
    priority: str
    severity: str = 'LOW'
    description: str | None
    source: str | None
    effective_from: datetime | None
    effective_until: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime

class WatchlistOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
