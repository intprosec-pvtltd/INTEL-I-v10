"""Versioned Kafka event contracts. Video/frame bytes are prohibited."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    CAMERA_DETECTION = "camera.detection"
    PERSON_TRACK = "person.track"
    VEHICLE_TRACK = "vehicle.track"
    ANPR_READ = "anpr.read"
    FRS_CANDIDATE = "frs.candidate"
    FRS_CONFIRMED = "frs.confirmed"
    WATCHLIST_MATCH = "watchlist.match"
    CORRELATION_MATCH = "correlation.match"
    ALERT_CREATED = "alert.created"
    CAMERA_HEALTH = "camera.health"
    WORKER_HEALTH = "worker.health"


class IntelIEvent(BaseModel):
    schema_version: str = "1.0"
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    camera_id: str
    track_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    worker_id: str
    model_version: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_uri: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def prohibit_frame_payloads(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = {"frame", "image", "jpeg", "video", "bytes", "base64"}
        keys = {str(key).lower() for key in value}
        overlap = keys & forbidden
        if overlap:
            raise ValueError(f"binary media must use evidence_uri, forbidden keys: {sorted(overlap)}")
        return value
