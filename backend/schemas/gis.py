from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CameraConnectionStatus = Literal["active", "inactive"]


class GISBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraConnectionCreate(GISBaseModel):
    source_camera_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    destination_camera_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    distance_km: float | None = Field(default=None, ge=0, le=10000)
    expected_min_seconds: float | None = Field(default=None, ge=0, le=86400)
    expected_max_seconds: float | None = Field(default=None, ge=0, le=86400)
    expected_seconds: float | None = Field(default=None, ge=0, le=86400)
    road_name: str | None = Field(default=None, max_length=255)
    direction: float | None = Field(default=None, ge=0, lt=360)
    direction_tolerance: float | None = Field(default=None, ge=0, le=180)
    road_distance_km: float | None = Field(default=None, ge=0, le=10000)
    road_duration_seconds: float | None = Field(default=None, ge=0, le=86400)
    route_geometry: str | None = Field(default=None, max_length=200000)
    routing_provider: str | None = Field(default=None, max_length=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    active: bool = True


class CameraConnectionResponse(GISBaseModel):
    id: int
    source_camera_id: str
    destination_camera_id: str
    distance_km: float | None = None
    expected_min_seconds: float | None = None
    expected_max_seconds: float | None = None
    expected_seconds: float | None = None
    road_name: str | None = None
    direction: float | None = None
    direction_tolerance: float | None = None
    road_distance_km: float | None = None
    road_duration_seconds: float | None = None
    route_geometry: str | None = None
    routing_provider: str | None = None
    confidence: float | None = None
    active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CameraConnectionListResponse(GISBaseModel):
    count: int
    connections: list[CameraConnectionResponse]


class GISTransitionEvaluationRequest(GISBaseModel):
    source_camera_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    destination_camera_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    source_timestamp: float = Field(ge=0, le=4102444800)
    destination_timestamp: float = Field(ge=0, le=4102444800)
    source_heading: float | None = Field(default=None, ge=0, lt=360)
    destination_heading: float | None = Field(default=None, ge=0, lt=360)
    min_speed_kmh: float = Field(default=2.0, ge=0, le=200)
    max_speed_kmh: float = Field(default=180.0, ge=1, le=300)


class GISTransitionEvaluationResponse(GISBaseModel):
    source_camera_id: str
    destination_camera_id: str
    gis_available: bool
    distance_meters: float | None = None
    bearing_degrees: float | None = None
    elapsed_seconds: float | None = None
    implied_speed_kmh: float | None = None
    travel_time_score: float
    direction_score: float
    distance_score: float
    gis_score: float
    feasible: bool
    reason: str


class GISJourneyObservation(GISBaseModel):
    observation_id: int
    camera_id: str
    vehicle_type: str | None = None
    confidence: float | None = None
    frame_timestamp: datetime
    latitude: float | None = None
    longitude: float | None = None
    local_track_id: str | None = None


class GISJourneySegment(GISBaseModel):
    from_camera_id: str
    to_camera_id: str
    from_timestamp: datetime
    to_timestamp: datetime
    elapsed_seconds: float
    distance_km: float | None = None
    bearing_degrees: float | None = None
    feasible: bool
    gis_score: float
    direction_score: float
    travel_time_score: float
    reason: str


class GISVehicleJourneyResponse(GISBaseModel):
    global_vehicle_id: str
    observation_count: int
    transition_count: int
    observations: list[GISJourneyObservation]
    segments: list[GISJourneySegment]
