"""Stable catalogue contract shared by Sentinel and VMS/NVR adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StreamDescriptor:
    url: str
    source_type: str
    profile: str | None = None
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    transport: str | None = None
    priority: int = 100

    def safe_dict(self) -> dict[str, Any]:
        """Return stream metadata without the secret-bearing URL."""
        value = asdict(self)
        value.pop("url", None)
        value["available"] = True
        return value


@dataclass(frozen=True)
class CatalogueCamera:
    external_id: str
    name: str
    live_status: str
    streams: tuple[StreamDescriptor, ...]
    location_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    heading: float | None = None
    fov: float | None = None
    direction: str | None = None
    road_name: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    vendor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def preferred_stream(self) -> StreamDescriptor | None:
        if not self.streams:
            return None
        return min(self.streams, key=lambda item: (item.priority, item.source_type, item.profile or ""))

    def safe_dict(self) -> dict[str, Any]:
        """Return discovery data suitable for an authenticated UI."""
        return {
            "external_id": self.external_id,
            "name": self.name,
            "live_status": self.live_status,
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "heading": self.heading,
            "fov": self.fov,
            "direction": self.direction,
            "road_name": self.road_name,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "vendor": self.vendor,
            "streams": [item.safe_dict() for item in self.streams],
        }


@dataclass(frozen=True)
class CatalogueResult:
    cameras: tuple[CatalogueCamera, ...]
    fetched_pages: int
    source_revision: str | None = None
    etag: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.cameras),
            "fetched_pages": self.fetched_pages,
            "source_revision": self.source_revision,
            "cameras": [camera.safe_dict() for camera in self.cameras],
        }


class CameraCatalogueConnector(ABC):
    provider_type: str = "unknown"

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})

    @abstractmethod
    def fetch_catalogue(self) -> CatalogueResult:
        raise NotImplementedError

    def health_check(self) -> dict[str, Any]:
        try:
            result = self.fetch_catalogue()
            return {"ok": True, "camera_count": len(result.cameras)}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}

