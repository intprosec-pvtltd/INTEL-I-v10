from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CameraIntegrationAdapter(ABC):
    """Stable adapter contract used by the onboarding control plane."""

    @abstractmethod
    def discover_cameras(self) -> list[dict[str, Any]]: ...

    def get_camera_metadata(self, camera: dict[str, Any]) -> dict[str, Any]:
        return dict(camera)

    def get_stream_urls(self, camera: dict[str, Any]) -> dict[str, str]:
        return {key: str(camera[key]) for key in ("main_stream_url", "substream_url", "rtsp_url") if camera.get(key)}

    def validate_camera(self, camera: dict[str, Any]) -> bool:
        return bool(self.get_stream_urls(camera))

    def sync_cameras(self) -> list[dict[str, Any]]:
        return self.discover_cameras()
