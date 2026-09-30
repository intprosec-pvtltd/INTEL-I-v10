from __future__ import annotations

from typing import Any

from camera_integrations.base_adapter import CameraIntegrationAdapter
from services.cameraIntegration import discover_integration


class SentinelAdapter(CameraIntegrationAdapter):
    def __init__(self, integration, transport=None):
        self.integration = integration
        self.transport = transport

    def discover_cameras(self) -> list[dict[str, Any]]:
        result = discover_integration(self.integration, transport=self.transport)
        cameras = []
        for descriptor in result.cameras:
            streams = descriptor.streams or []
            cameras.append({
                "camera_id": descriptor.external_id,
                "camera_name": descriptor.name,
                "location": descriptor.location_name,
                "latitude": descriptor.latitude,
                "longitude": descriptor.longitude,
                "vendor": descriptor.vendor or "Sentinel",
                "status": descriptor.live_status,
                "rtsp_url": streams[0].url if streams else None,
                "streams": [stream.safe_dict() for stream in streams],
            })
        return cameras
