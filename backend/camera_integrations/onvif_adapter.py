from __future__ import annotations

from typing import Any

from camera_integrations.base_adapter import CameraIntegrationAdapter
from connectors.onvif.connector import ONVIFConnector


class ONVIFAdapter(CameraIntegrationAdapter):
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)

    def discover_cameras(self) -> list[dict[str, Any]]:
        connector = ONVIFConnector(self.config)
        try:
            profiles = connector.discover_profiles()
            return [{"camera_id": self.config.get("camera_id") or self.config.get("host"), "camera_name": self.config.get("name") or self.config.get("host"), "ip_address": self.config.get("host"), "streams": profiles}]
        finally:
            connector.close()
