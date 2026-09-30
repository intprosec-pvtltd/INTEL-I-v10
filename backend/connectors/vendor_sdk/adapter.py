from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from connectors.base import CameraConnector, ConnectorResult


class VendorSDKAdapter(CameraConnector, ABC):
    """Contract implemented by each approved vendor SDK adapter.

    SDKs that expose native frames should return a FrameSource from
    resolve().source. FrameSource must implement read() -> (bool, frame)
    and release().
    """

    connector_type = "vendor_sdk"

    @abstractmethod
    def resolve(self) -> ConnectorResult:
        raise NotImplementedError
