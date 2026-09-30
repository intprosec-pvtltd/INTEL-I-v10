from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConnectorResult:
    """Normalized connector output.

    A connector may resolve to a normal network URI or provide a native
    frame source. The AI pipeline never depends on the connector type.
    """

    source: Any
    source_type: str
    metadata: dict[str, Any]


class CameraConnector(ABC):
    connector_type: str = "unknown"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def resolve(self) -> ConnectorResult:
        raise NotImplementedError

    def health_check(self) -> dict[str, Any]:
        return {"ok": True}

    def close(self) -> None:
        return None
