from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    camera_id: str
    track_id: str
    timestamp: float
    image: np.ndarray
    detector_confidence: float
    landmarks: np.ndarray | None = None


@dataclass(frozen=True)
class FaceEmbedding:
    camera_id: str
    track_id: str
    timestamp: float
    vector: np.ndarray
    quality: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FaceMatch:
    identity_id: str
    similarity: float
    metadata: dict[str, Any]
