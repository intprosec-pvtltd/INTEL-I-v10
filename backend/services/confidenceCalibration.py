from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class CalibrationConfig:
    temperature: float = 1.0
    quality_weight: float = 0.25
    temporal_weight: float = 0.20
    min_quality_floor: float = 0.20


class ConfidenceCalibrator:
    """
    Lightweight quality-aware confidence calibration.

    For production, learn temperature and weights from labeled validation
    data per model/camera. This class provides the runtime mechanism.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()

    def calibrate(
        self,
        raw_confidence: float,
        *,
        quality_score: float = 1.0,
        temporal_support: float = 0.0,
    ) -> float:
        p = max(1e-6, min(1-1e-6, float(raw_confidence)))
        q = max(
            self.config.min_quality_floor,
            min(1.0, float(quality_score)),
        )
        t = max(0.0, min(1.0, float(temporal_support)))

        # Logit temperature scaling.
        logit = math.log(p / (1 - p))
        scaled = 1.0 / (1.0 + math.exp(-logit / max(0.1, self.config.temperature)))

        adjusted = (
            scaled * (1 - self.config.quality_weight - self.config.temporal_weight)
            + q * self.config.quality_weight
            + t * self.config.temporal_weight
        )
        return round(max(0.0, min(1.0, adjusted)), 4)
