from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class AudioEvent:
    event: str
    score: float
    anomaly: bool
    metrics: Dict[str, float]


class AudioIntelligence:
    """
    Dependency-light audio anomaly gate.

    It detects strong impulsive energy/anomalies. It is NOT a semantic
    gunshot/scream classifier. Plug a licensed ONNX audio classifier into
    classify_semantic() when those labels are required.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = int(sample_rate)
        self.baseline_rms = 0.01

    def analyze_pcm(self, pcm: np.ndarray) -> AudioEvent:
        if not isinstance(pcm, np.ndarray) or pcm.size == 0:
            return AudioEvent("invalid", 0.0, False, {})

        x = pcm.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        peak = float(np.max(np.abs(x)))

        # Smooth baseline.
        self.baseline_rms = self.baseline_rms * 0.98 + rms * 0.02
        impulse_ratio = rms / max(self.baseline_rms, 1e-6)

        anomaly = impulse_ratio >= 4.0 or peak >= 0.95
        score = min(1.0, max(0.0, (impulse_ratio - 1.0) / 8.0))

        return AudioEvent(
            event="impulsive_audio" if anomaly else "normal",
            score=round(score, 4),
            anomaly=anomaly,
            metrics={
                "rms": round(rms, 6),
                "peak": round(peak, 6),
                "impulse_ratio": round(impulse_ratio, 4),
            },
        )

    def classify_semantic(self, pcm: np.ndarray) -> Dict[str, float]:
        # Model hook. Add ONNX Runtime classifier here.
        return {"unknown": 1.0}
