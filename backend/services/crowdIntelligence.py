from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class CrowdResult:
    count: int
    density: float
    growth_rate: float
    anomaly: bool
    severity: str
    reason: str


class CrowdIntelligence:
    def __init__(self, history_size: int = 30, rapid_growth_per_second: float = 3.0):
        self.history = deque(maxlen=max(5, int(history_size)))
        self.rapid_growth_per_second = float(rapid_growth_per_second)

    def update(
        self,
        count: int,
        frame_area: int,
        now_seconds: float,
    ) -> CrowdResult:
        count = max(0, int(count))
        area = max(1, int(frame_area))
        density = count / (area / 10000.0)

        current = (float(now_seconds), count)
        self.history.append(current)

        growth = 0.0
        if len(self.history) >= 2:
            t0, c0 = self.history[0]
            t1, c1 = self.history[-1]
            dt = max(0.001, t1 - t0)
            growth = (c1 - c0) / dt

        anomaly = growth >= self.rapid_growth_per_second
        severity = "HIGH" if anomaly else "INFO"
        reason = "rapid_crowd_formation" if anomaly else "normal"

        return CrowdResult(
            count=count,
            density=round(density, 4),
            growth_rate=round(growth, 4),
            anomaly=anomaly,
            severity=severity,
            reason=reason,
        )
