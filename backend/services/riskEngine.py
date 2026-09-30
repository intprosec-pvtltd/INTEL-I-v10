from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class EvidenceSignal:
    name: str
    weight: float
    confidence: float = 1.0
    enabled: bool = True
    detail: str = ""


@dataclass
class RiskResult:
    score: float
    severity: str
    signals: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""


class RiskEngine:
    """
    Deterministic, explainable risk scoring.
    Score is 0..100 and is intentionally auditable.
    """

    def __init__(self):
        self.weights = {
            "watchlist_match": 40,
            "restricted_zone": 20,
            "loitering": 15,
            "wrong_way": 18,
            "camera_tamper": 30,
            "behavior_anomaly": 20,
            "crowd_anomaly": 20,
            "audio_anomaly": 15,
            "prior_incident": 10,
            "cross_camera_match": 12,
            "plate_stable": 5,
        }

    def score(self, signals: Iterable[EvidenceSignal]) -> RiskResult:
        total = 0.0
        rows = []

        for signal in signals:
            if not signal.enabled:
                continue

            weight = float(signal.weight)
            contribution = max(0.0, min(weight, weight * float(signal.confidence)))
            total += contribution

            rows.append({
                "name": signal.name,
                "weight": weight,
                "confidence": float(signal.confidence),
                "contribution": round(contribution, 3),
                "detail": signal.detail,
            })

        total = min(100.0, total)

        if total >= 85:
            severity = "CRITICAL"
        elif total >= 65:
            severity = "HIGH"
        elif total >= 40:
            severity = "MEDIUM"
        elif total >= 20:
            severity = "LOW"
        else:
            severity = "INFO"

        contributing = [
            f"{r['name']} (+{r['contribution']:.1f})"
            for r in rows
            if r["contribution"] > 0
        ]

        explanation = (
            "Risk driven by: " + ", ".join(contributing)
            if contributing else
            "No significant risk signals."
        )

        return RiskResult(
            score=round(total, 2),
            severity=severity,
            signals=rows,
            explanation=explanation,
        )

    def from_payload(self, payload: Dict[str, Any]) -> RiskResult:
        signals = []

        for name, weight in self.weights.items():
            raw = payload.get(name)

            if isinstance(raw, dict):
                enabled = bool(raw.get("enabled", True))
                confidence = float(raw.get("confidence", raw.get("score", 0.0)))
                detail = str(raw.get("detail", ""))
            else:
                enabled = bool(raw)
                confidence = 1.0 if raw else 0.0
                detail = ""

            signals.append(
                EvidenceSignal(
                    name=name,
                    weight=weight,
                    confidence=max(0.0, min(1.0, confidence)),
                    enabled=enabled,
                    detail=detail,
                )
            )

        return self.score(signals)
