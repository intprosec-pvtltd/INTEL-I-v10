from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class Observation:
    event_id: str
    camera_id: str
    timestamp: float
    entity_key: str
    event_type: str
    confidence: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Incident:
    incident_id: str
    created_at: float
    updated_at: float
    entity_key: str
    events: List[Observation] = field(default_factory=list)
    risk_score: float = 0.0
    severity: str = "INFO"


class IncidentCorrelator:
    """
    Lightweight in-memory incident fusion.
    Group observations by entity/camera/time. Persist incidents in your DB
    for durable operation.
    """

    def __init__(self, max_gap_seconds: float = 180.0):
        self.max_gap_seconds = float(max_gap_seconds)
        self.incidents: Dict[str, Incident] = {}

    def _incident_key(self, entity_key: str, event: Observation) -> str:
        return f"{entity_key}:{event.camera_id}"

    def ingest(self, event: Observation) -> Incident:
        candidates = []
        for incident in self.incidents.values():
            if incident.entity_key != event.entity_key:
                continue
            if event.timestamp - incident.updated_at <= self.max_gap_seconds:
                candidates.append(incident)

        if candidates:
            incident = max(candidates, key=lambda x: x.updated_at)
        else:
            incident = Incident(
                incident_id=f"INC-{int(event.timestamp*1000)}-{len(self.incidents)+1}",
                created_at=event.timestamp,
                updated_at=event.timestamp,
                entity_key=event.entity_key,
            )
            self.incidents[incident.incident_id] = incident

        incident.events.append(event)
        incident.updated_at = max(incident.updated_at, event.timestamp)
        self._recalculate(incident)
        return incident

    @staticmethod
    def _recalculate(incident: Incident) -> None:
        conf = [max(0.0, min(1.0, e.confidence)) for e in incident.events]
        unique_types = {e.event_type for e in incident.events}
        base = min(100.0, sum(conf) * 18.0)
        diversity_bonus = min(35.0, len(unique_types) * 6.0)
        cross_camera = len({e.camera_id for e in incident.events}) > 1
        cross_bonus = 15.0 if cross_camera else 0.0
        score = min(100.0, base + diversity_bonus + cross_bonus)
        incident.risk_score = round(score, 2)

        if score >= 85:
            incident.severity = "CRITICAL"
        elif score >= 65:
            incident.severity = "HIGH"
        elif score >= 40:
            incident.severity = "MEDIUM"
        elif score >= 20:
            incident.severity = "LOW"
        else:
            incident.severity = "INFO"

    def get(self, incident_id: str) -> Optional[Incident]:
        return self.incidents.get(incident_id)

    def all(self) -> List[Incident]:
        return sorted(
            self.incidents.values(),
            key=lambda x: x.updated_at,
            reverse=True,
        )
