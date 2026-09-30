from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class TimelineItem:
    event_time: datetime | None
    event_type: str
    camera_id: str | None = None
    camera_name: str | None = None
    result: str | None = None
    confidence: float | None = None
    timestamp_source: str | None = None
    timestamp_quality: str | None = None
    source_pts_seconds: float | None = None
    alert_id: int | None = None
    evidence_id: int | None = None


@dataclass
class Exhibit:
    exhibit_id: str
    title: str
    evidence_id: int
    alert_id: int | None
    snapshot_id: int | None
    event_time: datetime | None
    camera_id: str | None
    camera_name: str | None
    location: str | None
    subject: str | None
    rule: str | None
    severity: str | None
    confidence: float | None
    watchlist_result: str | None
    description: str | None
    snapshot_sha256: str | None
    image_bytes: bytes | None = field(repr=False, default=None)
    image_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceReportData:
    report_id: str
    export_id: str
    incident_id: int
    generated_at: datetime
    generated_by: str
    classification: str
    classification_code: str
    case_id: str
    report_version: str
    scope: str
    incident_type: str
    incident_status: str
    severity: str
    description: str
    evaluation_status: str | None
    evaluation_confidence: float | None
    evaluation_reason: str | None
    camera_ids: list[str]
    camera_names: list[str]
    locations: list[str]
    started_at: datetime | None
    ended_at: datetime | None
    primary_track_id: str | None
    global_vehicle_id: str | None
    plate_candidates: list[str]
    watchlist_results: list[str]
    timeline: list[TimelineItem]
    journey: list[dict[str, Any]]
    exhibits: list[Exhibit]
    limitations: list[str]
    provenance_rows: list[tuple[str, str]]
    source_manifest_sha256: str
    model_context: list[str]
    approver: str = "Pending investigating officer review"

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for exhibit in payload.get("exhibits", []):
            exhibit.pop("image_bytes", None)
            exhibit.pop("metadata", None)
        return payload
