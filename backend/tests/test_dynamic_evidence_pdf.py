from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image
try:
    from PyPDF2 import PdfReader
except ImportError:
    from pypdf import PdfReader

from evidence.images import prepare_snapshot_for_report
from evidence.integrity import sha256_bytes
from evidence.pdf_renderer import render_incident_evidence_pdf
from evidence.policy import resolve_report_policy
from evidence.schemas import EvidenceReportData, Exhibit, TimelineItem


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (320, 180), (40, 70, 100)).save(output, format="JPEG", quality=90)
    return output.getvalue()


def _sample_data(snapshot: bytes) -> EvidenceReportData:
    now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    return EvidenceReportData(
        report_id="INTEL-I-20260908T100000Z-ABCDEF12",
        export_id="EXP-ABCDEF123456",
        incident_id=151,
        generated_at=now,
        generated_by="Authorised Officer",
        classification="RESTRICTED - INVESTIGATION USE",
        classification_code="RESTRICTED",
        case_id="INC-151",
        report_version="2.0",
        scope="Vehicle / ANPR / Watchlist / Multi-camera correlation / journey",
        incident_type="VEHICLE_WATCHLIST_MATCH",
        incident_status="OPEN",
        severity="HIGH",
        description="Exact watchlist-associated vehicle evidence was recorded and requires authorised verification.",
        evaluation_status="PENDING",
        evaluation_confidence=0.98,
        evaluation_reason="Stored system assessment.",
        camera_ids=["CAM-A", "CAM-B"],
        camera_names=["Camera A", "Camera B"],
        locations=["Location A", "Location B"],
        started_at=now,
        ended_at=None,
        primary_track_id="TRACK-3",
        global_vehicle_id="GV-123",
        plate_candidates=["KA03NA8694"],
        watchlist_results=["ACTIVE - STOLEN"],
        timeline=[TimelineItem(now, "WATCHLIST_EXACT_MATCH", "CAM-A", "Camera A", "KA03NA8694 / ACTIVE - STOLEN", 0.98, "SOURCE_FRAME_TIME", "SOURCE_FRAME_TIME", 12.5, 151, 1)],
        journey=[{"timestamp": now.isoformat(), "camera_id": "CAM-A", "camera_name": "Camera A", "location": "Location A", "plate": "KA03NA8694", "confidence": 0.98}],
        exhibits=[Exhibit("E-001", "Vehicle watchlist alert", 1, 151, 10, now, "CAM-A", "Camera A", "Location A", "Vehicle - plate KA03NA8694", "WATCHLIST_EXACT_MATCH", "HIGH", 0.98, "ACTIVE - STOLEN", "Derived report exhibit", sha256_bytes(snapshot), snapshot, "image/jpeg", {})],
        limitations=["Original source media must be reviewed before evidentiary reliance."],
        provenance_rows=[("Original evidence", "Retained separately in the INTEL-I evidence store.")],
        source_manifest_sha256="a" * 64,
        model_context=["anpr_model: test-model"],
    )


def test_report_policy_rejects_unknown_classification():
    try:
        resolve_report_policy("PUBLIC")
    except ValueError:
        return
    raise AssertionError("unknown classification must be rejected")


def test_snapshot_preparation_preserves_source_hash_and_bounds_output():
    source = _jpeg()
    prepared, media_type, digest = prepare_snapshot_for_report(source, "image/jpeg")
    assert prepared
    assert media_type == "image/jpeg"
    assert digest == sha256_bytes(source)
    with Image.open(BytesIO(prepared)) as image:
        assert image.width <= 3000
        assert image.height <= 3000


def test_dynamic_evidence_pdf_contains_watermark_report_and_integrity_text():
    source = _jpeg()
    data = _sample_data(source)
    logo = Path(__file__).resolve().parents[1] / "assets" / "gujarat_police_logo.png"
    payload = render_incident_evidence_pdf(data, logo)
    assert payload.startswith(b"%PDF")
    assert len(payload) > 5000

    reader = PdfReader(BytesIO(payload))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "DIGITAL EVIDENCE REPORT" in text
    assert "GUJARAT POLICE" in text
    assert "RESTRICTED" in text
    assert "KA03NA8694" in text
    assert "Source evidence manifest" in text
    assert data.report_id in text
    assert data.export_id in text
