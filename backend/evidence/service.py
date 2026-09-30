from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from .collector import collect_incident_report_data
from .integrity import sha256_bytes
from .pdf_renderer import render_incident_evidence_pdf


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOGO = _BACKEND_ROOT / "assets" / "gujarat_police_logo.png"


def build_incident_evidence_report(
    db: Session,
    *,
    incident_id: int,
    user_id: int,
    generated_by: str,
    classification: str | None = None,
):
    data = collect_incident_report_data(
        db,
        incident_id=incident_id,
        user_id=user_id,
        generated_by=generated_by,
        classification=classification,
    )
    pdf = render_incident_evidence_pdf(data, _DEFAULT_LOGO)
    return data, pdf, sha256_bytes(pdf)
