from __future__ import annotations

from dataclasses import dataclass
from typing import Final


_ALLOWED_CLASSIFICATIONS: Final[dict[str, tuple[str, str]]] = {
    "RESTRICTED": (
        "RESTRICTED - INVESTIGATION USE",
        "GUJARAT POLICE • RESTRICTED",
    ),
    "CONFIDENTIAL": (
        "CONFIDENTIAL - AUTHORISED USE ONLY",
        "GUJARAT POLICE • CONFIDENTIAL",
    ),
    "INTERNAL": (
        "INTERNAL USE",
        "GUJARAT POLICE • INTERNAL USE",
    ),
    "DRAFT": (
        "DRAFT - NOT FOR EVIDENTIARY USE",
        "GUJARAT POLICE • DRAFT",
    ),
}


@dataclass(frozen=True)
class ReportPolicy:
    classification_code: str
    classification_label: str
    watermark_text: str
    evidence_purpose: str
    handling_notice: str


def resolve_report_policy(classification: str | None) -> ReportPolicy:
    code = str(classification or "RESTRICTED").strip().upper()
    if code not in _ALLOWED_CLASSIFICATIONS:
        allowed = ", ".join(sorted(_ALLOWED_CLASSIFICATIONS))
        raise ValueError(f"Unsupported evidence-report classification. Allowed: {allowed}")
    label, watermark = _ALLOWED_CLASSIFICATIONS[code]
    return ReportPolicy(
        classification_code=code,
        classification_label=label,
        watermark_text=watermark,
        evidence_purpose=(
            "This report records digital evidence associated with an INTEL-I incident or alert. "
            "It is a system-generated analytical evidence record and must be reviewed by an authorised investigator "
            "before operational or evidentiary reliance."
        ),
        handling_notice=(
            "This document contains system-generated investigative information. Do not alter, redact, or distribute it "
            "outside the authorised case workflow. Original media is retained separately from derived report exhibits."
        ),
    )


def allowed_classifications() -> list[str]:
    return sorted(_ALLOWED_CLASSIFICATIONS)
