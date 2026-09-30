from __future__ import annotations
from typing import Any, Dict, List


def build_alert_explanation(
    rule: str,
    camera_id: str,
    confidence: float | None = None,
    evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    evidence = dict(evidence or {})
    reasons: List[str] = []

    if rule:
        reasons.append(f"Rule triggered: {rule}")

    if confidence is not None:
        reasons.append(f"Detection confidence: {float(confidence):.1%}")

    for key, label in (
        ("watchlist_match", "Watchlist match confirmed"),
        ("plate_stable", "Plate remained stable across observations"),
        ("cross_camera_match", "Same entity correlated across cameras"),
        ("restricted_zone", "Entity entered a restricted zone"),
        ("loitering", "Entity remained beyond the configured dwell threshold"),
        ("wrong_way", "Movement direction differs from the configured road direction"),
        ("camera_tamper", "Camera tampering was detected"),
    ):
        value = evidence.get(key)
        if isinstance(value, dict):
            value = value.get("enabled", value.get("score", False))
        if value:
            reasons.append(label)

    return {
        "rule": rule,
        "camera_id": camera_id,
        "confidence": confidence,
        "reasons": reasons,
        "evidence": evidence,
    }
