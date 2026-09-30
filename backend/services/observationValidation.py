"""Production observation validation for INTEL-I Phase 9.

This module is deliberately conservative: AI output is treated as an
observation, not as a fact. Low-quality, malformed, stale, impossible, or
ambiguous observations are rejected or downgraded before correlation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Any


def _float(name: str, default: float, low: float, high: float) -> float:
    try:
        v = float(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        v = default
    if not math.isfinite(v):
        v = default
    return max(low, min(high, v))


MIN_DETECTION_CONFIDENCE = _float("OBS_MIN_DETECTION_CONFIDENCE", 0.35, 0.0, 1.0)
MIN_TRACK_QUALITY = _float("OBS_MIN_TRACK_QUALITY", 0.35, 0.0, 1.0)
MIN_FRAME_QUALITY = _float("OBS_MIN_FRAME_QUALITY", 0.20, 0.0, 1.0)
MAX_OBSERVATION_AGE_SECONDS = _float("OBS_MAX_AGE_SECONDS", 10.0, 0.5, 120.0)
MIN_BBOX_AREA_RATIO = _float("OBS_MIN_BBOX_AREA_RATIO", 0.0001, 0.0, 0.25)
MAX_BBOX_AREA_RATIO = _float("OBS_MAX_BBOX_AREA_RATIO", 0.95, 0.1, 1.0)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    confidence: float
    quality: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _finite01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0.0, min(1.0, x)) if math.isfinite(x) else default


def _bbox_valid(bbox: Any, width: int, height: int) -> tuple[bool, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False, 0.0
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False, 0.0
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return False, 0.0
    # Reject wildly out-of-frame boxes; small tolerance handles decoder rounding.
    if x2 < -2 or y2 < -2 or x1 > width + 2 or y1 > height + 2:
        return False, 0.0
    area_ratio = max(0.0, min(1.0, ((x2 - x1) * (y2 - y1)) / float(width * height)))
    return MIN_BBOX_AREA_RATIO <= area_ratio <= MAX_BBOX_AREA_RATIO, area_ratio


def validate_vehicle_observation(
    observation: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    now_monotonic: float | None = None,
) -> ValidationResult:
    if not isinstance(observation, dict):
        return ValidationResult(False, 0.0, 0.0, ("OBSERVATION_NOT_OBJECT",), ())

    reasons: list[str] = []
    warnings: list[str] = []
    now = time.monotonic() if now_monotonic is None else float(now_monotonic)

    camera_id = str(observation.get("camera_id") or "").strip()
    track_id = str(observation.get("track_id") or "").strip()
    if not camera_id or len(camera_id) > 256:
        reasons.append("INVALID_CAMERA_ID")
    if not track_id or len(track_id) > 64:
        reasons.append("INVALID_TRACK_ID")

    confidence = _finite01(observation.get("confidence"), 0.0)
    track_quality = _finite01(observation.get("track_quality"), 0.0)
    frame_quality = _finite01(observation.get("frame_quality_score"), 1.0)
    bbox_ok, area_ratio = _bbox_valid(observation.get("bbox"), frame_width, frame_height)
    if not bbox_ok:
        reasons.append("INVALID_BBOX")
    elif area_ratio < MIN_BBOX_AREA_RATIO:
        reasons.append("TINY_BBOX")

    if confidence < MIN_DETECTION_CONFIDENCE:
        reasons.append("LOW_DETECTION_CONFIDENCE")
    if track_quality < MIN_TRACK_QUALITY:
        reasons.append("LOW_TRACK_QUALITY")
    if frame_quality < MIN_FRAME_QUALITY:
        reasons.append("LOW_FRAME_QUALITY")

    submitted = observation.get("observation_monotonic")
    if submitted is not None:
        try:
            age = now - float(submitted)
            if not math.isfinite(age) or age < -1.0:
                reasons.append("INVALID_OBSERVATION_TIME")
            elif age > MAX_OBSERVATION_AGE_SECONDS:
                reasons.append("STALE_OBSERVATION")
        except (TypeError, ValueError, OverflowError):
            reasons.append("INVALID_OBSERVATION_TIME")

    # An OCR string without meaningful confidence is not allowed to strengthen
    # identity. It remains a weak observation for later human review.
    plate = str(observation.get("plate") or "").strip().upper()
    if plate:
        plate_conf = _finite01(observation.get("plate_confidence"), 0.0)
        det_conf = _finite01(observation.get("plate_detection_confidence"), 0.0)
        if plate_conf < 0.50 or det_conf < 0.40:
            warnings.append("LOW_PLATE_EVIDENCE")

    # Camera tamper/frozen frames are never accepted as strong identity evidence.
    if bool(observation.get("camera_tampered")):
        warnings.append("CAMERA_TAMPER")
    if bool(observation.get("stream_frozen")):
        warnings.append("STREAM_FROZEN")

    # The calibrated confidence is bounded by the evidence quality. This keeps
    # a high raw detector score from becoming a high-confidence identity.
    evidence_quality = (
        0.50 * confidence
        + 0.25 * track_quality
        + 0.25 * frame_quality
    )
    if warnings:
        evidence_quality *= 0.75
    final_confidence = max(0.0, min(1.0, evidence_quality))

    return ValidationResult(
        accepted=not reasons,
        confidence=round(final_confidence, 4),
        quality=round(max(0.0, min(1.0, evidence_quality)), 4),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
