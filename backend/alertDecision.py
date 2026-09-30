import math

from config import MIN_ALERT_CONFIDENCE, RULE_LEVEL


VALID_ALERT_LEVELS = {
    "CRITICAL",
    "WARNING",
    "MEDIUM",
    "LOW",
}


def get_alert_level(rule: str, confidence: float = 0.0) -> str:
    level = RULE_LEVEL.get(rule, "LOW")

    if level not in VALID_ALERT_LEVELS:
        return "LOW"

    return level


def final_alert_allowed(alert, zone_result):
    rule = alert.get("rule")
    try:
        confidence = float(alert.get("confidence", 0))
    except (ValueError, TypeError):
        return False, "invalid_confidence"
    if not math.isfinite(confidence):
        return False, "invalid_confidence"

    if not rule:
        return False, "missing_rule"

    if rule not in RULE_LEVEL:
        return False, "unknown_rule"

    if confidence < MIN_ALERT_CONFIDENCE.get(rule, 0.50):
        return False, "low_confidence"

    if not zone_result or not zone_result.get("allowed"):
        return False, "zone_not_allowed"

    if alert.get("track_id") is None and alert.get("track") is None:
        return False, "missing_track"

    box = alert.get("box")
    try:
        if box is None or len(box) != 4:
            return False, "missing_box"
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return False, "invalid_box"
    except (TypeError, ValueError):
        return False, "invalid_box"

    alert["level"] = get_alert_level(rule, confidence)

    return True, "allowed"