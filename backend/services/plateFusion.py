from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

PLATE_CLEAN = re.compile(r"[^A-Z0-9]")
INDIA_PATTERNS = (
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"),
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
)
SOURCE_WEIGHT = {"temporal": 1.0, "ppocr": 0.95, "ollama": 0.55, "candidate": 0.75}


def normalize_plate(value: str) -> str:
    return PLATE_CLEAN.sub("", str(value or "").upper())[:16]


def looks_like_indian_plate(value: str) -> bool:
    plate = normalize_plate(value)
    return any(pattern.fullmatch(plate) for pattern in INDIA_PATTERNS)


def fuse_plate_candidates(candidates: Iterable[dict]) -> dict:
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        plate = normalize_plate(item.get("text"))
        if len(plate) < 6:
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        source = str(item.get("source") or "candidate").lower()
        weighted = confidence * SOURCE_WEIGHT.get(source, 0.65)
        scores[plate] = scores.get(plate, 0.0) + weighted
        sources.setdefault(plate, set()).add(source)

    if not scores:
        return {"plate": None, "confidence": 0.0, "decision": "UNREADABLE", "sources": []}

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    plate, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    agreeing = sources[plate]
    format_bonus = 0.08 if looks_like_indian_plate(plate) else 0.0
    agreement_bonus = min(0.18, max(0, len(agreeing) - 1) * 0.09)
    confidence = min(1.0, score / max(1.0, len(agreeing)) + format_bonus + agreement_bonus)
    margin = max(0.0, score - second)
    # Ollama is advisory. It can never establish a confirmed result alone.
    non_ollama = agreeing - {"ollama"}
    if not non_ollama:
        decision = "POSSIBLE"
        confidence = min(confidence, 0.49)
    elif confidence >= 0.82 and margin >= 0.15 and len(agreeing) >= 2:
        decision = "CONFIRMED"
    elif confidence >= 0.62:
        decision = "PROBABLE"
    else:
        decision = "POSSIBLE"
    return {
        "plate": plate,
        "confidence": round(confidence, 4),
        "decision": decision,
        "sources": sorted(agreeing),
        "format_valid": looks_like_indian_plate(plate),
        "margin": round(margin, 4),
        "ranked": [{"plate": p, "score": round(s, 4)} for p, s in ranked[:5]],
    }
