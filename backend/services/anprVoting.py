from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import re


PLATE_RE = re.compile(r"^[A-Z0-9]{5,12}$")


@dataclass
class OCRCandidate:
    text: str
    confidence: float
    source: str = ""


def normalize_plate(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(text or "")).upper()
    return text


def plate_format_score(text: str) -> float:
    t = normalize_plate(text)
    if not t:
        return 0.0
    if PLATE_RE.fullmatch(t):
        return 1.0
    return 0.5 if 5 <= len(t) <= 12 else 0.1


def character_vote(candidates: Iterable[OCRCandidate]) -> Tuple[str, float]:
    rows = [
        OCRCandidate(normalize_plate(x.text), float(x.confidence), x.source)
        for x in candidates
        if normalize_plate(x.text)
    ]

    if not rows:
        return "", 0.0

    target_len = max(
        set(len(x.text) for x in rows),
        key=lambda n: sum(1 for x in rows if len(x.text) == n),
    )

    aligned = [x for x in rows if len(x.text) == target_len]
    if not aligned:
        aligned = rows

    result = []
    total_score = 0.0

    for i in range(target_len):
        votes = defaultdict(float)
        for row in aligned:
            ch = row.text[i]
            weight = max(0.0, min(1.0, row.confidence))
            votes[ch] += weight

        if not votes:
            result.append("?")
            continue

        char, weight = max(votes.items(), key=lambda kv: kv[1])
        result.append(char)
        total_score += weight / max(1, len(aligned))

    text = "".join(result)
    score = min(
        1.0,
        total_score * 0.8 + plate_format_score(text) * 0.2,
    )
    return text, round(score, 4)
