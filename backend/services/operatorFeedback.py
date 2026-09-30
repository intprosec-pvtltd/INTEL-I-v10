from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List


@dataclass
class FeedbackRecord:
    alert_id: str
    decision: str
    operator_id: str
    note: str
    created_at: str


class OperatorFeedbackStore:
    """
    Runtime store. In production, persist these records in PostgreSQL.
    """

    VALID_DECISIONS = {
        "confirmed",
        "false_positive",
        "dismissed",
        "needs_review",
    }

    def __init__(self):
        self._lock = Lock()
        self._records: List[FeedbackRecord] = []

    def add(self, alert_id: str, decision: str, operator_id: str, note: str = "") -> FeedbackRecord:
        decision = decision.strip().lower()
        if decision not in self.VALID_DECISIONS:
            raise ValueError(f"Unsupported feedback decision: {decision}")

        record = FeedbackRecord(
            alert_id=str(alert_id),
            decision=decision,
            operator_id=str(operator_id),
            note=str(note)[:2000],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._records.append(record)

        return record

    def summary(self) -> Dict[str, int]:
        result = {k: 0 for k in self.VALID_DECISIONS}
        with self._lock:
            for record in self._records:
                result[record.decision] = result.get(record.decision, 0) + 1
        return result

    def all(self) -> List[dict]:
        with self._lock:
            return [asdict(x) for x in self._records]
