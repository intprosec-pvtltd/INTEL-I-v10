import json
from datetime import timedelta
import os
from db.database import SessionLocal
from db.model import OutboxEvent, indian_time

OUTBOX_STATUSES_RETRYABLE = {"PENDING", "FAILED"}


def init_outbox():
    return True


def add_alert_event(topic: str, event_key: str | None, payload: dict):
    db = SessionLocal()

    try:
        event = OutboxEvent(
            topic=topic,
            event_key=event_key,
            payload=json.dumps(payload),
            status="PENDING",
            attempts=0,
            next_attempt_at=indian_time(),
        )

        db.add(event)
        db.commit()
        db.refresh(event)

        return event.id

    finally:
        db.close()


def get_retryable_events(limit: int = 50, max_attempts: int = 5):
    db = SessionLocal()

    try:
        now = indian_time()

        stale_seconds = max(30, int(os.getenv("OUTBOX_PROCESSING_TIMEOUT_SECONDS", "120")))
        stale_before = now - timedelta(seconds=stale_seconds)
        events = (
            db.query(OutboxEvent)
            .filter(
                (
                    OutboxEvent.status.in_(["PENDING", "FAILED"])
                    & (OutboxEvent.next_attempt_at <= now)
                )
                | (
                    (OutboxEvent.status == "PROCESSING")
                    & (OutboxEvent.updated_at <= stale_before)
                ),
                OutboxEvent.attempts < max_attempts,
            )
            .order_by(OutboxEvent.created_at.asc())
            .limit(limit)
            .all()
        )

        for event in events:
            event.status = "PROCESSING"
            event.updated_at = indian_time()

        db.commit()

        return [event.id for event in events]

    finally:
        db.close()


def get_outbox_event(event_id: int):
    db = SessionLocal()

    try:
        return db.query(OutboxEvent).filter(OutboxEvent.id == event_id).first()

    finally:
        db.close()


def mark_event_sent(event_id: int):
    db = SessionLocal()

    try:
        event = db.query(OutboxEvent).filter(OutboxEvent.id == event_id).first()

        if event:
            event.status = "SENT"
            event.updated_at = indian_time()
            db.commit()

    finally:
        db.close()


def mark_event_failed(
    event_id: int,
    error: Exception,
    max_attempts: int = 5,
    retry_delay_seconds: int = 10,
):
    db = SessionLocal()

    try:
        event = db.query(OutboxEvent).filter(OutboxEvent.id == event_id).first()

        if not event:
            return

        event.attempts += 1
        event.last_error = str(error)[:1000]
        event.updated_at = indian_time()

        if event.attempts >= max_attempts:
            event.status = "DEAD"
        else:
            event.status = "FAILED"
            event.next_attempt_at = indian_time() + timedelta(
                seconds=retry_delay_seconds * (2 ** max(0, event.attempts - 1))
            )

        db.commit()

    finally:
        db.close()