import os
import time
import json
import threading
import logging

from events.outbox import (
    get_retryable_events,
    get_outbox_event,
    mark_event_sent,
    mark_event_failed,
)

from events.kafkaProducer import produce_event, kafka_status

logger = logging.getLogger(__name__)

OUTBOX_MAX_ATTEMPTS = int(os.getenv("OUTBOX_MAX_ATTEMPTS", "5"))
OUTBOX_BATCH_SIZE = int(os.getenv("OUTBOX_BATCH_SIZE", "50"))
OUTBOX_RETRY_BASE_SECONDS = int(os.getenv("OUTBOX_RETRY_BASE_SECONDS", "10"))
OUTBOX_IDLE_SLEEP_SECONDS = int(os.getenv("OUTBOX_IDLE_SLEEP_SECONDS", "3"))
OUTBOX_ERROR_SLEEP_SECONDS = int(os.getenv("OUTBOX_ERROR_SLEEP_SECONDS", "5"))

_worker_thread = None
_stop_event = threading.Event()


def _safe_json_loads(payload: str):
    try:
        return json.loads(payload)
    except Exception as exc:
        raise ValueError("Invalid outbox JSON payload") from exc


def _process_event(event_id: int):
    event = get_outbox_event(event_id)

    if not event:
        return

    try:
        payload = _safe_json_loads(event.payload)

        produce_event(
            topic=event.topic,
            key=event.event_key,
            value=payload,
        )

        mark_event_sent(event_id)

    except Exception as exc:
        logger.exception("Outbox event failed | event_id=%s", event_id)

        mark_event_failed(
            event_id=event_id,
            error=exc,
            max_attempts=OUTBOX_MAX_ATTEMPTS,
            retry_delay_seconds=OUTBOX_RETRY_BASE_SECONDS,
        )


def _outbox_loop():
    logger.info("Outbox worker started")

    while not _stop_event.is_set():
        try:
            if not kafka_status():
                logger.warning("Kafka unavailable. Outbox worker waiting.")
                time.sleep(OUTBOX_ERROR_SLEEP_SECONDS)
                continue

            event_ids = get_retryable_events(
                limit=OUTBOX_BATCH_SIZE,
                max_attempts=OUTBOX_MAX_ATTEMPTS,
            )

            if not event_ids:
                time.sleep(OUTBOX_IDLE_SLEEP_SECONDS)
                continue

            for event_id in event_ids:
                if _stop_event.is_set():
                    break

                _process_event(event_id)

        except Exception:
            logger.exception("Outbox worker loop failed")
            time.sleep(OUTBOX_ERROR_SLEEP_SECONDS)

    logger.info("Outbox worker stopped")


def start_outbox_worker():
    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        return

    _stop_event.clear()

    _worker_thread = threading.Thread(
        target=_outbox_loop,
        name="outbox-worker",
        daemon=True,
    )

    _worker_thread.start()


def stop_outbox_worker():
    _stop_event.set()

    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=5)