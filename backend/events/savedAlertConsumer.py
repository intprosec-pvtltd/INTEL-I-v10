from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Awaitable, Callable

from confluent_kafka import Consumer
from db.database import SessionLocal
from services.openSearchIntelligence import opensearch_intelligence

logger = logging.getLogger("saved-alert-consumer")
_thread: threading.Thread | None = None
_stop = threading.Event()


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _config() -> dict:
    cfg = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "group.id": os.getenv("KAFKA_SAVED_ALERT_CONSUMER_GROUP", "intel-i-saved-alert-realtime-v1"),
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    }
    if cfg["security.protocol"] in {"SASL_SSL", "SASL_PLAINTEXT"}:
        cfg.update({
            "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
            "sasl.username": os.getenv("KAFKA_SASL_USERNAME", ""),
            "sasl.password": os.getenv("KAFKA_SASL_PASSWORD", ""),
        })
    ca = os.getenv("KAFKA_SSL_CA_LOCATION", "").strip()
    if ca:
        cfg["ssl.ca.location"] = ca
    return cfg


def _normalize(value: dict) -> dict:
    # Supports both outbox payloads and legacy CCTV_ALERT_SAVED wrappers.
    if value.get("event_type") == "CCTV_ALERT_SAVED" and isinstance(value.get("data"), dict):
        payload = dict(value["data"])
        payload.setdefault("alert_id", value.get("alert_id"))
        return payload
    return dict(value)


def start_saved_alert_consumer(*, loop: asyncio.AbstractEventLoop, websocket_sender: Callable[[dict], Awaitable[None]]) -> None:
    global _thread
    if not _bool("KAFKA_SAVED_ALERT_WS_ENABLED", True):
        logger.info("Saved-alert Kafka realtime consumer disabled")
        return
    if _thread and _thread.is_alive():
        return
    topic = os.getenv("KAFKA_SAVED_ALERT_TOPIC", "cctv.alerts.saved")
    _stop.clear()

    def run() -> None:
        consumer = Consumer(_config())
        consumer.subscribe([topic])
        logger.info("Saved-alert consumer started | topic=%s", topic)
        try:
            while not _stop.is_set():
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error("Saved-alert Kafka error | %s", msg.error())
                    continue
                try:
                    raw = json.loads(msg.value().decode("utf-8"))
                    payload = _normalize(raw)
                    alert_id = payload.get("alert_id") or payload.get("id")
                    user_id = payload.get("user_id")
                    if not alert_id or not user_id:
                        raise ValueError("saved alert missing alert_id/user_id")
                    # Search indexing is off the camera hot path and idempotent by ID.
                    if opensearch_intelligence.enabled:
                        with SessionLocal() as db:
                            opensearch_intelligence.index_alert_by_id(db, user_id=int(user_id), alert_id=int(alert_id))
                    future = asyncio.run_coroutine_threadsafe(websocket_sender(payload), loop)
                    future.result(timeout=max(2.0, float(os.getenv("KAFKA_WS_DELIVERY_TIMEOUT_SECONDS", "8"))))
                    consumer.commit(msg, asynchronous=False)
                except Exception:
                    # Do not commit: Kafka will redeliver after recovery. Frontend also
                    # has DB recovery by alert ID, so duplicate WS delivery is harmless.
                    logger.exception("Saved-alert delivery failed; offset not committed")
        finally:
            consumer.close()
            logger.info("Saved-alert consumer stopped")

    _thread = threading.Thread(target=run, name="saved-alert-realtime-consumer", daemon=True)
    _thread.start()


def stop_saved_alert_consumer() -> None:
    _stop.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=5)
