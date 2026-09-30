"""Small Redis Pub/Sub bridge for worker -> API realtime UI events."""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Callable

from redis.exceptions import RedisError

from security.redisClient import redis_client, make_key

logger = logging.getLogger(__name__)
CHANNEL = make_key(os.getenv("INTEL_I_REALTIME_CHANNEL", "realtime:events"))

_listener_thread: threading.Thread | None = None
_stop_event = threading.Event()


def publish_realtime_event(payload: dict) -> bool:
    if redis_client is None:
        return False
    if not isinstance(payload, dict):
        return False
    try:
        redis_client.publish(CHANNEL, json.dumps(payload, default=str))
        return True
    except (RedisError, TypeError, ValueError):
        logger.exception("Realtime event publish failed")
        return False


def start_realtime_subscriber(callback: Callable[[dict], None]) -> None:
    global _listener_thread
    if redis_client is None:
        logger.warning("Realtime Redis subscriber disabled because Redis is unavailable")
        return
    if _listener_thread and _listener_thread.is_alive():
        return
    _stop_event.clear()

    def _loop():
        while not _stop_event.is_set():
            pubsub = None
            try:
                pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(CHANNEL)
                while not _stop_event.is_set():
                    message = pubsub.get_message(timeout=1.0)
                    if not message:
                        continue
                    raw = message.get("data")
                    try:
                        payload = json.loads(raw)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        logger.warning("Realtime event ignored because payload is invalid JSON")
                        continue
                    if not isinstance(payload, dict):
                        continue
                    try:
                        callback(payload)
                    except Exception:
                        logger.exception("Realtime event callback failed")
            except RedisError:
                if not _stop_event.is_set():
                    logger.exception("Realtime Redis subscriber failed")
            finally:
                try:
                    pubsub.close()
                except Exception:
                    pass
            _stop_event.wait(1.0)

    _listener_thread = threading.Thread(
        target=_loop,
        name="intel-i-realtime-redis-subscriber",
        daemon=True,
    )
    _listener_thread.start()


def stop_realtime_subscriber() -> None:
    _stop_event.set()
    thread = _listener_thread
    if thread and thread.is_alive():
        thread.join(timeout=3.0)
