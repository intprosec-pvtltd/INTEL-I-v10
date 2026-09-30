"""Validated event publication facade."""
from __future__ import annotations

import json
from typing import Any, Callable

from .schemas import IntelIEvent


class EventPublisher:
    def __init__(self, producer: Any, topic_prefix: str = "intel-i") -> None:
        self.producer = producer
        self.topic_prefix = topic_prefix.strip(".")

    def publish(self, event: IntelIEvent) -> None:
        body = event.model_dump(mode="json")
        topic = f"{self.topic_prefix}.{event.event_type.value}"
        payload = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
        key = event.camera_id.encode("utf-8")
        if hasattr(self.producer, "produce"):
            self.producer.produce(topic, key=key, value=payload)
        elif callable(self.producer):
            self.producer(topic, key, payload)
        else:
            raise TypeError("producer must be callable or expose produce()")
