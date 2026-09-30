from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Any, Callable


@dataclass
class ModelRecord:
    name: str
    model: Any = None
    provider: Callable[[], Any] | None = None
    health_provider: Callable[[], dict[str, Any]] | None = None
    version: str = "current"
    role: str = "model"
    device: str | None = None
    required: bool = False
    external: bool = False
    registered_at: float = 0.0


class ModelRegistry:
    """Central registry for executable models and remote model services.

    In central mode this registry lives inside the single GPU inference worker.
    Remote FRS/OCR services are represented as external records rather than
    duplicated into every camera worker.
    """

    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        model: Any,
        version: str | None = None,
        *,
        role: str = "model",
        device: str | None = None,
        required: bool = False,
    ) -> Any:
        key = str(name).strip().lower()
        with self._lock:
            existing = self._records.get(key)
            if existing and existing.model is not None and existing.model is not model:
                raise RuntimeError(f"duplicate model registration: {key}")
            self._records[key] = ModelRecord(
                name=key,
                model=model,
                version=version or os.getenv(f"{key.upper()}_MODEL_VERSION", "current"),
                role=role,
                device=device,
                required=bool(required),
                external=False,
                registered_at=time.time(),
            )
        return model

    def register_provider(
        self,
        name: str,
        provider: Callable[[], Any] | None,
        *,
        health_provider: Callable[[], dict[str, Any]] | None = None,
        version: str | None = None,
        role: str = "model",
        device: str | None = None,
        required: bool = False,
        external: bool = False,
    ) -> None:
        key = str(name).strip().lower()
        with self._lock:
            self._records[key] = ModelRecord(
                name=key,
                provider=provider,
                health_provider=health_provider,
                version=version or os.getenv(f"{key.upper()}_MODEL_VERSION", "current"),
                role=role,
                device=device,
                required=bool(required),
                external=bool(external),
                registered_at=time.time(),
            )

    def get(self, name: str, *, load: bool = True) -> Any:
        key = str(name).strip().lower()
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if record.model is None and load and record.provider is not None:
                record.model = record.provider()
            return record.model

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        with self._lock:
            records = list(self._records.items())
        for key, record in records:
            health: dict[str, Any] = {}
            if record.health_provider is not None:
                try:
                    health = dict(record.health_provider() or {})
                except Exception as exc:
                    health = {"healthy": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
            loaded = record.model is not None
            if record.external:
                loaded = bool(health.get("loaded") or health.get("healthy") or health.get("ready"))
            result[key] = {
                "loaded": loaded,
                "version": record.version,
                "role": record.role,
                "device": record.device,
                "required": record.required,
                "external": record.external,
                "health": health,
            }
        return result

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._records)


REGISTRY = ModelRegistry()
