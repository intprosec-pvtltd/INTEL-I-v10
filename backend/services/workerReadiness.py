"""Truthful readiness aggregation for analytics workers."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable


def _required(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ReadinessCheck:
    name: str
    probe: Callable[[], Any]
    required: bool = True


class WorkerReadiness:
    def __init__(self, runtime: Any, checks: list[ReadinessCheck] | None = None) -> None:
        self.runtime = runtime
        self.checks = checks or []

    @staticmethod
    def _normalise(value: Any) -> tuple[bool, dict[str, Any]]:
        if isinstance(value, dict):
            status = str(value.get("status", "")).upper()
            ok = bool(value.get("ready", value.get("available", value.get("loaded", False))))
            if status:
                ok = status in {"READY", "OK", "HEALTHY"}
            return ok, value
        return bool(value), {"value": bool(value)}

    def evaluate(self) -> dict[str, Any]:
        runtime_state = self.runtime.status()
        result: dict[str, Any] = {"analytics_runtime": runtime_state}
        required_ok = str(runtime_state.get("status")).upper() == "READY"
        for check in self.checks:
            try:
                ok, detail = self._normalise(check.probe())
            except Exception as exc:
                ok, detail = False, {"error": f"{type(exc).__name__}: {str(exc)[:240]}"}
            result[check.name] = {"ready": ok, "required": check.required, **detail}
            if check.required and not ok:
                required_ok = False
        result["status"] = "READY" if required_ok else "NOT_READY"
        return result


def default_worker_readiness(runtime: Any) -> WorkerReadiness:
    from db.database import SessionLocal
    from sqlalchemy import text
    from security.redisClient import redis_ping

    def database() -> bool:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return True

    checks = [
        ReadinessCheck("database", database, True),
        ReadinessCheck("redis", redis_ping, True),
    ]

    if _required("KAFKA_ENABLED", False):
        from events.kafkaProducer import kafka_status
        checks.append(ReadinessCheck("kafka", kafka_status, True))

    def remote_health(url_env: str, token_env: str, header: str) -> Callable[[], dict[str, Any]]:
        def probe() -> dict[str, Any]:
            import requests
            url = os.getenv(url_env, "").rstrip("/")
            token = os.getenv(token_env, "")
            if not url:
                return {"status": "NOT_READY", "error": f"{url_env} is not configured"}
            response = requests.get(
                f"{url}/internal/health", headers={header: token}, timeout=2.0
            )
            response.raise_for_status()
            return response.json()
        return probe

    checks.extend([
        ReadinessCheck(
            "anpr",
            remote_health("ANPR_OCR_REMOTE_URL", "ANPR_OCR_INTERNAL_TOKEN", "X-Intel-I-OCR-Token"),
            _required("ANPR_REQUIRED", False),
        ),
        ReadinessCheck(
            "frs",
            remote_health("FRS_REMOTE_URL", "FRS_INTERNAL_TOKEN", "X-Intel-I-FRS-Token"),
            _required("FRS_REQUIRED", False),
        ),
    ])
    return WorkerReadiness(runtime, checks)
