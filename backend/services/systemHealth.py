from __future__ import annotations

import os
import time
import subprocess
from typing import Any

import psutil
from sqlalchemy import text
from core.camera_state import CameraState


def _status(ok: bool) -> str:
    return "ONLINE" if ok else "OFFLINE"


def check_database(db) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        db.execute(text("SELECT 1"))
        latency_ms = (time.perf_counter() - started) * 1000

        return {
            "status": "ONLINE",
            "healthy": True,
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as exc:
        return {
            "status": "OFFLINE",
            "healthy": False,
            "latency_ms": None,
            "error": str(exc)[:200],
        }


def check_redis(redis_ping) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        ok = bool(redis_ping())
        latency_ms = (time.perf_counter() - started) * 1000

        return {
            "status": _status(ok),
            "healthy": ok,
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as exc:
        return {
            "status": "OFFLINE",
            "healthy": False,
            "latency_ms": None,
            "error": str(exc)[:200],
        }


def check_kafka(kafka_status) -> dict[str, Any]:
    enabled = os.getenv("KAFKA_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        return {
            "status": "DISABLED",
            "healthy": True,
            "enabled": False,
            "latency_ms": None,
        }
    started = time.perf_counter()

    try:
        ok = bool(kafka_status())
        latency_ms = (time.perf_counter() - started) * 1000

        return {
            "status": _status(ok),
            "healthy": ok,
            "enabled": True,
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as exc:
        return {
            "status": "OFFLINE",
            "healthy": False,
            "latency_ms": None,
            "error": str(exc)[:200],
        }


def check_gpu() -> dict[str, Any]:
    """
    Uses nvidia-smi when available.

    Does not fail the whole health endpoint if NVIDIA tooling is unavailable.
    """

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.total,memory.used,"
                "temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

        if result.returncode != 0:
            return {
                "status": "UNAVAILABLE",
                "healthy": False,
                "available": False,
                "reason": "nvidia-smi unavailable",
            }

        line = result.stdout.strip().splitlines()[0]

        gpu_util, memory_total, memory_used, temperature = [
            float(x.strip())
            for x in line.split(",")
        ]

        memory_percent = (
            (memory_used / memory_total) * 100
            if memory_total > 0
            else 0
        )

        return {
            "status": "ONLINE",
            "healthy": True,
            "available": True,
            "utilization_percent": round(gpu_util, 2),
            "memory_total_mb": round(memory_total, 2),
            "memory_used_mb": round(memory_used, 2),
            "memory_percent": round(memory_percent, 2),
            "temperature_c": round(temperature, 2),
        }

    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "healthy": False,
            "available": False,
            "error": str(exc)[:200],
        }


def check_host_resources() -> dict[str, Any]:
    return {
        "cpu_percent": round(
            psutil.cpu_percent(interval=0.1),
            2,
        ),
        "ram_percent": round(
            psutil.virtual_memory().percent,
            2,
        ),
        "ram_total_gb": round(
            psutil.virtual_memory().total / (1024 ** 3),
            2,
        ),
        "disk_percent": round(
            psutil.disk_usage(os.getcwd()).percent,
            2,
        ),
    }


def build_system_health(
    *,
    db,
    redis_ping,
    kafka_status,
    camera_states: dict[str, str],
    active_camera_workers: int,
    websocket_clients: int,
    camera_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:

    database = check_database(db)
    redis = check_redis(redis_ping)
    kafka = check_kafka(kafka_status)
    gpu = check_gpu()
    host = check_host_resources()

    cameras = []
    for value in camera_states.values():
        try:
            cameras.append(value.value if isinstance(value, CameraState) else CameraState(str(value).upper()).value)
        except ValueError:
            cameras.append(CameraState.OFFLINE.value)

    camera_counts = {
        "total": len(cameras),
        "online": cameras.count(CameraState.ONLINE.value),
        "degraded": cameras.count(CameraState.DEGRADED.value),
        "reconnecting": cameras.count(CameraState.RECONNECTING.value),
        "offline": cameras.count(CameraState.OFFLINE.value),
        "authentication_failed": cameras.count(CameraState.AUTHENTICATION_FAILED.value),
        "timeout": cameras.count(CameraState.TIMEOUT.value),
        "ai_disabled": cameras.count(CameraState.AI_DISABLED.value),
    }

    critical_services = {
        "database": database["healthy"],
        "redis": redis["healthy"],
        "kafka": kafka["healthy"],
    }

    overall = (
        "HEALTHY"
        if all(critical_services.values())
        else "DEGRADED"
    )

    # Camera operational health is independent from infrastructure health.
    # A degraded/offline camera must not falsely mark PostgreSQL/Redis/Kafka
    # offline, but it must be visible to operators.
    return {
        "overall": overall,
        "timestamp": time.time(),

        "backend": {
            "status": "ONLINE",
            "healthy": True,
        },

        "database": database,

        "redis": redis,

        "kafka": kafka,

        "outbox": {
            "status": "ONLINE" if kafka["healthy"] else "DEGRADED",
            "healthy": kafka["healthy"],
        },

        "websocket": {
            "status": "ONLINE",
            "healthy": True,
            "active_clients": websocket_clients,
        },

        "cameras": {
            "status": (
                "ONLINE"
                if camera_counts["offline"] == 0
                and camera_counts["reconnecting"] == 0
                and camera_counts["authentication_failed"] == 0
                and camera_counts["timeout"] == 0
                else "DEGRADED"
            ),
            "healthy": camera_counts["offline"] == 0 and camera_counts["authentication_failed"] == 0 and camera_counts["timeout"] == 0,
            "active_workers": active_camera_workers,
            **camera_counts,
            "details": camera_details or [],
        },

        "ai": {
            "status": "ONLINE",
            "healthy": True,
            "gpu": gpu,
        },

        "host": host,
    }
