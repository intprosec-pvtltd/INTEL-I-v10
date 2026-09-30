"""Load-aware, deterministic camera-to-stream-worker placement helper."""
from __future__ import annotations

import hashlib
from typing import Any


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def worker_load_score(worker: dict[str, Any]) -> float:
    capacity = max(1.0, _number(worker.get("capacity"), 1.0))
    owned = max(0.0, _number(worker.get("owned_cameras"), 0.0))
    occupancy = min(1.5, owned / capacity)
    cpu = min(1.0, max(0.0, _number(worker.get("cpu_percent"), 0.0) / 100.0))
    memory = min(1.0, max(0.0, _number(worker.get("memory_percent"), 0.0) / 100.0))
    q_depth = max(0.0, _number(worker.get("queue_depth"), 0.0))
    q_cap = max(1.0, _number(worker.get("queue_capacity"), 1.0))
    queue = min(1.0, q_depth / q_cap)
    # Occupancy is the strongest signal; CPU/RAM protect overloaded stream
    # nodes; queue pressure is a final tie-breaker.
    return occupancy * 0.55 + cpu * 0.20 + memory * 0.15 + queue * 0.10


def preferred_worker_id(camera_pk: int, workers: list[dict[str, Any]]) -> str | None:
    eligible = []
    for worker in workers:
        worker_id = str(worker.get("worker_id") or "").strip()
        if not worker_id:
            continue
        status = str(worker.get("status") or "").upper()
        capacity = max(0, int(_number(worker.get("capacity"), 0)))
        owned = max(0, int(_number(worker.get("owned_cameras"), 0)))
        if status not in {"READY", "DEGRADED", "RUNNING"} or capacity <= owned:
            continue
        digest = hashlib.sha256(f"{int(camera_pk)}:{worker_id}".encode("utf-8")).digest()
        tie = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        eligible.append((worker_load_score(worker), tie, worker_id))
    if not eligible:
        return None
    # Least loaded first; deterministic hashing prevents all equal workers from
    # racing for the same camera.
    eligible.sort(key=lambda row: (row[0], row[1], row[2]))
    return eligible[0][2]
