"""Distributed camera ownership, worker heartbeat, and runtime telemetry.

This module is deliberately independent from the FastAPI process.  Camera
ownership is represented by short-lived Redis leases so a crashed analytics
worker cannot permanently strand a camera and a replacement cannot be started
while the old worker still proves ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
import uuid
from typing import Any

from redis.exceptions import RedisError

from security.redisClient import redis_client, make_key, set_json, get_json, delete_key, scan_keys


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


CAMERA_LEASE_TTL_SECONDS = _env_int(
    "INTEL_I_CAMERA_LEASE_TTL", 30, 10, 300
)
WORKER_HEARTBEAT_TTL_SECONDS = _env_int(
    "INTEL_I_WORKER_HEARTBEAT_TTL", 30, 10, 300
)
CAMERA_RUNTIME_TTL_SECONDS = _env_int(
    "INTEL_I_CAMERA_RUNTIME_TTL", 60, 15, 600
)


@dataclass(frozen=True)
class CameraLease:
    camera_pk: int
    cam_id: str
    worker_id: str
    value: str


def camera_lease_key(camera_pk: int) -> str:
    return f"camera:lease:{int(camera_pk)}"


def legacy_camera_id_lease_key(cam_id: str) -> str:
    # The current analytics runtime stores substantial state by cam_id.  Until
    # that state is made tenant-keyed, prevent two tenants with the same cam_id
    # from running concurrently and corrupting one another's runtime state.
    clean = str(cam_id or "").strip()
    if not clean:
        raise ValueError("Camera ID is required")
    return f"camera:legacy_id_lease:{clean}"


def camera_runtime_key(camera_pk: int) -> str:
    return f"camera:runtime:{int(camera_pk)}"


def worker_heartbeat_key(worker_id: str) -> str:
    clean = str(worker_id or "").strip()
    if not clean:
        raise ValueError("Worker ID is required")
    return f"worker:heartbeat:{clean}"


def _require_redis():
    if redis_client is None:
        raise RuntimeError("Redis is required for distributed camera workers")
    return redis_client


# Acquire both the canonical primary-key lease and a compatibility cam_id lease
# atomically.  The second lease protects legacy in-process state that is still
# keyed by cam_id while INTEL-I transitions to fully tenant-keyed runtime state.
_ACQUIRE_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 or redis.call('EXISTS', KEYS[2]) == 1 then
    return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2])
return 1
"""

_RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
    return 0
end
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[2])
return 1
"""

_RELEASE_SCRIPT = """
local removed = 0
if redis.call('GET', KEYS[1]) == ARGV[1] then
    removed = removed + redis.call('DEL', KEYS[1])
end
if redis.call('GET', KEYS[2]) == ARGV[1] then
    removed = removed + redis.call('DEL', KEYS[2])
end
return removed
"""


def acquire_camera_lease(
    camera_pk: int,
    cam_id: str,
    worker_id: str,
) -> CameraLease | None:
    client = _require_redis()
    token = uuid.uuid4().hex
    value = f"{str(worker_id).strip()}|{token}"
    keys = (
        make_key(camera_lease_key(camera_pk)),
        make_key(legacy_camera_id_lease_key(cam_id)),
    )
    try:
        acquired = client.eval(
            _ACQUIRE_SCRIPT,
            2,
            *keys,
            value,
            int(CAMERA_LEASE_TTL_SECONDS),
        )
    except RedisError as exc:
        raise RuntimeError("Unable to acquire camera lease") from exc
    if int(acquired or 0) != 1:
        return None
    return CameraLease(
        camera_pk=int(camera_pk),
        cam_id=str(cam_id),
        worker_id=str(worker_id),
        value=value,
    )


def renew_camera_lease(lease: CameraLease) -> bool:
    client = _require_redis()
    try:
        result = client.eval(
            _RENEW_SCRIPT,
            2,
            make_key(camera_lease_key(lease.camera_pk)),
            make_key(legacy_camera_id_lease_key(lease.cam_id)),
            lease.value,
            int(CAMERA_LEASE_TTL_SECONDS),
        )
        return int(result or 0) == 1
    except RedisError as exc:
        raise RuntimeError("Unable to renew camera lease") from exc


def release_camera_lease(lease: CameraLease) -> bool:
    client = _require_redis()
    try:
        result = client.eval(
            _RELEASE_SCRIPT,
            2,
            make_key(camera_lease_key(lease.camera_pk)),
            make_key(legacy_camera_id_lease_key(lease.cam_id)),
            lease.value,
        )
        return int(result or 0) > 0
    except RedisError as exc:
        raise RuntimeError("Unable to release camera lease") from exc


def get_camera_lease_owner(camera_pk: int) -> str | None:
    client = _require_redis()
    try:
        value = client.get(make_key(camera_lease_key(camera_pk)))
    except RedisError as exc:
        raise RuntimeError("Unable to read camera lease") from exc
    if not value:
        return None
    return str(value).split("|", 1)[0] or None


def set_worker_heartbeat(
    worker_id: str,
    *,
    capacity: int,
    owned_cameras: int,
    status: str,
    preview_base_url: str | None,
    started_at: float,
    cpu_percent: float | None = None,
    memory_percent: float | None = None,
    queue_depth: int | None = None,
    queue_capacity: int | None = None,
    load_score: float | None = None,
) -> bool:
    payload={
        "worker_id": str(worker_id), "status": str(status),
        "capacity": max(0,int(capacity)), "owned_cameras": max(0,int(owned_cameras)),
        "preview_base_url": str(preview_base_url or "").strip() or None,
        "started_at": float(started_at), "updated_at": time.time(),
    }
    if cpu_percent is not None: payload["cpu_percent"]=max(0.0,min(100.0,float(cpu_percent)))
    if memory_percent is not None: payload["memory_percent"]=max(0.0,min(100.0,float(memory_percent)))
    if queue_depth is not None: payload["queue_depth"]=max(0,int(queue_depth))
    if queue_capacity is not None: payload["queue_capacity"]=max(1,int(queue_capacity))
    if load_score is not None: payload["load_score"]=max(0.0,float(load_score))
    return set_json(worker_heartbeat_key(worker_id),payload,ttl=WORKER_HEARTBEAT_TTL_SECONDS)


def get_worker_heartbeat(worker_id: str) -> dict[str, Any] | None:
    return get_json(worker_heartbeat_key(worker_id))


def delete_worker_heartbeat(worker_id: str) -> bool:
    return delete_key(worker_heartbeat_key(worker_id))


def set_camera_runtime(
    camera_pk: int,
    *,
    cam_id: str,
    user_id: int,
    worker_id: str,
    state: str,
    source_type: str | None = None,
    preview_base_url: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    frame_count: int | None = None,
    processing_fps: float | None = None,
    connection_state: str | None = None,
    last_frame_at: float | None = None,
) -> bool:
    payload = {
        "camera_pk": int(camera_pk),
        "cam_id": str(cam_id),
        "user_id": int(user_id),
        "worker_id": str(worker_id),
        "state": str(state or "UNKNOWN").upper(),
        "source_type": str(source_type or "").lower() or None,
        "preview_base_url": str(preview_base_url or "").strip() or None,
        "error_code": str(error_code or "").strip()[:100] or None,
        "error_message": str(error_message or "").strip()[:500] or None,
        "frame_count": max(0, int(frame_count)) if frame_count is not None else None,
        "processing_fps": max(0.0, float(processing_fps)) if processing_fps is not None else None,
        "connection_state": str(connection_state or "").strip().upper() or None,
        "last_frame_at": float(last_frame_at) if last_frame_at is not None else None,
        "updated_at": time.time(),
    }
    return set_json(
        camera_runtime_key(camera_pk),
        payload,
        ttl=CAMERA_RUNTIME_TTL_SECONDS,
    )


def get_camera_runtime(camera_pk: int) -> dict[str, Any] | None:
    return get_json(camera_runtime_key(camera_pk))


def delete_camera_runtime(camera_pk: int) -> bool:
    return delete_key(camera_runtime_key(camera_pk))


def list_worker_heartbeats() -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []
    for key in scan_keys("worker:heartbeat:*"):
        data = get_json(key)
        if isinstance(data, dict):
            workers.append(data)
    workers.sort(key=lambda item: str(item.get("worker_id") or ""))
    return workers


def list_camera_runtimes() -> list[dict[str, Any]]:
    runtimes: list[dict[str, Any]] = []
    for key in scan_keys("camera:runtime:*"):
        data = get_json(key)
        if isinstance(data, dict):
            runtimes.append(data)
    runtimes.sort(key=lambda item: int(item.get("camera_pk") or 0))
    return runtimes


def distributed_capacity_summary() -> dict[str, Any]:
    workers = list_worker_heartbeats()
    total_capacity = sum(max(0, int(item.get("capacity") or 0)) for item in workers)
    owned = sum(max(0, int(item.get("owned_cameras") or 0)) for item in workers)
    return {
        "worker_count": len(workers),
        "total_capacity": total_capacity,
        "owned_cameras": owned,
        "available_capacity": max(0, total_capacity - owned),
        "workers": workers,
    }
