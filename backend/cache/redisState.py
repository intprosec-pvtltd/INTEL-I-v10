import os
import time

from security.redisClient import (
    set_json,
    get_json,
    delete_key,
    incr_key,
    scan_keys,
)

from core.camera_state import CameraState


CAMERA_STATE_TTL = int(
    os.getenv("CAMERA_STATE_TTL", "86400")
)

UPLOAD_STATE_TTL = int(
    os.getenv("UPLOAD_STATE_TTL", "86400")
)

STREAM_SESSION_TTL = int(
    os.getenv("STREAM_SESSION_TTL", "60")
)


def camera_state_key(cam_id: str) -> str:
    return f"camera:state:{cam_id}"


def camera_frame_key(cam_id: str) -> str:
    return f"camera:frame_count:{cam_id}"


def upload_stream_key(stream_id: str) -> str:
    return f"upload:stream:{stream_id}"


def upload_cam_key(cam_id: str) -> str:
    return f"upload:cam:{cam_id}"


def stream_session_key(sid: str) -> str:
    return f"stream:session:{sid}"


# ============================================================
# CAMERA STATE
# ============================================================

def set_camera_state(
    cam_id: str,
    source: str,
    source_type: str,
    status: bool | None = None,
    frame_count: int = 0,
    state: str | CameraState | None = None,
    worker_running: bool | None = None,
):
    cam_id = str(cam_id or "").strip()

    if not cam_id:
        raise ValueError("Camera ID is required")

    if state is None:
        state_value = (
            CameraState.ONLINE
            if bool(status)
            else CameraState.OFFLINE
        )

    elif isinstance(state, CameraState):
        state_value = state

    else:
        state_value = CameraState(
            str(state).strip().upper()
        )

    existing = get_camera_state(cam_id) or {}

    if worker_running is None:
        # Preserve existing worker lifecycle state when the caller
        # is only changing connection health.
        worker_running_value = bool(
            existing.get("worker_running", False)
        )
    else:
        worker_running_value = bool(worker_running)

    data = {
        "cam_id": cam_id,
        "source": str(source or cam_id),
        "source_type": str(source_type or "unknown").strip().lower(),

        # Connection/source health.
        "status": (
            state_value == CameraState.ONLINE
        ),

        "connection_state": state_value.value,

        # Backend worker lifecycle.
        "worker_running": worker_running_value,

        "frame_count": max(
            0,
            int(frame_count or 0),
        ),

        "updated_at": time.time(),
    }

    return set_json(
        camera_state_key(cam_id),
        data,
        ttl=CAMERA_STATE_TTL,
    )


def get_camera_state(cam_id: str):
    return get_json(
        camera_state_key(str(cam_id))
    )


def update_camera_status(
    cam_id: str,
    status: bool,
    state: str | CameraState | None = None,
    worker_running: bool | None = None,
):

    cam_id = str(cam_id or "").strip()

    if not cam_id:
        raise ValueError("Camera ID is required")

    data = get_camera_state(cam_id) or {
        "cam_id": cam_id,
        "source": cam_id,
        "source_type": "unknown",
        "frame_count": 0,
    }

    if state is None:
        state_value = (
            CameraState.ONLINE
            if bool(status)
            else CameraState.OFFLINE
        )

    elif isinstance(state, CameraState):
        state_value = state

    else:
        state_value = CameraState(
            str(state).strip().upper()
        )

    data["cam_id"] = cam_id
    data["status"] = (
        state_value == CameraState.ONLINE
    )
    data["connection_state"] = state_value.value

    if worker_running is not None:
        data["worker_running"] = bool(
            worker_running
        )
    else:
        data["worker_running"] = bool(
            data.get("worker_running", False)
        )

    data["updated_at"] = time.time()

    return set_json(
        camera_state_key(cam_id),
        data,
        ttl=CAMERA_STATE_TTL,
    )


def increment_camera_frame(cam_id: str) -> int:
    """
    Increment the Redis frame counter and synchronize the
    canonical camera state.
    """

    cam_id = str(cam_id or "").strip()

    if not cam_id:
        return 0

    count = incr_key(
        camera_frame_key(cam_id),
        ttl=CAMERA_STATE_TTL,
    )

    data = get_camera_state(cam_id)

    if data:
        data["frame_count"] = int(count)
        data["updated_at"] = time.time()

        # Never accidentally remove worker_running.
        data["worker_running"] = bool(
            data.get("worker_running", False)
        )

        set_json(
            camera_state_key(cam_id),
            data,
            ttl=CAMERA_STATE_TTL,
        )

    return int(count)


def delete_camera_state(cam_id: str):
    cam_id = str(cam_id or "").strip()

    if not cam_id:
        return

    delete_key(
        camera_state_key(cam_id)
    )

    delete_key(
        camera_frame_key(cam_id)
    )


def is_camera_running(cam_id: str) -> bool:
    """
    Return whether the backend worker is supposed to be running.

    This deliberately uses worker_running rather than connection
    status so temporary RTSP degradation/reconnection does not
    terminate the worker.
    """

    data = get_camera_state(cam_id)

    if not data:
        return False

    return bool(
        data.get("worker_running", False)
    )


# ============================================================
# UPLOAD STATE
# ============================================================

def set_upload_state(
    stream_id: str,
    cam_id: str,
    storage_key: str,
):
    """
    Store secure upload metadata.

    Never store raw filesystem paths here.
    """

    data = {
        "stream_id": str(stream_id),
        "cam_id": str(cam_id),
        "storage_key": str(storage_key),
        "updated_at": time.time(),
    }

    set_json(
        upload_stream_key(stream_id),
        data,
        ttl=UPLOAD_STATE_TTL,
    )

    set_json(
        upload_cam_key(cam_id),
        data,
        ttl=UPLOAD_STATE_TTL,
    )

    return True


def get_upload_by_stream(stream_id: str):
    return get_json(
        upload_stream_key(stream_id)
    )


def get_upload_by_cam(cam_id: str):
    return get_json(
        upload_cam_key(cam_id)
    )


def delete_upload_state(
    stream_id: str | None = None,
    cam_id: str | None = None,
):
    if stream_id:
        delete_key(
            upload_stream_key(stream_id)
        )

    if cam_id:
        delete_key(
            upload_cam_key(cam_id)
        )

    return True


# ============================================================
# STREAM SESSION
# ============================================================

def set_stream_session(
    sid: str,
    user_id: int,
    target_id: str,
    stream_type: str,
    ttl: int | None = None,
):
    return set_json(
        stream_session_key(sid),
        {
            "sid": str(sid),
            "user_id": int(user_id),
            "target_id": str(target_id),
            "stream_type": str(stream_type),
            "created_at": time.time(),
        },
        ttl=ttl or STREAM_SESSION_TTL,
    )


def get_stream_session(sid: str):
    return get_json(
        stream_session_key(sid)
    )


def delete_stream_session(sid: str):
    return delete_key(
        stream_session_key(sid)
    )


# ============================================================
# ACTIVE LIVE CAMERA COUNT
# ============================================================

def get_active_live_camera_count_from_redis() -> int:
    live_types = {
        "rtsp",
        "http",
        "https",
        "live",
        "hls",
        "upload",
        "webcam",
        "onvif",
        "vendor_api",
        "vendor_sdk",
    }

    count = 0

    keys = scan_keys(
        "camera:state:*"
    )

    for key in keys:
        data = get_json(key)

        if not data:
            continue

        connection_state = data.get(
            "connection_state"
        )

        status = data.get(
            "status"
        )

        if (
            connection_state
            not in {
                CameraState.ONLINE.value,
                CameraState.DEGRADED.value,
            }
            and status is not True
        ):
            continue

        if (
            str(data.get("source_type", ""))
            .strip()
            .lower()
            in live_types
        ):
            count += 1

    return count