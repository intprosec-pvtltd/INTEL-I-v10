from datetime import datetime, timezone
from core.camera_state import CameraState, is_valid_camera_state_transition


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_camera_state(value: str | CameraState) -> CameraState:
    if isinstance(value, CameraState):
        return value
    normalized = str(value or "").strip().upper()
    aliases = {
        "CONNECTED": CameraState.ONLINE,
        "ONLINE": CameraState.ONLINE,
        "DISCONNECTED": CameraState.OFFLINE,
        "OFFLINE": CameraState.OFFLINE,
        "RECONNECTING": CameraState.RECONNECTING,
        "DEGRADED": CameraState.DEGRADED,
        "AUTH_FAILED": CameraState.AUTHENTICATION_FAILED,
        "AUTHENTICATION_FAILED": CameraState.AUTHENTICATION_FAILED,
        "TIMEOUT": CameraState.TIMEOUT,
        "AI_DISABLED": CameraState.AI_DISABLED,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown camera state: {value!r}") from exc


def transition_camera_state(current: str | CameraState, new: str | CameraState) -> CameraState:
    current_state = normalize_camera_state(current)
    new_state = normalize_camera_state(new)
    if current_state == new_state:
        return new_state
    if not is_valid_camera_state_transition(current_state, new_state):
        raise ValueError(f"Invalid camera state transition: {current_state.value} -> {new_state.value}")
    return new_state
