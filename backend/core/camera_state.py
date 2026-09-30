"""Canonical camera lifecycle state contract.

Only these values are persisted/exposed by INTEL-I. Legacy CONNECTED and
DISCONNECTED are accepted as input aliases by services/cameraState.py but are
never emitted by the canonical contract.
"""
from enum import Enum


class CameraState(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    TIMEOUT = "TIMEOUT"
    AI_DISABLED = "AI_DISABLED"


VALID_CAMERA_STATE_TRANSITIONS = {
    CameraState.ONLINE: {
        CameraState.ONLINE, CameraState.DEGRADED, CameraState.RECONNECTING,
        CameraState.TIMEOUT, CameraState.AI_DISABLED, CameraState.OFFLINE,
        CameraState.AUTHENTICATION_FAILED,
    },
    CameraState.DEGRADED: {
        CameraState.ONLINE, CameraState.DEGRADED, CameraState.RECONNECTING,
        CameraState.TIMEOUT, CameraState.OFFLINE, CameraState.AI_DISABLED,
    },
    CameraState.RECONNECTING: {
        CameraState.ONLINE, CameraState.DEGRADED, CameraState.RECONNECTING,
        CameraState.OFFLINE, CameraState.AUTHENTICATION_FAILED, CameraState.TIMEOUT,
    },
    CameraState.OFFLINE: {
        CameraState.OFFLINE, CameraState.RECONNECTING, CameraState.AUTHENTICATION_FAILED,
    },
    CameraState.AUTHENTICATION_FAILED: {
        CameraState.AUTHENTICATION_FAILED, CameraState.RECONNECTING, CameraState.OFFLINE,
    },
    CameraState.TIMEOUT: {
        CameraState.TIMEOUT, CameraState.DEGRADED, CameraState.RECONNECTING, CameraState.OFFLINE,
    },
    CameraState.AI_DISABLED: {
        CameraState.AI_DISABLED, CameraState.ONLINE, CameraState.DEGRADED,
        CameraState.RECONNECTING, CameraState.OFFLINE,
    },
}


def is_valid_camera_state_transition(current: CameraState, new: CameraState) -> bool:
    return new in VALID_CAMERA_STATE_TRANSITIONS.get(current, set())
