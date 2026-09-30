from core.camera_contract import normalize_camera_config, normalize_source_type
from core.camera_state import CameraState
from services.cameraState import normalize_camera_state, transition_camera_state


def test_source_aliases_are_normalized():
    assert normalize_source_type("recorded-video") == "recorded_video"
    assert normalize_source_type(" NVR ") == "nvr"


def test_rtsp_transport_is_always_tcp():
    config = normalize_camera_config(
        camera_id="CAM-001",
        name="Main Gate",
        source_type="rtsp",
        direction="north",
        transport="udp",
        stream_fps=25,
        stream_width=1920,
        stream_height=1080,
        latitude=12.9716,
        longitude=77.5946,
    )
    assert config.connection_state == CameraState.OFFLINE
    assert config.transport == "tcp"
    assert config.direction == "NORTH"
    assert config.stream_fps == 25


def test_legacy_state_aliases_are_input_only():
    assert normalize_camera_state("CONNECTED") == CameraState.ONLINE
    assert normalize_camera_state("DISCONNECTED") == CameraState.OFFLINE
    assert transition_camera_state("OFFLINE", "RECONNECTING") == CameraState.RECONNECTING
