from datetime import datetime, timezone

from core.camera_state import CameraState
from services.cameraHealth import CameraHealthRegistry
from services.streamManager import BoundedFrameBuffer, FramePacket
from services.timeSync import CameraTimeSynchronizer


def test_camera_health_tracks_frames_and_drops():
    registry = CameraHealthRegistry(stale_after_seconds=2, persist_interval_seconds=1)
    registry.ensure("CAM-001", queue_capacity=2)
    registry.set_state("CAM-001", CameraState.ONLINE)
    registry.frame("CAM-001", read_latency_ms=40.0, queue_depth=1)
    registry.drop("CAM-001", 2)
    item = registry.get("CAM-001")
    assert item["state"] == "ONLINE"
    assert item["decoded_frames"] == 1
    assert item["dropped_frames"] == 2


def test_bounded_frame_buffer_drops_oldest():
    buffer = BoundedFrameBuffer(capacity=2)
    now = datetime.now(timezone.utc)
    for i in range(3):
        buffer.put_latest(FramePacket("CAM-001", i, i, now, 0.0, None, 10, 10))
    packet = buffer.get_latest()
    assert packet.frame == 2
    assert buffer.dropped == 1


def test_time_sync_marks_missing_source_timestamp_as_estimated():
    sync = CameraTimeSynchronizer(max_offset_ms=5000)
    result = sync.observe("CAM-001", camera_timestamp=None)
    assert result.sync_status == "ESTIMATED"
    assert result.sync_method == "ARRIVAL_TIME"
    assert result.clock_offset_ms == 0.0
