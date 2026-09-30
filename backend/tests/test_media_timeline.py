from datetime import datetime, timezone

import numpy as np
import pytest

from services.mediaTimeline import MediaTimeline, SourcePTSRequiredError


def frame(level: int):
    return np.full((40, 60, 3), level, dtype=np.uint8)


def test_pts_drives_analytics_time_instead_of_frame_count():
    timeline = MediaTimeline("CAM-1", strict_pts=True, scene_cut_threshold=0.9)
    start = datetime(2026, 9, 4, tzinfo=timezone.utc)
    first = timeline.observe(pts_seconds=100.0, received_at=start, received_monotonic=10.0, frame=frame(20))
    second = timeline.observe(pts_seconds=100.08, received_at=start, received_monotonic=12.0, frame=frame(20))
    assert first.analytics_seconds == 0
    assert second.analytics_seconds == pytest.approx(0.08)
    assert second.timestamp_source == "PTS_RELATIVE"
    assert second.timestamp_quality == "SOURCE_PTS"


def test_backward_pts_loop_resets_segment():
    timeline = MediaTimeline("CAM-1", strict_pts=True, scene_cut_threshold=0.9)
    start = datetime(2026, 9, 4, tzinfo=timezone.utc)
    timeline.observe(pts_seconds=10.0, received_at=start, received_monotonic=1.0, frame=frame(20))
    timeline.observe(pts_seconds=11.0, received_at=start, received_monotonic=2.0, frame=frame(20))
    looped = timeline.observe(pts_seconds=0.0, received_at=start, received_monotonic=3.0, frame=frame(20))
    assert looped.discontinuity is True
    assert looped.discontinuity_reason == "PTS_BACKWARD_LOOP"
    assert looped.analytics_seconds == 0
    assert looped.discontinuity_count == 1


def test_visual_hard_cut_resets_track_segment():
    timeline = MediaTimeline(
        "CAM-1",
        strict_pts=True,
        scene_cut_threshold=0.25,
        scene_cut_cooldown_seconds=0,
    )
    start = datetime(2026, 9, 4, tzinfo=timezone.utc)
    timeline.observe(pts_seconds=0.0, received_at=start, received_monotonic=1.0, frame=frame(0))
    cut = timeline.observe(pts_seconds=0.04, received_at=start, received_monotonic=1.04, frame=frame(255))
    assert cut.discontinuity is True
    assert cut.discontinuity_reason == "SCENE_HARD_CUT"


def test_strict_pts_rejects_arrival_time_fallback():
    timeline = MediaTimeline("CAM-1", strict_pts=True)
    with pytest.raises(SourcePTSRequiredError):
        timeline.observe(
            pts_seconds=None,
            received_at=datetime.now(timezone.utc),
            received_monotonic=1.0,
            frame=frame(20),
        )

