import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.observationValidation import validate_vehicle_observation


def test_valid_vehicle_observation():
    result = validate_vehicle_observation(
        {
            "camera_id": "CAM-01",
            "track_id": "12",
            "bbox": [100, 100, 300, 300],
            "confidence": 0.9,
            "track_quality": 0.9,
            "frame_quality_score": 0.9,
        },
        frame_width=1920,
        frame_height=1080,
        now_monotonic=10.0,
    )
    assert result.accepted is True
    assert result.confidence > 0.7


def test_low_confidence_is_rejected():
    result = validate_vehicle_observation(
        {
            "camera_id": "CAM-01",
            "track_id": "12",
            "bbox": [100, 100, 300, 300],
            "confidence": 0.1,
            "track_quality": 0.9,
            "frame_quality_score": 0.9,
        },
        frame_width=1920,
        frame_height=1080,
        now_monotonic=10.0,
    )
    assert result.accepted is False
    assert "LOW_DETECTION_CONFIDENCE" in result.reasons


def test_stale_observation_is_rejected():
    result = validate_vehicle_observation(
        {
            "camera_id": "CAM-01",
            "track_id": "12",
            "bbox": [100, 100, 300, 300],
            "confidence": 0.9,
            "track_quality": 0.9,
            "frame_quality_score": 0.9,
            "observation_monotonic": 0.0,
        },
        frame_width=1920,
        frame_height=1080,
        now_monotonic=100.0,
    )
    assert result.accepted is False
    assert "STALE_OBSERVATION" in result.reasons
