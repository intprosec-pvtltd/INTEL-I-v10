from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from events.schemas import EventType, IntelIEvent
from services.frs.schemas import FaceEmbedding, FaceMatch
from services.frs.trackFusion import TrackEmbeddingFusion
from services.frs.watchlistSearch import TemporalConfirmation
from services.latestFrameBuffer import LatestFrameBuffer


ROOT = Path(__file__).resolve().parents[1]


class DistributedRuntimeContractTests(unittest.TestCase):
    def test_main_has_no_yolo_constructor(self):
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call.func, ast.Name) and call.func.id == "YOLO" for call in calls))

    def test_latest_buffer_drops_stale_frame(self):
        buffer = LatestFrameBuffer()
        first = buffer.put("first", 1.0)
        buffer.put("latest", 2.0)
        self.assertEqual(buffer.get(after_sequence=first.sequence).frame, "latest")
        self.assertEqual(buffer.dropped_frames, 1)

    def test_fusion_is_l2_normalised(self):
        fusion = TrackEmbeddingFusion()
        vector = fusion.add(FaceEmbedding("cam", "7", 1.0, np.array([3.0, 4.0]), 0.8, {}))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_confirmation_requires_multiple_observations(self):
        gate = TemporalConfirmation(min_similarity=0.5, confirmations=2, window_seconds=5)
        match = FaceMatch("identity", 0.8, {})
        self.assertFalse(gate.observe("cam", "7", match, 1.0)["confirmed"])
        self.assertTrue(gate.observe("cam", "7", match, 2.0)["confirmed"])

    def test_events_reject_inline_media(self):
        with self.assertRaises(ValueError):
            IntelIEvent(
                event_type=EventType.CAMERA_DETECTION,
                camera_id="cam", worker_id="worker", model_version="1",
                payload={"frame": "base64"},
            )

    def test_api_image_excludes_gpu_frameworks(self):
        dockerfile = (ROOT / "docker" / "api.Dockerfile").read_text(encoding="utf-8")
        filter_line = next(line for line in dockerfile.splitlines() if line.startswith("RUN sed"))
        for package in ("torch", "ultralytics", "paddle", "onnxruntime-gpu"):
            self.assertIn(package, filter_line)


if __name__ == "__main__":
    unittest.main()
