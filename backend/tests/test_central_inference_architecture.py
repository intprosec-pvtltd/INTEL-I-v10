from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace

import numpy as np

from inference.batcher import DynamicBatcher, StaleFrameDropped
from inference.frame_scheduler import FrameScheduler
from inference.policy import CAMERA_POLICIES
from services.cameraPipeline import CameraPipeline


class DynamicBatcherTests(unittest.TestCase):
    def test_latest_frame_replaces_pending_for_same_camera(self):
        batcher = DynamicBatcher(lambda items: items, maxsize=4)
        first = batcher.submit({"camera_id": "CAM01", "value": 1})
        second = batcher.submit({"camera_id": "CAM01", "value": 2})
        self.assertEqual(batcher.depth, 1)
        self.assertTrue(first.done())
        self.assertIsInstance(first.exception(), StaleFrameDropped)
        self.assertFalse(second.done())
        batcher.stop()

    def test_cross_camera_batch_handler_receives_multiple_frames(self):
        seen = []

        def handler(items):
            seen.append([item["camera_id"] for item in items])
            return [{"ok": True, "camera_id": item["camera_id"]} for item in items]

        batcher = DynamicBatcher(handler, maxsize=8)
        # Queue before starting to make the batch deterministic.
        futures = [
            batcher.submit({"camera_id": "CAM01"}, priority=3),
            batcher.submit({"camera_id": "CAM02"}, priority=3),
            batcher.submit({"camera_id": "CAM03"}, priority=3),
        ]
        batcher.start()
        results = [future.result(timeout=2) for future in futures]
        batcher.stop()
        self.assertEqual(len(results), 3)
        self.assertTrue(any(len(batch) >= 3 for batch in seen))


class SchedulerTests(unittest.TestCase):
    def tearDown(self):
        CAMERA_POLICIES.clear("CAM-SCHED")

    def test_adaptive_fps_and_backpressure(self):
        CAMERA_POLICIES.configure(
            "CAM-SCHED",
            {"fps": {"idle_fps": 2, "active_fps": 6, "incident_fps": 8}},
        )
        scheduler = FrameScheduler()
        self.assertAlmostEqual(scheduler.target_fps("CAM-SCHED", 5), 2.0)
        scheduler.mark_active("CAM-SCHED", seconds=2)
        self.assertAlmostEqual(scheduler.target_fps("CAM-SCHED", 4), 6.0)
        scheduler.set_load(100, 100)
        self.assertLessEqual(scheduler.target_fps("CAM-SCHED", 4), 1.0)
        scheduler.mark_critical("CAM-SCHED", seconds=2)
        self.assertGreaterEqual(scheduler.target_fps("CAM-SCHED", 0), 8.0)


class TrueBatchRuntimeTests(unittest.TestCase):
    def test_primary_detector_receives_one_multi_frame_call(self):
        # AnalyticsRuntime imports supervision lazily. Supply the smallest
        # compatible stub so this test does not require the heavy CV wheel.
        original_sv = sys.modules.get("supervision")
        fake_sv = types.ModuleType("supervision")

        class FakeDetections:
            def __init__(self):
                self.class_id = np.array([], dtype=int)

            def __len__(self):
                return 0

            def __getitem__(self, key):
                return self

            @classmethod
            def from_ultralytics(cls, result):
                return cls()

            @classmethod
            def empty(cls):
                return cls()

        fake_sv.Detections = FakeDetections
        sys.modules["supervision"] = fake_sv
        try:
            from runtime.analytics_runtime import AnalyticsRuntime

            calls = []

            class FakePrimary:
                def __call__(self, frames, **kwargs):
                    calls.append((frames, kwargs))
                    return [object() for _ in frames]

            class FakeVehicleEngine:
                def __init__(self):
                    self.calls = 0

                def detect_batch(self, frames):
                    self.calls += 1
                    return [FakeDetections() for _ in frames]

            vehicle_engine = FakeVehicleEngine()
            legacy = SimpleNamespace(
                FRAME_WIDTH=64, FRAME_HEIGHT=32, model_lock=threading.RLock(),
                PERSON_CONF=0.25, VEHICLE_CONF=0.25, VEHICLE_IOU=0.45,
                DEVICE="cpu", VEHICLE_DETECTOR_MODE="dedicated",
                vehicle_analytics_engine=vehicle_engine,
                person_model=FakePrimary(),
            )
            runtime = AnalyticsRuntime()
            runtime._legacy = legacy
            runtime._loaded = True
            runtime._primary_model = legacy.person_model
            frames = [np.zeros((32, 64, 3), dtype=np.uint8) for _ in range(4)]
            results = runtime.detect_batch(frames)
            self.assertEqual(len(results), 4)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0][0]), 4)
            self.assertEqual(vehicle_engine.calls, 1)
        finally:
            if original_sv is None:
                sys.modules.pop("supervision", None)
            else:
                sys.modules["supervision"] = original_sv


class _FakeCapture:
    def __init__(self):
        self.index = 0
        self.closed = False

    def read(self):
        if self.closed:
            return False, None
        self.index += 1
        # Emulate a ~25 FPS decoder without depending on a real camera.
        time.sleep(0.04)
        return True, np.full((32, 32, 3), self.index % 255, dtype=np.uint8)

    def release(self):
        self.closed = True


class _SlowRuntime:
    loaded = True

    def should_submit(self, camera):
        return True

    def priority_for(self, camera):
        return 5

    def target_fps(self, camera):
        return 8.0

    def process_frame(self, camera, frame, timestamp, **kwargs):
        time.sleep(0.20)
        return frame

    def reset_camera(self, camera_id):
        return None


class CameraPipelineTests(unittest.TestCase):
    def test_preview_progresses_while_ai_is_slow(self):
        preview_count = 0
        processed_count = 0

        def on_preview(frame, item):
            nonlocal preview_count
            preview_count += 1

        def on_processed(frame, item):
            nonlocal processed_count
            processed_count += 1

        pipeline = CameraPipeline(
            camera=SimpleNamespace(cam_id="CAM-PREVIEW"),
            runtime=_SlowRuntime(),
            capture_factory=lambda camera: _FakeCapture(),
            analytics_fps=8,
            preview_fps=20,
            on_preview=on_preview,
            on_processed=on_processed,
            reconnect_initial_seconds=0.05,
            reconnect_max_seconds=0.1,
        )
        pipeline.start()
        time.sleep(0.8)
        pipeline.stop(timeout=2)
        self.assertGreater(preview_count, processed_count)
        self.assertGreaterEqual(preview_count, 5)
        self.assertGreaterEqual(pipeline.status()["dropped_frames"], 1)


if __name__ == "__main__":
    unittest.main()
