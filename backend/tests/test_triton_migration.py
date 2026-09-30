import json
import os
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from inference.batcher import DynamicBatcher, InferenceQueueFull, StaleFrameDropped
from inference.track_cache import TrackResultCache
from inference.triton_client import TritonInferenceClient, InferenceUnavailable, encode_payload, decode_payload, triton_enabled
from services.latestFrameBuffer import LatestFrameBuffer
from services.workerBalancer import preferred_worker_id


class AdmissionTests(unittest.TestCase):
    def test_queue_capacity_and_latest_frame(self):
        b = DynamicBatcher(lambda items: items, maxsize=1)
        first = b.submit({"camera_id": "a"})
        replacement = b.submit({"camera_id": "a"})
        self.assertIsInstance(first.exception(), StaleFrameDropped)
        rejected = b.submit({"camera_id": "b"})
        self.assertIsInstance(rejected.exception(), InferenceQueueFull)
        self.assertEqual(b.depth, 1)
        b.stop()
        self.assertTrue(replacement.done())

    def test_expiry_never_reaches_handler(self):
        seen = []
        b = DynamicBatcher(lambda rows: seen.extend(rows) or rows)
        future = b.submit({"camera_id": "a"})
        b._pending[0].submitted_at -= 60
        b.start()
        try:
            with self.assertRaises(StaleFrameDropped):
                future.result(timeout=2)
            self.assertEqual(seen, [])
            self.assertEqual(b.expired, 1)
        finally:
            b.stop()

    def test_consumed_frame_is_not_counted_as_dropped(self):
        b = LatestFrameBuffer()
        b.put("a", 1)
        b.get()
        b.put("b", 2)
        self.assertEqual(b.dropped_frames, 0)
        b.put("c", 3)
        self.assertEqual(b.dropped_frames, 1)

    def test_full_and_unready_workers_are_excluded(self):
        workers = [{"worker_id":"a", "status":"READY", "capacity":10,"owned_cameras":10},
                   {"worker_id":"b", "status":"STARTING", "capacity":10,"owned_cameras":0},
                   {"worker_id":"c", "status":"READY", "capacity":10,"owned_cameras":4}]
        self.assertEqual(preferred_worker_id(1, workers), "c")

    def test_reid_cooldown_is_per_track_and_camera(self):
        cache = TrackResultCache()
        cache.mark_attempt("a", 1, "reid", quality=.8, successful=True)
        self.assertFalse(cache.should_run("a",1,"reid", quality=.8,refresh_seconds=10)[0])
        self.assertTrue(cache.should_run("b",1,"reid", quality=.8,refresh_seconds=10)[0])


class Input:
    def __init__(self, name, shape, dtype):
        self.data = None
    def set_data_from_numpy(self, data):
        self.data = data


class Transport:
    def __init__(self, failure=False, mismatch=False):
        self.failure, self.mismatch = failure, mismatch
        self.calls = 0
    def is_server_ready(self, **kwargs):
        return True
    def is_model_ready(self, *args, **kwargs):
        return True
    def infer(self, model, inputs, **kwargs):
        self.calls += 1
        if self.failure:
            raise TimeoutError("deadline")
        rows = [json.loads(row[0]) for row in inputs[0].data]
        output = [[json.dumps({"ok":True, "camera_id":"wrong" if self.mismatch else row["camera_id"], "request_id":row["request_id"]}).encode()] for row in rows]
        return types.SimpleNamespace(as_numpy=lambda name:np.asarray(output,dtype=object))
    def close(self):
        pass


class TransportTests(unittest.TestCase):
    def setUp(self):
        grpc = types.ModuleType("tritonclient.grpc")
        grpc.InferInput = Input
        grpc.InferRequestedOutput = lambda name:name
        package = types.ModuleType("tritonclient")
        package.grpc = grpc
        self.modules = patch.dict(sys.modules, {"tritonclient":package,"tritonclient.grpc":grpc})
        self.modules.start()
    def tearDown(self):
        self.modules.stop()
    def test_routes_batch_to_correct_cameras(self):
        client = TritonInferenceClient(Transport())
        rows = client.infer_batch([{"camera_id":"a","jpeg":b"abc"},{"camera_id":"b","jpeg":b"xyz"}])
        self.assertEqual([r["camera_id"] for r in rows], ["a","b"])
    def test_rejects_wrong_camera(self):
        with self.assertRaises(InferenceUnavailable):
            TritonInferenceClient(Transport(mismatch=True)).infer_batch([{"camera_id":"a"}])
    def test_timeout_is_not_replayed_and_circuit_opens(self):
        transport = Transport(failure=True)
        client = TritonInferenceClient(transport)
        for _ in range(4):
            with self.assertRaises(InferenceUnavailable):
                client.infer_batch([{"camera_id":"a"}])
        self.assertEqual(transport.calls, 3)
    def test_binary_wire_roundtrip(self):
        row={"jpeg":b"\x00\xffabc", "camera_id":"a"}
        self.assertEqual(decode_payload(encode_payload(row)),row)
    def test_unknown_mode_fails_closed(self):
        with patch.dict(os.environ,{"INTEL_I_INFERENCE_MODE":"typo"}):
            with self.assertRaises(ValueError):
                triton_enabled()


if __name__ == "__main__":
    unittest.main()
