"""CPU regressions: no model weights, database or GPU needed."""
import ast
from pathlib import Path
import threading
import time
import types
import unittest
import numpy as np
from services.playbackClock import PlaybackClock
from services.cameraPipeline import CameraPipeline
from runtime.analytics_runtime import AnalyticsRuntime

ROOT = Path(__file__).resolve().parents[1]

class FakeClock:
    def __init__(self): self.now = 0.0; self.waits = []
    def wait(self, seconds):
        self.waits.append(seconds); self.now += seconds
        return False

class PlaybackTests(unittest.TestCase):
    def test_source_rates_and_variable_pts(self):
        for fps in (10, 25, 30, 60):
            fake = FakeClock(); clock = PlaybackClock(lambda: fake.now)
            for n in range(fps + 1): clock.wait(fake, n / fps, 12)
            self.assertAlmostEqual(fake.now, 1.0)
        fake = FakeClock(); clock = PlaybackClock(lambda: fake.now)
        for pts in (8, 8.04, 8.12, 8.5): clock.wait(fake, pts, 25)
        self.assertAlmostEqual(fake.now, 0.5)

    def test_missing_duplicate_backward_pts(self):
        fake = FakeClock(); clock = PlaybackClock(lambda: fake.now)
        for pts in (None, None, float('nan'), 3, 3, 2): clock.wait(fake, pts, 20)
        self.assertAlmostEqual(fake.now, 0.25)

    def test_slow_inference_does_not_catch_up_in_burst(self):
        fake = FakeClock(); clock = PlaybackClock(lambda: fake.now)
        clock.wait(fake, 0, 25); fake.now = 5
        clock.wait(fake, .04, 25); clock.wait(fake, .08, 25)
        self.assertAlmostEqual(fake.now, 5.04)

    def test_stop_interrupts_long_wait(self):
        event = threading.Event(); clock = PlaybackClock()
        clock.wait(event, 0, 25); event.set()
        start = time.monotonic()
        self.assertTrue(clock.wait(event, 60, 25))
        self.assertLess(time.monotonic() - start, .1)

class PipelineTests(unittest.TestCase):
    def test_real_runtime_accepts_pipeline_arguments_and_preserves_frame(self):
        received = []
        runtime = AnalyticsRuntime()
        runtime._loaded = True
        runtime._legacy = types.SimpleNamespace(processFrame=lambda **kwargs: received.append(kwargs) or kwargs['frame'])
        camera = types.SimpleNamespace(cam_id='upload_test', source_type='upload')
        frame = np.zeros((20, 40, 3), np.uint8)
        pipeline = CameraPipeline(camera=camera, runtime=runtime, capture_factory=lambda _: None,
                                  on_processed=lambda *_: pipeline.stop_event.set())
        pipeline.buffer.put(frame, time.time(), pts_seconds=1.25)
        thread = threading.Thread(target=pipeline.analytics_loop)
        thread.start(); thread.join(1)
        pipeline.stop_event.set(); thread.join(1)
        self.assertEqual(len(received), 1, pipeline.last_error)
        self.assertIs(received[0]['frame'], frame)
        self.assertEqual(received[0]['analytics_ts'], 1.25)
        self.assertEqual(received[0]['source_type'], 'upload')
        self.assertIsNotNone(received[0]['frame_timestamp'])

    def test_file_capture_is_paced_and_eof_does_not_reopen(self):
        opened = []; frames = []; released = []
        class Capture:
            metadata = {'fps': 20}
            index = 0
            def read_timestamped(self):
                self.index += 1
                return types.SimpleNamespace(ok=self.index <= 4, frame=object() if self.index <= 4 else None,
                                             pts_seconds=(self.index - 1) / 20)
            def release(self): released.append(True)
        def factory(_): opened.append(True); return Capture()
        pipeline = CameraPipeline(camera=types.SimpleNamespace(cam_id='upload_t', source_type='upload'),
                                  runtime=types.SimpleNamespace(loaded=True), capture_factory=factory,
                                  preview_fps=60, on_preview=lambda *_: frames.append(time.monotonic()))
        thread = threading.Thread(target=pipeline.capture_loop); thread.start()
        time.sleep(.3); pipeline.stop_event.set(); thread.join(1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(len(frames), 4)
        self.assertGreaterEqual(frames[-1] - frames[0], .14)
        self.assertEqual(len(released), 1)

class AlertValidationTests(unittest.TestCase):
    def setUp(self):
        # Load the actual validation functions without config's deployment deps.
        tree = ast.parse((ROOT/'alertDecision.py').read_text())
        tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
        self.ns = {'RULE_LEVEL': {'TEST': 'WARNING'}, 'MIN_ALERT_CONFIDENCE': {'TEST': .5}}
        exec(compile(tree, 'alertDecision.py', 'exec'), self.ns)
    def check(self, **kwargs):
        alert = dict(rule='TEST', confidence=.9, track_id=0, box=np.array([2,3,12,23]))
        alert.update(kwargs)
        return self.ns['final_alert_allowed'](alert, {'allowed': True})[0]
    def test_numpy_box_and_zero_track_are_valid(self): self.assertTrue(self.check())
    def test_invalid_inputs_rejected(self):
        for kwargs in ({'confidence': float('nan')}, {'confidence': 'bad'}, {'confidence': .2},
                       {'box': [1,1,0,0]}, {'box': None}, {'rule': 'unknown'}):
            self.assertFalse(self.check(**kwargs))


class RealtimeReconnectTests(unittest.TestCase):
    def test_listener_recovers_after_redis_disconnect(self):
        import logging
        class RedisError(Exception): pass
        stop = threading.Event(); delivered = []; attempts = []
        class Pubsub:
            def subscribe(self, _): pass
            def get_message(self, timeout):
                if len(attempts) == 1: raise RedisError('simulated disconnect')
                return {'data': '{"kind":"alert","payload":{"alert_id":7}}'}
            def close(self): pass
        def pubsub(**_): attempts.append(True); return Pubsub()
        tree = ast.parse((ROOT/'services/realtimeBus.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'start_realtime_subscriber']
        import json
        ns = dict(Callable=__import__('typing').Callable, redis_client=types.SimpleNamespace(pubsub=pubsub),
                  _listener_thread=None, _stop_event=stop, threading=threading, RedisError=RedisError,
                  CHANNEL='test', json=json, logger=logging.getLogger('test.redis'))
        exec(compile(tree, 'realtimeBus.py', 'exec'), ns)
        def callback(payload): delivered.append(payload); stop.set()
        ns['start_realtime_subscriber'](callback)
        ns['_listener_thread'].join(2)
        stop.set()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(delivered[0]['payload']['alert_id'], 7)

class PreviewTransportTests(unittest.TestCase):
    def test_client_returns_exact_annotated_image_not_raw_camera_frame(self):
        import base64
        import cv2
        from inference.client import CentralInferenceClient
        from unittest.mock import patch
        with patch("inference.client.httpx.Client"):
            client = CentralInferenceClient()
        raw = np.zeros((1080, 1920, 3), np.uint8)
        annotated = np.zeros((540, 960, 3), np.uint8)
        cv2.rectangle(annotated, (400,200), (700,450), (0,255,0), 4)
        ok, jpeg = cv2.imencode('.jpg', annotated)
        self.assertTrue(ok)
        response = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'ok':True, 'camera_id':'a', 'annotated_jpeg':base64.b64encode(jpeg).decode('ascii')})
        client.client = types.SimpleNamespace(post=lambda *args, **kwargs: response)
        result = client.process_frame(types.SimpleNamespace(cam_id='a', user_id=1), raw, 1)
        np.testing.assert_array_equal(result, cv2.imdecode(jpeg, cv2.IMREAD_COLOR))
        self.assertEqual(result.shape, annotated.shape)

if __name__ == '__main__': unittest.main()
