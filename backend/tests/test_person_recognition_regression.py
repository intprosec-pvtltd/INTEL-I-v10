from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

import numpy as np


class _CvError(Exception):
    pass


def _load_module():
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.error = _CvError
    sys.modules.setdefault("cv2", fake_cv2)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "services"
        / "personRecognition.py"
    )
    spec = importlib.util.spec_from_file_location(
        "person_recognition_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _Detector:
    def __init__(self, face):
        self.face = np.asarray([face], dtype=np.float32)
        self.calls = 0

    def setInputSize(self, _size):
        return None

    def setScoreThreshold(self, _threshold):
        return None

    def detect(self, _frame):
        self.calls += 1
        return None, self.face.copy()


class PersonRecognitionRegressionTests(unittest.TestCase):
    def setUp(self):
        os.environ["PERSON_FACE_MIN_DET_SCORE"] = "0.45"
        os.environ["PERSON_MIN_FACE_SIZE"] = "40"
        os.environ["PERSON_FACE_MIN_BLUR_SCORE"] = "12"
        self.module = _load_module()
        self.frame = np.full((240, 320, 3), 127, dtype=np.uint8)
        # x, y, w, h, five landmarks, detection score
        self.face = [
            20, 20, 35, 35,
            30, 30, 44, 30, 37, 38, 31, 47, 44, 47,
            0.95,
        ]
        self.detector = _Detector(self.face)
        self.module._models = lambda: (self.detector, object())
        self.module._reference_variants = lambda frame: [frame]
        self.module._face_blur_score = lambda _roi: 100.0
        self.module._extract_sface_embedding = (
            lambda _recognizer, _frame, _face: np.asarray(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
        )

    def test_live_profile_retries_after_candidate_quality_rejection(self):
        results = self.module.extract_embeddings(
            self.frame,
            profile="live",
            allow_fallback=True,
        )
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(self.detector.calls, 2)
        self.assertEqual(results[0]["box"], [20, 20, 55, 55])

    def test_enrollment_profile_does_not_use_relaxed_live_gate(self):
        results = self.module.extract_embeddings(
            self.frame,
            profile="enrollment",
            allow_fallback=False,
        )
        self.assertEqual(results, [])
        self.assertEqual(self.detector.calls, 1)

    def test_match_threshold_remains_enforced(self):
        entry = object()
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        accepted, score = self.module.match_embeddings(
            query,
            [(entry, np.asarray([0.8, 0.6], dtype=np.float32))],
            threshold=0.75,
        )
        self.assertIs(accepted, entry)
        self.assertAlmostEqual(score, 0.8, places=5)

        rejected, _ = self.module.match_embeddings(
            query,
            [(entry, np.asarray([0.8, 0.6], dtype=np.float32))],
            threshold=0.85,
        )
        self.assertIsNone(rejected)


if __name__ == "__main__":
    unittest.main()
