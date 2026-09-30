"""Regression: restored-only matches must not alert from one source frame."""
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

from services import liveFRS


class Gate:
    def __init__(self):
        self.times = []

    def maybe_enhance(self, *args, **kwargs):
        return np.ones((64, 64, 3), dtype=np.uint8)

    def confirmed_enhanced_only(self, camera, track, identity, timestamp):
        if timestamp not in self.times:
            self.times.append(timestamp)
        return len(self.times) >= 3


class Client:
    def observe(self, **kwargs):
        if str(kwargs["track_id"]).endswith(":darkir"):
            return {"confirmed": True, "identity_id": "7", "similarity": 0.9}
        return {"confirmed": False}


def test_enhanced_only_requires_three_source_frames():
    gate = Gate()
    main = SimpleNamespace(_face_darkir_gate=lambda: gate,
                           DARKIR_ENABLED=True, ADVANCED_INTELLIGENCE_ENABLED=True)
    policies = SimpleNamespace(get=lambda camera: SimpleNamespace(darkir=True))
    candidate = {"box": [0, 0, 64, 64], "detection_score": 0.9}
    frame = np.ones((64, 64, 3), dtype=np.uint8)
    class Tracked:
        xyxy = [[0, 0, 64, 64]]
        tracker_id = ["1"]
        def __len__(self):
            return 1
    with patch.dict(sys.modules, {"main": main}), \
         patch.object(liveFRS, "_get_client", return_value=Client()), \
         patch.object(liveFRS, "detect_face_evidence_candidates", return_value=[candidate]), \
         patch("inference.policy.CAMERA_POLICIES", policies), \
         patch.dict("os.environ", {"FRS_REMOTE_URL": "http://frs-worker:9202", "FRS_DARKIR_ENABLED": "true"}):
        assert liveFRS.match_live_faces(user_id=1, frame=frame, camera_id="cam", timestamp=1, tracked_persons=Tracked()) == []
        assert liveFRS.match_live_faces(user_id=1, frame=frame, camera_id="cam", timestamp=2, tracked_persons=Tracked()) == []
        matches = liveFRS.match_live_faces(user_id=1, frame=frame, camera_id="cam", timestamp=3, tracked_persons=Tracked())
        assert len(matches) == 1 and matches[0]["remote_identity_id"] == "7"
