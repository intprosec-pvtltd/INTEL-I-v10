from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from services.frs.watchlistSearch import WatchlistIndex


class DistributedFRSTests(unittest.TestCase):
    def test_watchlist_index_is_tenant_scoped(self):
        index = WatchlistIndex()
        index.replace([
            ("11", np.asarray([1.0, 0.0], dtype=np.float32), {"user_id": 1}),
            ("22", np.asarray([0.0, 1.0], dtype=np.float32), {"user_id": 2}),
        ])
        first = index.search(np.asarray([1.0, 0.0], dtype=np.float32), user_id=1)
        second = index.search(np.asarray([1.0, 0.0], dtype=np.float32), user_id=2)
        missing = index.search(np.asarray([1.0, 0.0], dtype=np.float32), user_id=3)
        self.assertIsNotNone(first)
        self.assertEqual(first.identity_id, "11")
        self.assertIsNotNone(second)
        self.assertEqual(second.identity_id, "22")
        self.assertIsNone(missing)

    def test_watchlist_index_rejects_embedding_dimension_mismatch(self):
        index = WatchlistIndex()
        index.replace([("11", np.asarray([1.0, 0.0], dtype=np.float32), {"user_id": 1})])
        self.assertIsNone(index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), user_id=1))

    def test_frs_client_circuit_opens_and_recovers(self):
        try:
            import requests
            from services.frsClient import RemoteFRSClient
        except ImportError:
            self.skipTest("requests is not installed in this validation environment")
        client = RemoteFRSClient(
            base_url="http://frs-worker:9202",
            token="x" * 32,
            failure_threshold=2,
            recovery_seconds=1,
        )
        client._session.post = Mock(side_effect=requests.ConnectionError("down"))
        image = np.full((64, 64, 3), 127, dtype=np.uint8)
        for _ in range(2):
            with self.assertRaises(requests.ConnectionError):
                client.observe(user_id=1, camera_id="CAM01", track_id="1", timestamp=1.0, face_image=image)
        with self.assertRaisesRegex(RuntimeError, "circuit is open"):
            client.observe(user_id=1, camera_id="CAM01", track_id="1", timestamp=1.0, face_image=image)

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "no_match", "confirmed": False}
        client._session.post = Mock(return_value=response)
        client._open_until = 0.0
        result = client.observe(user_id=1, camera_id="CAM01", track_id="1", timestamp=1.0, face_image=image)
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(client._failures, 0)


if __name__ == "__main__":
    unittest.main()
