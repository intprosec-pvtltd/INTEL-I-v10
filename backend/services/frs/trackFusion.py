from __future__ import annotations

from collections import defaultdict, deque
import threading
import time
import numpy as np

from .schemas import FaceEmbedding


class TrackEmbeddingFusion:
    def __init__(self, max_observations: int = 8, ttl_seconds: float = 15.0) -> None:
        self.max_observations = max(2, min(20, int(max_observations)))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._tracks: dict[tuple[str, str], deque[FaceEmbedding]] = defaultdict(lambda: deque(maxlen=self.max_observations))
        self._lock = threading.RLock()

    def add(self, observation: FaceEmbedding) -> np.ndarray:
        key = (observation.camera_id, observation.track_id)
        with self._lock:
            self.cleanup(observation.timestamp)
            self._tracks[key].append(observation)
            return self.fused(*key)

    def fused(self, camera_id: str, track_id: str) -> np.ndarray:
        values = list(self._tracks.get((str(camera_id), str(track_id)), ()))
        if not values:
            raise KeyError("track has no embeddings")
        weights = np.asarray([max(1e-3, item.quality) for item in values], dtype=np.float32)
        matrix = np.stack([item.vector for item in values]).astype(np.float32)
        vector = (matrix * weights[:, None]).sum(axis=0) / weights.sum()
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise RuntimeError("fused embedding norm is zero")
        return vector / norm

    def count(self, camera_id: str, track_id: str) -> int:
        return len(self._tracks.get((str(camera_id), str(track_id)), ()))

    def cleanup(self, now: float | None = None) -> None:
        cutoff = float(now if now is not None else time.time()) - self.ttl_seconds
        for key, values in list(self._tracks.items()):
            while values and values[0].timestamp < cutoff:
                values.popleft()
            if not values:
                self._tracks.pop(key, None)
