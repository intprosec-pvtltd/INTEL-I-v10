from __future__ import annotations

import threading
import time
import numpy as np

from .schemas import FaceMatch


class WatchlistIndex:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []
        self._metadata: list[dict] = []
        self._lock = threading.RLock()

    def replace(self, entries: list[tuple[str, np.ndarray, dict]]) -> None:
        vectors, ids, metadata = [], [], []
        for identity_id, raw, detail in entries:
            vector = np.asarray(raw, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12 or not np.isfinite(norm):
                continue
            vectors.append(vector / norm); ids.append(str(identity_id)); metadata.append(dict(detail))
        with self._lock:
            self._vectors = np.stack(vectors) if vectors else None
            self._ids, self._metadata = ids, metadata

    def search(self, embedding: np.ndarray, *, user_id: int | None = None) -> FaceMatch | None:
        with self._lock:
            if self._vectors is None:
                return None
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            if vector.shape[0] != self._vectors.shape[1]:
                return None
            allowed = np.asarray([
                user_id is None or int(detail.get("user_id", -1)) == int(user_id)
                for detail in self._metadata
            ], dtype=bool)
            if not bool(np.any(allowed)):
                return None
            similarities = self._vectors @ vector
            similarities = np.where(allowed, similarities, -np.inf)
            index = int(np.argmax(similarities))
            return FaceMatch(self._ids[index], float(similarities[index]), dict(self._metadata[index]))

    def size(self) -> int:
        with self._lock:
            return len(self._ids)


class TemporalConfirmation:
    def __init__(self, min_similarity: float = 0.55, confirmations: int = 3, window_seconds: float = 5.0) -> None:
        self.min_similarity = float(min_similarity)
        self.confirmations = max(1, int(confirmations))
        self.window_seconds = max(0.5, float(window_seconds))
        self._state: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
        self._lock = threading.RLock()

    def observe(self, camera_id: str, track_id: str, match: FaceMatch, timestamp: float | None = None) -> dict:
        now = float(timestamp if timestamp is not None else time.time())
        key = (str(camera_id), str(track_id), match.identity_id)
        with self._lock:
            values = [(ts, score) for ts, score in self._state.get(key, []) if now - ts <= self.window_seconds]
            if match.similarity >= self.min_similarity:
                values.append((now, match.similarity))
            self._state[key] = values
            confirmed = len(values) >= self.confirmations
            return {
                "status": "confirmed" if confirmed else "candidate", "confirmed": confirmed,
                "identity_id": match.identity_id, "similarity": match.similarity,
                "observations": len(values), "required_observations": self.confirmations,
            }
