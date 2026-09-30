from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class FrozenFrameResult:
    frozen: bool
    score: float
    consecutive_identical: int
    reason: str


class FrozenFrameDetector:
    def __init__(self, similarity_threshold: float = 0.999, confirm_frames: int = 12):
        self.similarity_threshold = float(similarity_threshold)
        self.confirm_frames = max(2, int(confirm_frames))
        self.prev: Optional[np.ndarray] = None
        self.identical_count = 0

    @staticmethod
    def _signature(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray) -> FrozenFrameResult:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return FrozenFrameResult(False, 0.0, 0, "invalid_frame")

        sig = self._signature(frame)

        if self.prev is None:
            self.prev = sig
            return FrozenFrameResult(False, 0.0, 0, "warming_up")

        a = sig.astype(np.float32)
        b = self.prev.astype(np.float32)
        mse = float(np.mean((a - b) ** 2))
        similarity = 1.0 - min(1.0, mse / (255.0 ** 2))

        if similarity >= self.similarity_threshold:
            self.identical_count += 1
        else:
            self.identical_count = 0

        self.prev = sig

        frozen = self.identical_count >= self.confirm_frames
        score = min(1.0, self.identical_count / float(self.confirm_frames))

        return FrozenFrameResult(
            frozen=frozen,
            score=score,
            consecutive_identical=self.identical_count,
            reason="stream_frozen" if frozen else "normal",
        )
