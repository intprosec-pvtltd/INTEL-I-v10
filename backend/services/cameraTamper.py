from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import cv2
import numpy as np


@dataclass
class TamperResult:
    tampered: bool
    score: float
    reason: str
    metrics: Dict[str, float]


class CameraTamperDetector:
    """
    Scene-reference tamper detector.
    Detects:
      - covered/black lens
      - sudden scene change
      - severe defocus/blur
      - strong brightness discontinuity

    It deliberately requires consecutive bad observations before alerting.
    """

    def __init__(
        self,
        warmup_frames: int = 30,
        min_change_score: float = 0.55,
        confirm_frames: int = 5,
        black_mean: float = 15.0,
        blur_threshold: float = 18.0,
    ):
        self.warmup_frames = max(1, warmup_frames)
        self.min_change_score = float(min_change_score)
        self.confirm_frames = max(1, confirm_frames)
        self.black_mean = float(black_mean)
        self.blur_threshold = float(blur_threshold)

        self.reference: Optional[np.ndarray] = None
        self.count = 0
        self.bad_count = 0

    @staticmethod
    def _signature_from_gray(gray: np.ndarray) -> np.ndarray:
        small = cv2.resize(
            gray,
            (64, 36),
            interpolation=cv2.INTER_AREA,
        )
        small = cv2.GaussianBlur(
            small,
            (5, 5),
            0,
        )

        return small.astype(
            np.float32,
            copy=False,
        ) / np.float32(255.0)


    def update(self, frame: np.ndarray) -> TamperResult:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return TamperResult(
                True,
                1.0,
                "invalid_frame",
                {},
            )

        try:
            # Use a bounded analysis resolution. Camera tamper detection
            # does not require the complete CCTV resolution.
            height, width = frame.shape[:2]

            analysis_width = min(width, 320)
            analysis_height = max(
                1,
                int(height * analysis_width / max(width, 1)),
            )

            if width > analysis_width:
                analysis_frame = cv2.resize(
                    frame,
                    (analysis_width, analysis_height),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                analysis_frame = frame

            gray = cv2.cvtColor(
                analysis_frame,
                cv2.COLOR_BGR2GRAY,
            )

            sig = self._signature_from_gray(gray)

            # cv2.mean avoids an additional NumPy reduction allocation.
            mean = float(cv2.mean(gray)[0])

            # CV_32F uses half the memory of CV_64F.
            laplacian = cv2.Laplacian(
                gray,
                cv2.CV_32F,
                ksize=3,
            )

            # meanStdDev avoids NumPy's temporary `(arr - mean)` allocation.
            _, standard_deviation = cv2.meanStdDev(laplacian)
            blur = float(
                standard_deviation[0, 0]
                * standard_deviation[0, 0]
            )

            del laplacian

        except (MemoryError, cv2.error):
            # Do not terminate the video pipeline because optional health
            # telemetry could not obtain memory.
            return TamperResult(
                tampered=False,
                score=0.0,
                reason="resource_limited",
                metrics={
                    "brightness": 0.0,
                    "blur": 0.0,
                    "scene_change": 0.0,
                },
            )

        if self.reference is None:
            self.reference = sig.copy()
            self.count = 1

            return TamperResult(
                False,
                0.0,
                "warming_up",
                {
                    "brightness": mean,
                    "blur": blur,
                    "scene_change": 0.0,
                },
            )

        scene_change = float(
            np.mean(
                np.abs(
                    sig - self.reference,
                )
            )
        )

        reasons = []
        score = 0.0

        if mean <= self.black_mean:
            reasons.append("lens_covered_or_black")
            score = max(score, 0.95)

        if blur <= self.blur_threshold:
            reasons.append("severe_defocus")
            score = max(score, 0.75)

        if scene_change >= self.min_change_score:
            reasons.append("major_scene_change")
            score = max(
                score,
                min(1.0, scene_change),
            )

        self.count += 1

        if (
            scene_change < self.min_change_score * 0.35
            and mean > self.black_mean
        ):
            self.reference *= np.float32(0.98)
            self.reference += sig * np.float32(0.02)

        if reasons:
            self.bad_count += 1
        else:
            self.bad_count = 0

        confirmed = self.bad_count >= self.confirm_frames

        return TamperResult(
            tampered=confirmed,
            score=float(
                score if confirmed else score * 0.5
            ),
            reason=(
                ",".join(reasons)
                if reasons
                else "normal"
            ),
            metrics={
                "brightness": mean,
                "blur": blur,
                "scene_change": scene_change,
                "bad_count": float(self.bad_count),
            },
        )

    def status(self) -> dict:
        return {
            "initialized": self.reference is not None,
            "frames_seen": self.count,
            "bad_count": self.bad_count,
        }
