from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class PipelinePolicy:
    target_fps: float
    process_every_n_frames: int
    jpeg_quality: int
    detect_resolution_scale: float
    reason: str


class AdaptivePipelineController:
    """
    Dynamically adapts analytics workload by scene activity and camera health.
    """

    def __init__(self, min_fps: float = 5.0, max_fps: float = 25.0):
        self.min_fps = float(min_fps)
        self.max_fps = float(max_fps)

    def policy(
        self,
        source_fps: float,
        activity_score: float,
        quality_score: float,
        network_limited: bool = False,
    ) -> PipelinePolicy:

        activity = max(0.0, min(1.0, float(activity_score)))
        quality = max(0.0, min(1.0, float(quality_score)))

        if activity >= 0.75:
            fps = min(self.max_fps, max(self.min_fps, source_fps))
            every = 1
            scale = 1.0
        elif activity >= 0.35:
            fps = min(self.max_fps, max(self.min_fps, source_fps * 0.55))
            every = 2
            scale = 0.85
        else:
            fps = min(self.max_fps, max(self.min_fps, source_fps * 0.30))
            every = 3
            scale = 0.70

        if quality < 0.50:
            # Spend more compute on degraded imagery but reduce output size
            # to keep the pipeline responsive.
            every = min(every, 2)
            scale = min(1.0, scale + 0.10)

        jpeg = 85
        if network_limited:
            jpeg = 65
            fps = min(fps, 10.0)

        return PipelinePolicy(
            target_fps=round(fps, 2),
            process_every_n_frames=every,
            jpeg_quality=jpeg,
            detect_resolution_scale=round(scale, 3),
            reason=(
                "high_activity" if activity >= 0.75 else
                "moderate_activity" if activity >= 0.35 else
                "low_activity"
            ),
        )
