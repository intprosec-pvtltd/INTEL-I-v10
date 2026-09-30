"""Interruptible 1x file pacing, independent of inference/preview FPS."""
import math
import time


class PlaybackClock:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.deadline = None
        self.last_pts = None

    def wait(self, stop_event, pts_seconds=None, fps=None):
        try:
            fps = float(fps)
            if not math.isfinite(fps) or fps <= 0:
                fps = 25.0
        except (TypeError, ValueError):
            fps = 25.0
        try:
            pts = float(pts_seconds)
            if not math.isfinite(pts):
                pts = None
        except (TypeError, ValueError):
            pts = None
        now = self.clock()
        if self.deadline is None:
            self.deadline = now
        else:
            delta = pts - self.last_pts if pts is not None and self.last_pts is not None else 1.0 / fps
            if delta <= 0:
                delta = 1.0 / fps
            # Rebase when processing falls behind; never fast-forward to catch up.
            self.deadline = max(self.deadline + delta, now)
        self.last_pts = pts
        return stop_event.wait(max(0.0, self.deadline - now))
