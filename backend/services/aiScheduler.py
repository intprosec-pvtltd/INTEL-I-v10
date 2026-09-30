
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import Future
from datetime import datetime, timezone
from enum import IntEnum
import heapq
import os
import subprocess
import threading
import time
from typing import Any, Callable


class AIPriority(IntEnum):
    BACKGROUND = 0
    NORMAL = 10
    IMPORTANT = 50
    WATCHLIST = 100


@dataclass
class AIJob:
    job_id: str
    camera_id: str
    frame: Any
    frame_number: int
    analytics_ts: float
    source_type: str
    frame_timestamp: datetime | None = None
    source_fps: float | None = None
    source_pts_seconds: float | None = None
    timestamp_source: str = "UNKNOWN"
    timestamp_quality: str = "UNKNOWN"
    priority: int = int(AIPriority.NORMAL)
    submitted_monotonic: float = field(default_factory=time.monotonic)
    future: Future = field(default_factory=Future)


class GPUResourceGuard:
    """Cached GPU telemetry with fail-open/closed admission policy.

    If GPU telemetry is unavailable, the scheduler does not pretend to know the
    load. In production with GPU protection enabled, admission becomes
    conservative after repeated telemetry failures.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("AI_GPU_PROTECTION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.require_telemetry = os.getenv("AI_GPU_PROTECTION_REQUIRE_TELEMETRY", "false").lower() in {"1", "true", "yes", "on"}
        self.max_util = self._float("AI_GPU_MAX_UTILIZATION_PERCENT", 95.0, 50.0, 100.0)
        self.max_mem = self._float("AI_GPU_MAX_MEMORY_PERCENT", 92.0, 50.0, 100.0)
        self.telemetry_timeout = self._float("AI_GPU_TELEMETRY_TIMEOUT_SECONDS", 0.75, 0.1, 3.0)
        self.cache_ttl = self._float("AI_GPU_TELEMETRY_CACHE_SECONDS", 1.0, 0.1, 5.0)
        self._lock = threading.RLock()
        self._last_check = 0.0
        self._util: float | None = None
        self._mem: float | None = None
        self._error: str | None = None
        self._failure_count = 0

    @staticmethod
    def _float(name: str, default: float, low: float, high: float) -> float:
        try:
            return max(low, min(high, float(os.getenv(name, str(default)))))
        except (TypeError, ValueError):
            return default

    def _query(self) -> None:
        util = mem = None
        error = None
        try:
            # NVML is preferred because it avoids parsing human-oriented output.
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                if count > 0:
                    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
                    utils = [pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
                    mems = []
                    for h in handles:
                        m = pynvml.nvmlDeviceGetMemoryInfo(h)
                        mems.append((m.used / m.total) * 100.0 if m.total else 0.0)
                    util = max(utils) if utils else 0.0
                    mem = max(mems) if mems else 0.0
                else:
                    error = "no NVIDIA GPU detected"
                pynvml.nvmlShutdown()
            except Exception as nvml_exc:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=self.telemetry_timeout,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or str(nvml_exc))
                rows = []
                for line in completed.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 3:
                        continue
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                if not rows:
                    raise RuntimeError("nvidia-smi returned no GPU rows")
                util = max(r[0] for r in rows)
                mem = max((r[1] / r[2]) * 100.0 for r in rows if r[2] > 0)
        except Exception as exc:
            error = str(exc)[:240]

        with self._lock:
            self._last_check = time.monotonic()
            self._util = util
            self._mem = mem
            self._error = error
            self._failure_count = self._failure_count + 1 if error else 0

    def snapshot(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            stale = (time.monotonic() - self._last_check) >= self.cache_ttl
        if force or stale:
            self._query()
        with self._lock:
            return {
                "enabled": self.enabled,
                "utilization_percent": self._util,
                "memory_percent": self._mem,
                "max_utilization_percent": self.max_util,
                "max_memory_percent": self.max_mem,
                "telemetry_error": self._error,
                "telemetry_failures": self._failure_count,
                "telemetry_known": self._util is not None and self._mem is not None,
                "require_telemetry": self.require_telemetry,
            }

    def allow(self, priority: int) -> bool:
        if not self.enabled:
            return True
        state = self.snapshot()
        if not state["telemetry_known"]:
            # In development/CPU-fallback deployments telemetry may be
            # unavailable. Only a production deployment that explicitly
            # requires telemetry should fail closed here.
            if self.require_telemetry:
                return priority >= int(AIPriority.WATCHLIST)
            return True
        overloaded = (
            state["utilization_percent"] >= self.max_util
            or state["memory_percent"] >= self.max_mem
        )
        if not overloaded:
            return True
        # High-priority jobs are admitted; normal/background jobs are shed.
        return priority >= int(AIPriority.WATCHLIST)


class AIScheduler:
    def __init__(self) -> None:
        self.worker_count = self._int("AI_WORKERS", 2, 1, 32)
        self.max_global_queue = self._int("AI_GLOBAL_QUEUE_LIMIT", 32, 1, 512)
        self.max_camera_queue = self._int("AI_CAMERA_QUEUE_LIMIT", 2, 1, 16)
        self.job_timeout = self._float("AI_JOB_TIMEOUT_SECONDS", 8.0, 0.5, 120.0)
        self.admission_timeout = self._float("AI_ADMISSION_TIMEOUT_SECONDS", 0.02, 0.0, 1.0)
        self.max_age = self._float("AI_MAX_JOB_AGE_SECONDS", 2.0, 0.1, 30.0)
        self._condition = threading.Condition(threading.RLock())
        self._heap: list[tuple[int, int, AIJob]] = []
        self._camera_pending: dict[str, int] = {}
        self._sequence = 0
        self._running = False
        self._workers: list[threading.Thread] = []
        self._handler: Callable[[AIJob], Any] | None = None
        self._gpu = GPUResourceGuard()
        self._active_jobs = 0
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        self._gpu_dropped = 0
        self._stale_dropped = 0
        self._queue_high_water = 0
        self._last_error: str | None = None
        self._started_at: float | None = None

    @staticmethod
    def _int(name: str, default: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(os.getenv(name, str(default)))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float(name: str, default: float, low: float, high: float) -> float:
        try:
            return max(low, min(high, float(os.getenv(name, str(default)))))
        except (TypeError, ValueError):
            return default

    def start(self, handler: Callable[[AIJob], Any]) -> None:
        with self._condition:
            if self._running:
                return
            self._handler = handler
            self._running = True
            self._started_at = time.time()
            self._workers = [
                threading.Thread(target=self._worker_loop, name=f"intel-ai-{i+1}", daemon=True)
                for i in range(self.worker_count)
            ]
            for worker in self._workers:
                worker.start()

    def stop(self, wait: bool = True) -> None:
        with self._condition:
            self._running = False
            while self._heap:
                _, _, job = heapq.heappop(self._heap)
                self._camera_pending[job.camera_id] = max(0, self._camera_pending.get(job.camera_id, 1) - 1)
                self._drop_locked(job, "scheduler_shutdown")
            self._condition.notify_all()
        if wait:
            for worker in self._workers:
                worker.join(timeout=max(1.0, self.job_timeout + 1.0))

    def submit(
        self,
        *,
        camera_id: str,
        frame: Any,
        frame_number: int,
        analytics_ts: float,
        source_type: str,
        frame_timestamp: datetime | None = None,
        source_fps: float | None = None,
        source_pts_seconds: float | None = None,
        timestamp_source: str = "UNKNOWN",
        timestamp_quality: str = "UNKNOWN",
        priority: int = int(AIPriority.NORMAL),
    ) -> Future:
        job = AIJob(
            job_id=f"ai-{time.time_ns():x}",
            camera_id=str(camera_id),
            frame=frame,
            frame_number=int(frame_number),
            analytics_ts=float(analytics_ts),
            source_type=str(source_type),
            frame_timestamp=frame_timestamp,
            source_fps=float(source_fps) if source_fps is not None else None,
            source_pts_seconds=float(source_pts_seconds) if source_pts_seconds is not None else None,
            timestamp_source=str(timestamp_source),
            timestamp_quality=str(timestamp_quality),
            priority=max(int(AIPriority.BACKGROUND), min(int(AIPriority.WATCHLIST), int(priority))),
        )
        with self._condition:
            if not self._running or self._handler is None:
                job.future.set_exception(RuntimeError("AI scheduler is not running"))
                return job.future

            if not self._gpu.allow(job.priority):
                self._gpu_dropped += 1
                self._drop_locked(job, "gpu_protection")
                return job.future

            pending = self._camera_pending.get(job.camera_id, 0)
            if pending >= self.max_camera_queue:
                # Keep high-priority work by evicting the oldest lower-priority
                # job from the same camera. Otherwise drop the new frame.
                candidate_index = None
                candidate_key = None
                for i, (_, _, existing) in enumerate(self._heap):
                    if existing.camera_id != job.camera_id:
                        continue
                    key = (existing.priority, existing.submitted_monotonic)
                    if candidate_key is None or key < candidate_key:
                        candidate_key = key
                        candidate_index = i
                if candidate_index is not None and self._heap[candidate_index][2].priority < job.priority:
                    _, _, evicted = self._heap[candidate_index]
                    self._heap[candidate_index] = self._heap[-1]
                    self._heap.pop()
                    heapq.heapify(self._heap)
                    self._camera_pending[evicted.camera_id] -= 1
                    self._drop_locked(evicted, "camera_queue_replaced")
                else:
                    self._drop_locked(job, "camera_queue_full")
                    return job.future

            if len(self._heap) >= self.max_global_queue:
                # Global pressure: find the lowest-priority queued job.
                idx = min(range(len(self._heap)), key=lambda i: (self._heap[i][0], self._heap[i][2].submitted_monotonic))
                if self._heap[idx][0] < job.priority:
                    _, _, evicted = self._heap[idx]
                    self._heap[idx] = self._heap[-1]
                    self._heap.pop()
                    heapq.heapify(self._heap)
                    self._camera_pending[evicted.camera_id] -= 1
                    self._drop_locked(evicted, "global_queue_replaced")
                else:
                    self._drop_locked(job, "global_queue_full")
                    return job.future

            self._sequence += 1
            # heapq is min-first; negate priority so high priority runs first.
            heapq.heappush(self._heap, (-job.priority, self._sequence, job))
            self._camera_pending[job.camera_id] = self._camera_pending.get(job.camera_id, 0) + 1
            self._submitted += 1
            self._queue_high_water = max(self._queue_high_water, len(self._heap))
            self._condition.notify()
            return job.future

    def _drop_locked(self, job: AIJob, reason: str) -> None:
        self._dropped += 1
        if reason == "stale":
            self._stale_dropped += 1
        if not job.future.done():
            job.future.set_exception(RuntimeError(f"AI job dropped: {reason}"))

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and not self._heap:
                    self._condition.wait(timeout=0.5)
                if not self._running and not self._heap:
                    return
                _, _, job = heapq.heappop(self._heap)
                self._camera_pending[job.camera_id] = max(0, self._camera_pending.get(job.camera_id, 1) - 1)
                self._active_jobs += 1

            try:
                age = time.monotonic() - job.submitted_monotonic
                if age > self.max_age and job.priority < int(AIPriority.WATCHLIST):
                    with self._condition:
                        self._drop_locked(job, "stale")
                    continue

                if not self._gpu.allow(job.priority):
                    with self._condition:
                        self._gpu_dropped += 1
                        self._drop_locked(job, "gpu_protection")
                    continue

                handler = self._handler
                if handler is None:
                    raise RuntimeError("AI scheduler handler unavailable")
                result = handler(job)
                if not job.future.done():
                    job.future.set_result(result)
                with self._condition:
                    self._completed += 1
            except Exception as exc:
                with self._condition:
                    self._failed += 1
                    self._last_error = str(exc)[:500]
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                with self._condition:
                    self._active_jobs = max(0, self._active_jobs - 1)
                    self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            gpu = self._gpu.snapshot()
            return {
                "running": self._running,
                "worker_count": self.worker_count,
                "active_jobs": self._active_jobs,
                "queue_depth": len(self._heap),
                "global_queue_limit": self.max_global_queue,
                "camera_queue_limit": self.max_camera_queue,
                "camera_pending": dict(self._camera_pending),
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "dropped": self._dropped,
                "gpu_protection_dropped": self._gpu_dropped,
                "stale_dropped": self._stale_dropped,
                "queue_high_water": self._queue_high_water,
                "last_error": self._last_error,
                "started_at": self._started_at,
                "gpu": gpu,
            }


scheduler = AIScheduler()
