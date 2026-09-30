"""Bounded, latest-frame, priority-aware dynamic batching."""
from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

from .settings import SETTINGS


class InferenceQueueFull(RuntimeError):
    pass


class StaleFrameDropped(RuntimeError):
    pass


@dataclass
class InferenceTask:
    priority: int
    sequence: int
    payload: Any
    future: Future
    submitted_at: float = field(default_factory=time.monotonic)

    @property
    def camera_id(self) -> str | None:
        if isinstance(self.payload, dict):
            value = self.payload.get("camera_id")
            return str(value) if value is not None else None
        return None


class DynamicBatcher:
    """A small bounded queue optimized for live video rather than FIFO history.

    Guarantees:
      * at most one pending frame per camera (newest wins),
      * critical/high-priority work can evict low-priority stale work,
      * priority aging prevents permanent starvation,
      * batches are formed across cameras until size or wait deadline.
    """

    def __init__(self, handler: Callable[[list[Any]], list[Any]], maxsize: int | None = None):
        self.handler = handler
        self.maxsize = max(1, int(maxsize or SETTINGS.gpu_queue_max_size))
        self._pending: list[InferenceTask] = []
        self._condition = threading.Condition(threading.RLock())
        self._seq = itertools.count()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_batch_size = 0
        self.last_batch_latency_ms = 0.0
        self.expired = 0
        self.dropped = 0
        self.replaced_stale = 0
        self.evicted_low_priority = 0
        self.completed_batches = 0
        self.completed_frames = 0
        self.batch_size_sum = 0
        self._selections = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="intel-i-dynamic-batcher",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _clamp_priority(priority: int) -> int:
        return max(0, min(5, int(priority)))

    def _effective_priority(self, task: InferenceTask, now: float) -> float:
        waited = max(0.0, now - task.submitted_at)
        age_bonus = waited / max(0.1, SETTINGS.priority_aging_seconds)
        return max(0.0, float(task.priority) - age_bonus)

    def _best_index(self, now: float) -> int:
        return min(
            range(len(self._pending)),
            key=lambda idx: (
                self._effective_priority(self._pending[idx], now),
                self._pending[idx].sequence,
            ),
        )

    def _worst_index(self, now: float) -> int:
        return max(
            range(len(self._pending)),
            key=lambda idx: (
                self._effective_priority(self._pending[idx], now),
                -self._pending[idx].sequence,
            ),
        )

    def submit(self, payload: Any, priority: int = 5) -> Future:
        future: Future = Future()
        task = InferenceTask(
            priority=self._clamp_priority(priority),
            sequence=next(self._seq),
            payload=payload,
            future=future,
        )
        with self._condition:
            if self._stop.is_set():
                future.set_exception(RuntimeError("inference batcher is stopping"))
                return future

            # Live video uses latest-frame semantics.  Replace an older pending
            # request from the same camera instead of growing latency.
            if task.camera_id is not None:
                for idx, previous in enumerate(self._pending):
                    if previous.camera_id == task.camera_id:
                        self._pending[idx] = task
                        self.replaced_stale += 1
                        self.dropped += 1
                        if not previous.future.done():
                            previous.future.set_exception(
                                StaleFrameDropped("stale frame replaced by newer camera frame")
                            )
                        self._condition.notify()
                        return future

            if len(self._pending) >= self.maxsize:
                if not SETTINGS.backpressure_enabled:
                    self.dropped += 1
                    future.set_exception(InferenceQueueFull("inference queue is full"))
                    return future
                now = time.monotonic()
                worst_idx = self._worst_index(now)
                worst = self._pending[worst_idx]
                # Only evict when the arriving work is materially more important.
                if task.priority < worst.priority:
                    self._pending.pop(worst_idx)
                    self.evicted_low_priority += 1
                    self.dropped += 1
                    if not worst.future.done():
                        worst.future.set_exception(
                            StaleFrameDropped("low-priority frame shed under inference pressure")
                        )
                else:
                    self.dropped += 1
                    future.set_exception(
                        InferenceQueueFull("inference queue full; non-critical frame shed")
                    )
                    return future

            self._pending.append(task)
            self._condition.notify()
            return future

    def _pop_best_locked(self) -> InferenceTask:
        self._selections += 1
        # TTL may be shorter than the priority-aging window. Reserve one slot
        # in eight for the oldest waiting routine task so aging is not its only
        # opportunity to run under sustained high-priority traffic.
        routine = [i for i, task in enumerate(self._pending) if task.priority >= 3]
        if self._selections % 8 == 0 and routine:
            idx = min(routine, key=lambda i: self._pending[i].sequence)
        else:
            idx = self._best_index(time.monotonic())
        return self._pending.pop(idx)

    def _take_batch(self) -> list[InferenceTask]:
        max_batch = SETTINGS.batch_size if SETTINGS.batching_enabled else 1
        wait_seconds = SETTINGS.batch_wait_ms / 1000.0
        with self._condition:
            while not self._pending and not self._stop.is_set():
                self._condition.wait(0.2)
            if not self._pending:
                return []
            first = self._pop_best_locked()
            tasks = [first]
            deadline = time.monotonic() + wait_seconds
            while len(tasks) < max_batch and not self._stop.is_set():
                if self._pending:
                    tasks.append(self._pop_best_locked())
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return tasks

    def _run(self) -> None:
        while not self._stop.is_set():
            tasks = self._take_batch()
            if not tasks:
                continue
            fresh = []
            now = time.monotonic()
            for task in tasks:
                if task.future.cancelled():
                    continue
                if (now - task.submitted_at) * 1000 > SETTINGS.frame_ttl_ms:
                    self.expired += 1
                    self.dropped += 1
                    if not task.future.done():
                        task.future.set_exception(StaleFrameDropped("inference frame TTL exceeded"))
                else:
                    fresh.append(task)
            tasks = fresh
            if not tasks:
                continue
            self.last_batch_size = len(tasks)
            started = time.perf_counter()
            try:
                results = self.handler([task.payload for task in tasks])
                if len(results) != len(tasks):
                    raise RuntimeError("batch handler returned wrong result count")
                for task, result in zip(tasks, results):
                    if not task.future.cancelled() and not task.future.done():
                        task.future.set_result(result)
            except Exception as exc:
                for task in tasks:
                    if not task.future.cancelled() and not task.future.done():
                        task.future.set_exception(exc)
            finally:
                self.last_batch_latency_ms = (time.perf_counter() - started) * 1000.0
                self.completed_batches += 1
                self.completed_frames += len(tasks)
                self.batch_size_sum += len(tasks)

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
            pending = list(self._pending)
            self._pending.clear()
        for task in pending:
            if not task.future.done():
                task.future.set_exception(RuntimeError("inference batcher stopped"))
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._pending)

    def priority_depth(self) -> dict[str, int]:
        with self._condition:
            result = {str(priority): 0 for priority in range(6)}
            for task in self._pending:
                result[str(task.priority)] = result.get(str(task.priority), 0) + 1
            return result

    def status(self) -> dict[str, Any]:
        return {
            "queue_depth": self.depth,
            "queue_capacity": self.maxsize,
            "current_batch_size": self.last_batch_size,
            "last_batch_latency_ms": round(self.last_batch_latency_ms, 3),
            "average_batch_size": round(
                self.batch_size_sum / max(1, self.completed_batches), 3
            ),
            "completed_batches": self.completed_batches,
            "completed_frames": self.completed_frames,
            "dropped_frames": self.dropped,
            "expired_frames": self.expired,
            "replaced_stale_frames": self.replaced_stale,
            "evicted_low_priority_frames": self.evicted_low_priority,
            "priority_depth": self.priority_depth(),
        }
