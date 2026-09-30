"""Thread-safe inference/model execution metrics for the centralized AI plane.

The counters are intentionally process-local.  The inference worker is the GPU
owner in centralized mode, so this gives an authoritative view without leaking
camera URLs, biometric embeddings, plate text, or other sensitive payloads.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
import threading
import time
from typing import Any

MODEL_NAMES = (
    "primary_detector",
    "behaviour",
    "anpr_detector",
    "ocr",
    "frs",
    "mask",
    "person_reid",
    "vehicle_reid",
    "pose",
    "darkir",
    "edsr",
)


@dataclass
class _Metric:
    calls: int = 0
    frames: int = 0
    errors: int = 0
    avoided_by_cache: int = 0
    avoided_by_trigger: int = 0
    latency_ms_sum: float = 0.0
    gpu_ms_sum: float = 0.0
    last_latency_ms: float = 0.0
    last_call_at: float | None = None


class ModelMetrics:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, _Metric] = defaultdict(_Metric)
        for name in MODEL_NAMES:
            self._data[name]

    @staticmethod
    def _name(name: str) -> str:
        value = str(name or "unknown").strip().lower()
        return value[:64] or "unknown"

    def record_call(
        self,
        name: str,
        *,
        frames: int = 1,
        latency_ms: float | None = None,
        gpu_ms: float | None = None,
        error: bool = False,
    ) -> None:
        key = self._name(name)
        with self._lock:
            metric = self._data[key]
            metric.calls += 1
            metric.frames += max(0, int(frames))
            metric.errors += int(bool(error))
            if latency_ms is not None:
                value = max(0.0, float(latency_ms))
                metric.latency_ms_sum += value
                metric.last_latency_ms = value
            if gpu_ms is not None:
                metric.gpu_ms_sum += max(0.0, float(gpu_ms))
            metric.last_call_at = time.time()

    def record_avoided(self, name: str, *, reason: str, count: int = 1) -> None:
        key = self._name(name)
        amount = max(0, int(count))
        with self._lock:
            metric = self._data[key]
            if str(reason).strip().lower() == "cache":
                metric.avoided_by_cache += amount
            else:
                metric.avoided_by_trigger += amount

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {}
            for name, metric in sorted(self._data.items()):
                row = asdict(metric)
                row["average_latency_ms"] = round(
                    metric.latency_ms_sum / max(1, metric.calls), 3
                )
                row["average_gpu_ms"] = round(
                    metric.gpu_ms_sum / max(1, metric.calls), 3
                )
                result[name] = row
            result["totals"] = {
                "calls_avoided_by_cache": sum(
                    item.avoided_by_cache for item in self._data.values()
                ),
                "calls_avoided_by_trigger": sum(
                    item.avoided_by_trigger for item in self._data.values()
                ),
            }
            return result

    def reset(self) -> None:
        with self._lock:
            self._data.clear()
            for name in MODEL_NAMES:
                self._data[name]


MODEL_METRICS = ModelMetrics()
