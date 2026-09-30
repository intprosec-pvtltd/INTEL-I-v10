"""Bounded transport for the existing, stateful INTEL-I analytics contract.

Only the Triton Python model owns AnalyticsRuntime in triton mode. Retrying a
frame after an ambiguous timeout could duplicate persistence/alerts, so frame
requests are deliberately never replayed. Readiness probes may be retried.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
import uuid
from typing import Any

import numpy as np


def triton_enabled() -> bool:
    mode = os.getenv("INTEL_I_INFERENCE_MODE", "local").strip().lower()
    if mode not in {"local", "triton"}:
        raise ValueError("INTEL_I_INFERENCE_MODE must be local or triton")
    return mode == "triton" and os.getenv("INTEL_I_TRITON_EMBEDDED") != "1"


class InferenceUnavailable(RuntimeError):
    pass


def json_default(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def json_hook(value):
    if "__ndarray__" in value:
        dtype = value.get("dtype", "float32")
        if dtype not in {"float32", "float64", "uint8", "int32", "int64"}:
            raise ValueError("unsupported inference array dtype")
        return np.asarray(value["__ndarray__"], dtype=dtype)
    return value


def encode_payload(value: dict[str, Any]) -> bytes:
    value = dict(value)
    if "jpeg" in value:
        value["jpeg"] = base64.b64encode(value["jpeg"]).decode("ascii")
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def decode_payload(value: bytes) -> dict[str, Any]:
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("inference payload must be an object")
    if "jpeg" in result:
        result["jpeg"] = base64.b64decode(result["jpeg"], validate=True)
    return result


class TritonInferenceClient:
    def __init__(self, transport=None):
        self.url = os.getenv("INTEL_I_TRITON_GRPC_URL", "triton:8001")
        self.model = os.getenv("INTEL_I_TRITON_MODEL", "intel_i_pipeline")
        self.timeout = float(os.getenv("INTEL_I_TRITON_TIMEOUT_MS", "10000")) / 1000
        self.max_retries = int(os.getenv("INTEL_I_TRITON_MAX_RETRIES", "2"))
        if not 0.05 <= self.timeout <= 120 or not 0 <= self.max_retries <= 5:
            raise ValueError("invalid Triton timeout/retry configuration")
        if transport is None:
            import tritonclient.grpc as grpc
            transport = grpc.InferenceServerClient(url=self.url)
        self.transport = transport
        self._admission = threading.BoundedSemaphore(32)
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0
        self.requests = self.errors = 0
        self.last_latency_ms = 0.0

    def health(self) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                ready = self.transport.is_server_ready(client_timeout=self.timeout)
                ready = ready and self.transport.is_model_ready(self.model, client_timeout=self.timeout)
                if not ready:
                    raise InferenceUnavailable("Triton server/model is not ready")
                return {"status": "READY", "backend": "triton", "model": self.model}
            except Exception as exc:
                if attempt == self.max_retries:
                    raise InferenceUnavailable("Triton readiness failed") from exc
                time.sleep(min(0.1 * 2 ** attempt, 0.5))
        raise AssertionError("unreachable")

    def infer_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._admission.acquire(blocking=False):
            raise InferenceUnavailable("Triton client admission limit reached")
        try:
            return self._infer_batch(items)
        finally:
            self._admission.release()

    def _infer_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        import tritonclient.grpc as grpc
        with self._lock:
            if time.monotonic() < self._open_until:
                raise InferenceUnavailable("Triton circuit is open")
            self.requests += len(items)
        request_id = uuid.uuid4().hex
        rows = [dict(item, request_id=f"{request_id}:{i}") for i, item in enumerate(items)]
        data = np.asarray([[encode_payload(row)] for row in rows], dtype=object)
        tensor = grpc.InferInput("REQUEST", data.shape, "BYTES")
        tensor.set_data_from_numpy(data)
        started = time.monotonic()
        try:
            response = self.transport.infer(
                self.model, [tensor], request_id=request_id,
                outputs=[grpc.InferRequestedOutput("RESPONSE")],
                client_timeout=self.timeout,
            )
            output = response.as_numpy("RESPONSE")
            if output is None or output.shape != data.shape:
                raise InferenceUnavailable("Triton response batch shape mismatch")
            results = [json.loads(row[0], object_hook=json_hook) for row in output]
            for original, result in zip(rows, results):
                if result.get("request_id") != original["request_id"] or result.get("camera_id") != original.get("camera_id"):
                    raise InferenceUnavailable("Triton response routing mismatch")
            with self._lock:
                self.errors += sum(not row.get("ok", False) for row in results)
                self._failures = 0
                self._open_until = 0
            return results
        except Exception as exc:
            with self._lock:
                self.errors += len(items)
                self._failures += 1
                if self._failures >= 3:
                    self._open_until = time.monotonic() + 5
            raise InferenceUnavailable("Triton inference failed; frame was not replayed") from exc
        finally:
            self.last_latency_ms = (time.monotonic() - started) * 1000

    async def infer_batch_async(self, items):
        # The gateway bounds admission using DynamicBatcher. Standalone callers
        # must likewise bound their tasks rather than spawning one per frame.
        if not self._admission.acquire(blocking=False):
            raise InferenceUnavailable("Triton client admission limit reached")
        def invoke():
            try:
                return self._infer_batch(items)
            finally:
                self._admission.release()
        return await asyncio.shield(asyncio.to_thread(invoke))

    def status(self):
        return {"backend": "triton", "model": self.model, "requests": self.requests,
                "errors": self.errors, "last_latency_ms": self.last_latency_ms,
                "circuit_open": time.monotonic() < self._open_until}

    def gpu_telemetry(self):
        import httpx
        from prometheus_client.parser import text_string_to_metric_families
        try:
            url = os.getenv("INTEL_I_TRITON_METRICS_URL", "http://triton:8002/metrics")
            response = httpx.get(url, timeout=0.5)
            response.raise_for_status()
            values = {}
            for family in text_string_to_metric_families(response.text):
                for sample in family.samples:
                    if sample.name.startswith("nv_gpu_"):
                        values[sample.name] = values.get(sample.name, 0.0) + sample.value
            total = values.get("nv_gpu_memory_total_bytes", 0)
            return {"owner": "triton", "cuda_available": total > 0,
                    "memory_used_mb": values.get("nv_gpu_memory_used_bytes", 0) / 1048576,
                    "memory_total_mb": total / 1048576,
                    "utilization_percent": values.get("nv_gpu_utilization", 0) * 100 if total else None}
        except Exception as exc:
            return {"owner": "triton", "telemetry_error": type(exc).__name__}

    def close(self):
        self.transport.close()


_shared_client = None
_shared_lock = threading.Lock()


def shared_client():
    global _shared_client
    with _shared_lock:
        if _shared_client is None:
            _shared_client = TritonInferenceClient()
        return _shared_client


def face_operation(operation, frame, **kwargs):
    import cv2
    ok, image = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("invalid face image")
    result = shared_client().infer_batch([{
        "operation": operation, "camera_id": "face-operation", "jpeg": image.tobytes(), "options": kwargs,
    }])[0]
    if not result.get("ok"):
        raise InferenceUnavailable(result.get("error", "face inference failed"))
    return result["faces"]
