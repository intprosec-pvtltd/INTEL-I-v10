"""Production vehicle analytics orchestration for INTEL-I Phase 9.

Separates detector execution, tracking handoff and observation construction
from the FastAPI control plane. It does not own correlation or watchlists.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import supervision as sv

from ai.tensorrt.runtime import runtime as tensorrt_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str
    conf: float
    iou: float
    device: str
    classes: tuple[int, ...]


class VehicleAnalyticsEngine:
    """Owns the vehicle detector instance and its production readiness gate."""

    def __init__(self, fallback_model: Any, class_ids: list[int], device: str) -> None:
        self._fallback_model = fallback_model
        self._class_ids = tuple(int(x) for x in class_ids)
        self._device = device
        self._lock = threading.RLock()
        self._engine_model: Any | None = None
        self._engine_error: str | None = None
        self._attempted_engine = False
        self.enabled = os.getenv("TENSORRT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.required = os.getenv("TENSORRT_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.conf = self._bounded("VEHICLE_CONF", 0.35, 0.05, 0.99)
        self.iou = self._bounded("VEHICLE_IOU", 0.45, 0.05, 0.95)

    @staticmethod
    def _bounded(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except (TypeError, ValueError, OverflowError):
            value = default
        return max(low, min(high, value))

    def _load_engine_model(self) -> Any:
        with self._lock:
            if self._engine_model is not None:
                return self._engine_model
            if self._attempted_engine:
                raise RuntimeError(self._engine_error or "TensorRT detector unavailable")
            self._attempted_engine = True

            state = tensorrt_runtime.status(validate_files=True)
            if not state.get("ready"):
                self._engine_error = "validated TensorRT manifest is not ready"
                raise RuntimeError(self._engine_error)

            try:
                # Validate/deserialise through the Phase 7 security gate first.
                tensorrt_runtime.load_engine("vehicle_detector")
                manifest = state["manifest_validation"]["engines"]
                entry = next(x for x in manifest if x.get("role") == "vehicle_detector")
                engine_path = str(entry["engine_path"])
                if not Path(engine_path).is_file():
                    raise RuntimeError("vehicle detector engine file missing")
                # Ultralytics supports TensorRT .engine inference and performs
                # the exact preprocessing/postprocessing associated with the
                # exported YOLO engine. Do not silently fall back when required.
                from ultralytics import YOLO
                self._engine_model = YOLO(engine_path)
                logger.info("Vehicle TensorRT detector loaded")
                return self._engine_model
            except Exception as exc:
                self._engine_error = str(exc)[:300]
                logger.exception("Vehicle TensorRT detector initialization failed")
                raise

    def model(self) -> Any:
        if not self.enabled:
            if self.required:
                raise RuntimeError("TENSORRT_REQUIRED=true but TENSORRT_ENABLED=false")
            return self._fallback_model
        try:
            return self._load_engine_model()
        except Exception:
            if self.required:
                raise
            logger.warning("TensorRT detector unavailable; using configured fallback detector")
            return self._fallback_model

    def detect(self, frame: Any) -> sv.Detections:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[Any]) -> list[sv.Detections]:
        """Run one real model invocation for a cross-camera frame batch.

        Ultralytics accepts a list of numpy frames and performs batched
        preprocessing/inference while restoring boxes to each original frame.
        TensorRT engines must have been exported with a compatible batch shape;
        when TensorRT is optional, an incompatible engine falls back to the
        configured PyTorch/ONNX model without breaking live cameras.
        """
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            return []
        model = self.model()
        started = time.perf_counter()
        try:
            results = model(
                frames,
                classes=list(self._class_ids),
                conf=self.conf,
                iou=self.iou,
                device=self._device,
                verbose=False,
                stream=False,
            )
        except Exception:
            if self.required or model is self._fallback_model:
                raise
            logger.warning(
                "TensorRT vehicle batch inference failed; retrying configured fallback model",
                exc_info=True,
            )
            results = self._fallback_model(
                frames,
                classes=list(self._class_ids),
                conf=self.conf,
                iou=self.iou,
                device=self._device,
                verbose=False,
                stream=False,
            )
        _ = time.perf_counter() - started
        return [sv.Detections.from_ultralytics(result) for result in list(results)]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "required": self.required,
                "engine_loaded": self._engine_model is not None,
                "engine_attempted": self._attempted_engine,
                "engine_error": self._engine_error,
                "class_ids": list(self._class_ids),
                "device": self._device,
            }
