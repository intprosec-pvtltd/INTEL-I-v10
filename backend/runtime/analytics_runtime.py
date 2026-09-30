"""Worker-owned analytics runtime.

The API/control plane never owns heavy executable models in centralized mode.
The inference worker imports the legacy, validated INTEL-I pipeline once, loads
models once, performs true cross-camera primary-detector batching, then feeds
precomputed detections into the existing ByteTrack/secondary/event pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
import logging
import os
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RuntimeNotLoadedError(RuntimeError):
    """Raised when inference is attempted before ``load`` succeeds."""


@dataclass(frozen=True)
class RuntimeStatus:
    status: str
    loaded: bool
    person_detector: bool
    pose_detector: bool
    crime_detector: bool
    ppe_detector: bool
    vehicle_detector: bool
    error: str | None = None


class AnalyticsRuntime:
    """Own all heavy model state for one analytics/inference worker process."""

    MODEL_BINDINGS = {
        "person_model": "MODEL_PERSON",
        "pose_model": "MODEL_POSE",
        "crime_model": "MODEL_CRIME",
        "ppe_model": "MODEL_PPE",
        "vehicle_model": "VEHICLE_MODEL_PATH",
    }

    def __init__(self, *, module_name: str = "main", yolo_factory: Callable[[str], Any] | None = None):
        self.module_name = module_name
        self._yolo_factory = yolo_factory
        self._legacy: Any = None
        self._loaded = False
        self._lock = threading.RLock()
        self._error: str | None = None
        self._batch_calls = 0
        self._batch_frames = 0
        self._batch_fallbacks = 0
        self._last_batch_ms = 0.0
        self._primary_model: Any = None
        self._primary_backend = "pytorch"

    @property
    def legacy(self) -> Any:
        if self._legacy is None:
            raise RuntimeNotLoadedError("Analytics runtime module is not loaded")
        return self._legacy

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> "AnalyticsRuntime":
        with self._lock:
            if self._loaded:
                return self
            from inference.triton_client import triton_enabled
            if triton_enabled():
                raise RuntimeError("Local model ownership is forbidden in Triton client processes")
            role = os.getenv("INTEL_I_PROCESS_ROLE", "").strip().lower()
            if role not in {"camera-worker", "analytics-worker", "inference-worker", "test"}:
                raise RuntimeError(
                    "AnalyticsRuntime may only load in a camera/analytics/inference worker process"
                )
            try:
                legacy = importlib.import_module(self.module_name)
                factory = self._yolo_factory
                if factory is None:
                    from ultralytics import YOLO
                    factory = YOLO

                shared_models = {}
                for target, path_name in self.MODEL_BINDINGS.items():
                    if getattr(legacy, target, None) is None:
                        from inference.engine_manifest import resolve_engine
                        path = resolve_engine(str(getattr(legacy, path_name)))
                        key = os.path.realpath(path)
                        if key not in shared_models:
                            if os.getenv("ENV", "dev").lower() in {"prod", "production"} and not os.path.isfile(path):
                                raise FileNotFoundError(f"Provision model before startup: {path}")
                            shared_models[key] = factory(path)
                        setattr(legacy, target, shared_models[key])

                resolver = getattr(legacy, "_resolve_configured_vehicle_class_ids")
                class_ids = resolver(legacy.vehicle_model)
                legacy.VEHICLE_CLASS_IDS = class_ids
                engine_type = type(legacy.vehicle_analytics_engine)
                legacy.vehicle_analytics_engine = engine_type(
                    fallback_model=legacy.vehicle_model,
                    class_ids=class_ids,
                    device=legacy.DEVICE,
                )
                # Optional TensorRT FP16 primary detector. Engines are external
                # deployment assets and are never bundled in the source archive.
                from inference.settings import SETTINGS
                self._primary_model = legacy.person_model
                self._primary_backend = "pytorch"
                engine_path = str(SETTINGS.primary_tensorrt_engine or "").strip()
                if SETTINGS.tensorrt_enabled and engine_path:
                    if os.path.isfile(engine_path):
                        try:
                            self._primary_model = factory(engine_path)
                            self._primary_backend = "tensorrt-fp16"
                            logger.info("TensorRT primary detector loaded | engine=%s", os.path.basename(engine_path))
                        except Exception:
                            logger.exception("TensorRT primary detector unavailable; using configured model fallback")
                            self._primary_model = legacy.person_model
                    else:
                        logger.warning("TensorRT primary engine not found; using configured model fallback | path=%s", engine_path)
                self._legacy = legacy
                self._loaded = True
                self._error = None
                logger.info("Analytics runtime loaded in process role=%s", role)
                return self
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {str(exc)[:300]}"
                self._loaded = False
                logger.exception("Analytics runtime load failed")
                raise

    @staticmethod
    def _filter_detections(detections: Any, allowed_ids: set[int]) -> Any:
        import numpy as np
        import supervision as sv
        if detections is None or len(detections) == 0:
            return sv.Detections.empty()
        class_ids = getattr(detections, "class_id", None)
        if class_ids is None:
            return sv.Detections.empty()
        mask = np.array([int(value) in allowed_ids for value in class_ids], dtype=bool)
        return detections[mask]

    def detect_batch(self, frames: list[Any]) -> list[dict[str, Any]]:
        """Perform real batched primary inference across cameras.

        Returns one ``{person, vehicle}`` detection pair per input frame.  The
        downstream legacy processFrame receives these objects and therefore
        does not execute primary YOLO again.
        """
        if not self._loaded:
            raise RuntimeNotLoadedError("Call AnalyticsRuntime.load() before detect_batch()")
        if not frames:
            return []

        import supervision as sv
        from inference.model_metrics import MODEL_METRICS
        from inference.settings import SETTINGS

        legacy = self.legacy
        # Primary detections must be generated in the same coordinate space
        # used by legacy processFrame (FRAME_WIDTH x FRAME_HEIGHT), otherwise
        # ByteTrack boxes would not align after processFrame resizes the frame.
        import cv2
        normalized_frames = [
            cv2.resize(frame, (int(legacy.FRAME_WIDTH), int(legacy.FRAME_HEIGHT)), interpolation=cv2.INTER_LINEAR)
            if frame is not None and tuple(frame.shape[1::-1]) != (int(legacy.FRAME_WIDTH), int(legacy.FRAME_HEIGHT))
            else frame
            for frame in frames
        ]
        started = time.perf_counter()
        gpu_start = gpu_end = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_start=torch.cuda.Event(enable_timing=True); gpu_end=torch.cuda.Event(enable_timing=True); gpu_start.record()
        except Exception:
            gpu_start=gpu_end=None
        output: list[dict[str, Any]] = []
        batch_fallback = False
        try:
            mode = SETTINGS.primary_detector_mode
            vehicle_mode = str(getattr(legacy, "VEHICLE_DETECTOR_MODE", "dedicated")).strip().lower()
            if mode == "unified" or vehicle_mode == "shared":
                class_ids = sorted({0, 2, 3, 5, 7})
                with legacy.model_lock:
                    results = self._primary_model(
                        normalized_frames,
                        classes=class_ids,
                        conf=min(float(legacy.PERSON_CONF), float(legacy.VEHICLE_CONF)),
                        iou=float(legacy.VEHICLE_IOU),
                        device=legacy.DEVICE,
                        verbose=False,
                        stream=False,
                    )
                results = list(results)
                if len(results) != len(frames):
                    raise RuntimeError("unified primary detector returned wrong batch size")
                for result in results:
                    detections = sv.Detections.from_ultralytics(result)
                    output.append({
                        "person": self._filter_detections(detections, {0}),
                        "vehicle": self._filter_detections(detections, {2, 3, 5, 7}),
                    })
            else:
                with legacy.model_lock:
                    person_results = self._primary_model(
                        normalized_frames,
                        classes=[0],
                        conf=float(legacy.PERSON_CONF),
                        device=legacy.DEVICE,
                        verbose=False,
                        stream=False,
                    )
                person_results = list(person_results)
                vehicle_results = legacy.vehicle_analytics_engine.detect_batch(normalized_frames)
                if len(person_results) != len(frames) or len(vehicle_results) != len(frames):
                    raise RuntimeError("split primary detector returned wrong batch size")
                for person_result, vehicle_detection in zip(person_results, vehicle_results):
                    output.append({
                        "person": sv.Detections.from_ultralytics(person_result),
                        "vehicle": vehicle_detection,
                    })
        except Exception:
            # A batch-incompatible engine should not terminate live streams.
            # Fall back per frame while surfacing the fallback in telemetry.
            if os.getenv("INFERENCE_BATCH_FALLBACK_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
                raise
            batch_fallback = True
            logger.exception("Primary batch inference failed; using per-frame compatibility fallback")
            output = []
            for frame in normalized_frames:
                try:
                    with legacy.model_lock:
                        person_result = legacy.person_model(
                            frame,
                            classes=[0],
                            conf=float(legacy.PERSON_CONF),
                            device=legacy.DEVICE,
                            verbose=False,
                        )[0]
                    person_detection = sv.Detections.from_ultralytics(person_result)
                except Exception:
                    logger.exception("Person detector fallback failed")
                    person_detection = sv.Detections.empty()
                try:
                    vehicle_detection = legacy.vehicle_analytics_engine.detect(frame)
                except Exception:
                    logger.exception("Vehicle detector fallback failed")
                    vehicle_detection = sv.Detections.empty()
                output.append({"person": person_detection, "vehicle": vehicle_detection})

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        gpu_ms = None
        if gpu_start is not None and gpu_end is not None:
            try:
                gpu_end.record()
                import torch
                torch.cuda.synchronize()
                gpu_ms=float(gpu_start.elapsed_time(gpu_end))
            except Exception:
                gpu_ms=None
        self._batch_calls += 1
        self._batch_frames += len(frames)
        self._last_batch_ms = elapsed_ms
        if batch_fallback:
            self._batch_fallbacks += 1
        MODEL_METRICS.record_call(
            "primary_detector",
            frames=len(frames),
            latency_ms=elapsed_ms, gpu_ms=gpu_ms,
        )
        return output

    def process_frame(
        self,
        camera: Any,
        frame: Any,
        timestamp: float,
        *,
        frame_count: int = 0,
        priority: int = 5,
        fps: float | None = None,
        source_type: str | None = None,
        frame_timestamp: datetime | None = None,
        source_pts_seconds: float | None = None,
        timestamp_source: str = "UNKNOWN",
        timestamp_quality: str = "UNKNOWN",
        precomputed_person_detections: Any = None,
        precomputed_vehicle_detections: Any = None,
    ) -> Any:
        if not self._loaded:
            raise RuntimeNotLoadedError("Call AnalyticsRuntime.load() before process_frame()")
        camera_id = str(getattr(camera, "cam_id", None) or getattr(camera, "id", None) or camera)
        resolved_source = source_type or str(getattr(camera, "source_type", "rtsp"))
        return self.legacy.processFrame(
            frame=frame,
            camID=camera_id,
            frame_count=frame_count,
            fps=fps,
            source_type=resolved_source,
            analytics_ts=float(timestamp),
            frame_timestamp=frame_timestamp,
            source_pts_seconds=source_pts_seconds,
            timestamp_source=timestamp_source,
            timestamp_quality=timestamp_quality,
            precomputed_person_detections=precomputed_person_detections,
            precomputed_vehicle_detections=precomputed_vehicle_detections,
        )

    def schedule_hint(
        self,
        camera_id: str,
        *,
        person_count: int = 0,
        vehicle_count: int = 0,
    ) -> dict[str, Any]:
        """Summarize existing alert/track state into the next-frame priority."""
        legacy = self.legacy
        camera_id = str(camera_id)
        active = list((getattr(legacy, "activeAlertBoxes", {}).get(camera_id) or {}).values())
        rules = [str(item.get("rule") or "").upper() for item in active if isinstance(item, dict)]
        levels = [str(item.get("level") or "").upper() for item in active if isinstance(item, dict)]
        if any(level == "CRITICAL" for level in levels):
            return {"mode": "critical", "priority": 0, "reason": "critical_incident"}
        if any("WATCHLIST" in rule for rule in rules):
            return {"mode": "incident", "priority": 1, "reason": "watchlist"}
        if active:
            return {"mode": "incident", "priority": 2, "reason": "active_behaviour_alert"}
        if person_count or vehicle_count:
            return {"mode": "active", "priority": 3, "reason": "active_objects"}
        return {"mode": "idle", "priority": 5, "reason": "ordinary_surveillance"}

    def register_models(self, registry: Any) -> None:
        legacy = self.legacy
        device = str(getattr(legacy, "DEVICE", "unknown"))
        registry.register("primary_detector", self._primary_model or legacy.person_model, role="primary_detector", device=device, required=True, version=self._primary_backend)
        registry.register("person_detector", legacy.person_model, role="primary_detector", device=device, required=True)
        registry.register("vehicle_detector", legacy.vehicle_analytics_engine.model(), role="primary_detector", device=device, required=True)
        registry.register("behaviour", legacy.crime_model, role="secondary", device=device)
        registry.register("pose", legacy.pose_model, role="secondary", device=device)
        registry.register("mask", legacy.ppe_model, role="secondary", device=device)
        try:
            import vehicleANPR
            registry.register_provider(
                "anpr_plate_detector",
                vehicleANPR.get_plate_model,
                health_provider=lambda: {"loaded": getattr(vehicleANPR, "_plate_model", None) is not None},
                role="secondary",
                device=device,
                required=True,
            )
            registry.register_provider(
                "ocr",
                vehicleANPR.get_ocr_engine,
                health_provider=vehicleANPR.get_ocr_runtime_status,
                role="recognizer",
                device=str(getattr(vehicleANPR, "OCR_DEVICE", "auto")),
                required=True,
            )
        except Exception:
            logger.exception("Unable to register ANPR/OCR model providers")
        try:
            import vehicleReid
            registry.register_provider(
                "vehicle_reid",
                lambda: getattr(vehicleReid, "_session", None) or getattr(vehicleReid, "_model", None),
                role="reid",
                device=device,
            )
        except Exception:
            logger.debug("Vehicle Re-ID registry provider unavailable", exc_info=True)
        try:
            from services import personRecognition
            registry.register_provider(
                "face_detector",
                lambda: getattr(personRecognition, "_face_detector", None),
                health_provider=personRecognition.model_status,
                role="frs",
                device=device,
            )
            registry.register_provider(
                "face_recognizer",
                lambda: getattr(personRecognition, "_face_recognizer", None),
                health_provider=personRecognition.model_status,
                role="frs",
                device=device,
            )
        except Exception:
            logger.debug("FRS registry provider unavailable", exc_info=True)
        registry.register_provider(
            "person_reid",
            lambda: getattr(legacy, "person_correlation_engine", None),
            role="reid",
            device="cpu/gpu",
        )
        registry.register_provider(
            "darkir",
            lambda: getattr(getattr(legacy, "advanced_intelligence", None), "darkir", None),
            role="enhancement",
            device=str(getattr(legacy, "DARKIR_DEVICE", "auto")),
        )
        registry.register_provider(
            "edsr",
            lambda: getattr(getattr(legacy, "advanced_intelligence", None), "enhancement", None),
            role="enhancement",
            device=str(getattr(legacy, "EDSR_DEVICE", "cpu")),
        )

    def status(self) -> dict[str, Any]:
        legacy = self._legacy
        values = {
            key: bool(legacy is not None and getattr(legacy, key, None) is not None)
            for key in self.MODEL_BINDINGS
        }
        ready = self._loaded and all(values.values())
        payload = RuntimeStatus(
            status="READY" if ready else "NOT_READY",
            loaded=self._loaded,
            person_detector=values["person_model"],
            pose_detector=values["pose_model"],
            crime_detector=values["crime_model"],
            ppe_detector=values["ppe_model"],
            vehicle_detector=values["vehicle_model"],
            error=self._error,
        ).__dict__
        payload["primary_backend"] = self._primary_backend
        payload["batching"] = {
            "batch_calls": self._batch_calls,
            "batch_frames": self._batch_frames,
            "batch_fallbacks": self._batch_fallbacks,
            "last_batch_ms": round(self._last_batch_ms, 3),
        }
        return payload

    def reset_camera(self, camera_id: str) -> None:
        if self._legacy is None:
            return
        reset = getattr(self._legacy, "_reset_camera_scene_state", None)
        if callable(reset):
            reset(str(camera_id), reason="pipeline_stopped")
        gate = getattr(self._legacy, "_face_darkir_gate_instance", None)
        if gate is not None:
            gate.clear_camera(str(camera_id))
        try:
            from inference.policy import CAMERA_POLICIES, BEHAVIOUR_GATE
            from inference.track_cache import TRACK_CACHE
            CAMERA_POLICIES.clear(str(camera_id))
            BEHAVIOUR_GATE.clear_camera(str(camera_id))
            TRACK_CACHE.clear_camera(str(camera_id))
        except Exception:
            logger.debug("Inference camera cache cleanup failed", exc_info=True)
