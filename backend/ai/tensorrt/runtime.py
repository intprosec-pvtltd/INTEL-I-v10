from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .manifest import gpu_name, tensorrt_version, validate_manifest


class TensorRTEngineRuntime:
    """Fail-closed TensorRT runtime registry.

    This class deliberately does not auto-build or silently substitute a PyTorch
    model. Engine loading is explicit, manifest-backed, checksum verified, and
    lazy so a CPU-only control-plane startup remains possible.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("TENSORRT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.required = os.getenv("TENSORRT_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.manifest_path = Path(os.getenv("TENSORRT_MANIFEST", "models/tensorrt/manifest.json"))
        self._lock = threading.RLock()
        self._engines: dict[str, Any] = {}
        self._errors: list[str] = []
        self._loaded = False

    def status(self, validate_files: bool = True) -> dict[str, Any]:
        with self._lock:
            validation = validate_manifest(self.manifest_path, require_engine=validate_files) if self.manifest_path.is_file() else {
                "ok": False,
                "errors": [f"TensorRT manifest missing: {self.manifest_path}"],
                "engines": [],
            }
            ready = bool(self.enabled and validation["ok"])
            return {
                "enabled": self.enabled,
                "required": self.required,
                "ready": ready,
                "manifest": str(self.manifest_path),
                "manifest_validation": validation,
                "loaded_roles": sorted(self._engines),
                "gpu_name": gpu_name(),
                "tensorrt_version": tensorrt_version(),
                "platform": os.name,
                "python": os.sys.version.split()[0],
                "error_count": len(self._errors),
                "errors": list(self._errors[-20:]),
            }

    def require_ready(self) -> None:
        state = self.status(validate_files=True)
        if self.required and not state["ready"]:
            raise RuntimeError("TensorRT is required but the validated engine manifest is not ready")

    def load_engine(self, role: str) -> Any:
        with self._lock:
            if role in self._engines:
                return self._engines[role]
            if not self.enabled:
                raise RuntimeError("TensorRT runtime is disabled")
            if not self.manifest_path.is_file():
                raise RuntimeError(f"TensorRT manifest missing: {self.manifest_path}")
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            entry = next((x for x in data.get("engines", []) if x.get("role") == role), None)
            if not entry:
                raise RuntimeError(f"No TensorRT engine registered for role={role}")
            from .manifest import validate_manifest_entry
            ok, errors = validate_manifest_entry(entry, require_engine=True)
            if not ok:
                raise RuntimeError("TensorRT engine validation failed: " + "; ".join(errors))
            try:
                import tensorrt as trt  # type: ignore
            except Exception as exc:
                self._errors.append(f"TensorRT Python bindings unavailable: {exc}")
                raise RuntimeError("TensorRT Python bindings are required for engine loading") from exc

            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            engine_bytes = Path(entry["engine_path"]).read_bytes()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            if engine is None:
                raise RuntimeError(f"TensorRT failed to deserialize engine: {entry['engine_path']}")
            self._engines[role] = engine
            self._loaded = True
            return engine


runtime = TensorRTEngineRuntime()
