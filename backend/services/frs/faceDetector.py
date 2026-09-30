from __future__ import annotations

import numpy as np


class OnnxFaceDetector:
    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        import onnxruntime as ort
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input = self.session.get_inputs()[0]

    def status(self) -> dict:
        active = self.session.get_providers()
        return {
            "status": "READY" if active else "NOT_READY",
            "providers": active,
            "cuda_primary": bool(active and active[0] == "CUDAExecutionProvider"),
        }

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input.name: np.ascontiguousarray(tensor)})
