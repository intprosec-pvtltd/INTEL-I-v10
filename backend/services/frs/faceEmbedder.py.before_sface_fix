from __future__ import annotations

import cv2
import numpy as np


class OnnxFaceEmbedder:
    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        import onnxruntime as ort
        requested = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=requested)
        self.input = self.session.get_inputs()[0]

    @property
    def active_providers(self) -> list[str]:
        return list(self.session.get_providers())

    @staticmethod
    def preprocess(face: np.ndarray) -> np.ndarray:
        if not isinstance(face, np.ndarray) or face.size == 0:
            raise ValueError("face image is empty")
        image = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - 127.5) / 127.5
        return np.transpose(image, (2, 0, 1))[None, ...]

    def embed(self, face: np.ndarray) -> np.ndarray:
        output = self.session.run(None, {self.input.name: self.preprocess(face)})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("face model returned an invalid embedding")
        return vector / norm
