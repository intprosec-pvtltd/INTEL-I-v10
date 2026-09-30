"""Secure, in-process adapter for the Awiros Indian ANPR OCR model.

The upstream release ships PaddlePaddle weights in SafeTensors format and
uses the PP-OCRv5 server-recognition architecture.  INTEL-I never clones code
or downloads weights at runtime: operators must provision both the model files
and a reviewed PaddleOCR source tree before startup.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np


logger = logging.getLogger(__name__)

_PLATE_TEXT = re.compile(r"[^A-Z0-9]")
_IMAGE_SHAPE = (3, 48, 320)
_MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": 64,
                "NRTRLabelDecode": 67,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 1e-5},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    }
}


def _local_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _paddle_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not (resolved / "ppocr" / "__init__.py").is_file():
        raise FileNotFoundError(
            "ANPR_AWIROS_PADDLEOCR_DIR must point to a reviewed PaddleOCR "
            f"source checkout containing ppocr/__init__.py: {resolved}"
        )
    return resolved


def _preprocess(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Awiros OCR received an empty plate crop")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Awiros OCR expects a three-channel BGR plate crop")

    _, target_h, target_w = _IMAGE_SHAPE
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Awiros OCR received invalid plate dimensions")

    resized_w = max(1, min(target_w, int(round(width * target_h / height))))
    resized = cv2.resize(image, (resized_w, target_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    canvas[:, :resized_w] = resized.astype(np.float32) / 255.0
    canvas = (canvas - 0.5) / 0.5
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None, ...])


class AwirosOCR:
    """Lazy, thread-safe Awiros recognizer suitable for FastAPI workers."""

    def __init__(
        self,
        *,
        weights_path: str | os.PathLike[str],
        dictionary_path: str | os.PathLike[str],
        paddleocr_dir: str | os.PathLike[str],
        device: str = "auto",
        strict_device: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.dictionary_path = Path(dictionary_path)
        self.paddleocr_dir = Path(paddleocr_dir)
        self.requested_device = str(device or "auto").strip().lower()
        self.strict_device = bool(strict_device)
        self.actual_device = "uninitialized"
        self.version = "Awiros-ANPR-OCR/PP-OCRv5-SafeTensors"
        self._model: Any = None
        self._post_process: Any = None
        self._paddle: Any = None
        self._lock = threading.RLock()

    def load(self) -> "AwirosOCR":
        if self._model is not None:
            return self

        with self._lock:
            if self._model is not None:
                return self

            weights = _local_file(self.weights_path, "Awiros model.safetensors")
            dictionary = _local_file(self.dictionary_path, "Awiros en_dict.txt")
            paddle_root = _paddle_root(self.paddleocr_dir)
            root_text = str(paddle_root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)

            try:
                import paddle
                from ppocr.modeling.architectures import build_model
                from ppocr.postprocess import build_post_process
                from safetensors.numpy import load_file
            except Exception as exc:
                raise RuntimeError(
                    "Awiros OCR dependencies are unavailable. Install the pinned "
                    "PaddlePaddle build and provision a compatible PaddleOCR checkout."
                ) from exc

            wants_gpu = self.requested_device in {"auto", "cuda", "gpu", "0"}
            gpu_available = bool(paddle.is_compiled_with_cuda())
            if wants_gpu and gpu_available:
                paddle.set_device("gpu")
                self.actual_device = "gpu"
            elif self.requested_device in {"cuda", "gpu", "0"} and self.strict_device:
                raise RuntimeError("Awiros OCR GPU was required but PaddlePaddle CUDA is unavailable")
            else:
                paddle.set_device("cpu")
                self.actual_device = "cpu"

            post_process = build_post_process(
                {
                    "name": "CTCLabelDecode",
                    "character_dict_path": str(dictionary),
                    "use_space_char": True,
                }
            )
            model = build_model(copy.deepcopy(_MODEL_CONFIG["Architecture"]))
            state = {
                key: paddle.to_tensor(value)
                for key, value in load_file(str(weights)).items()
            }
            model.set_state_dict(state)
            model.eval()

            self._paddle = paddle
            self._post_process = post_process
            self._model = model
            logger.info(
                "Awiros Indian ANPR OCR ready | device=%s weights=%s",
                self.actual_device,
                weights.name,
            )
            return self

    def recognize(self, plate_crop: np.ndarray) -> tuple[str | None, float]:
        self.load()
        tensor = self._paddle.to_tensor(_preprocess(plate_crop))
        with self._lock, self._paddle.no_grad():
            predictions = self._model(tensor)

        if isinstance(predictions, dict):
            prediction = predictions.get("ctc", next(iter(predictions.values())))
        elif isinstance(predictions, (list, tuple)):
            prediction = predictions[0]
        else:
            prediction = predictions

        decoded = self._post_process(prediction.numpy())
        if not isinstance(decoded, (list, tuple)) or not decoded:
            return None, 0.0
        text, confidence = decoded[0]
        normalized = _PLATE_TEXT.sub("", str(text or "").upper())[:25]
        score = max(0.0, min(1.0, float(confidence or 0.0)))
        return (normalized or None), score

    def health(self) -> dict[str, Any]:
        return {
            "engine": "awiros_paddle",
            "model": "Awiros-ANPR-OCR",
            "loaded": self._model is not None,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "weights_present": self.weights_path.expanduser().is_file(),
            "dictionary_present": self.dictionary_path.expanduser().is_file(),
            "paddleocr_present": (
                self.paddleocr_dir.expanduser() / "ppocr" / "__init__.py"
            ).is_file(),
        }


__all__ = ["AwirosOCR", "_preprocess"]
