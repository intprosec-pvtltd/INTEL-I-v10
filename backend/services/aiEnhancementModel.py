from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EDSR4x:
    """
    OpenCV DNN Super Resolution wrapper for EDSR_x4.pb.

    Important:
    OpenCV's built-in DNN backend must not be paired with DNN_TARGET_CUDA.
    This implementation therefore uses the OpenCV backend with CPU for EDSR.

    Other INTEL-I models such as YOLO, ReID, ANPR/ONNX Runtime can continue
    using CUDA independently.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.requested_device = str(device or "cpu").lower()

        self.model = None
        self.loaded_device = None

        self._load_failed = False
        self._inference_failed = False

    def load(self) -> bool:
        """
        Load EDSR safely.

        Returns True only when the model is ready for inference.
        """

        if self.model is not None:
            return True

        # Avoid repeatedly trying to load a model that already failed.
        if self._load_failed:
            return False

        if not self.model_path.is_file():
            logger.error(
                "EDSR model not found: %s",
                self.model_path,
            )
            self._load_failed = True
            return False

        try:
            from cv2 import dnn_superres

            sr = dnn_superres.DnnSuperResImpl_create()

            sr.readModel(str(self.model_path))
            sr.setModel("edsr", 4)

            # -----------------------------------------------------
            # IMPORTANT
            # -----------------------------------------------------
            # DNN_BACKEND_OPENCV + DNN_TARGET_CUDA is invalid.
            #
            # Use the OpenCV backend with CPU for this model.
            # This does NOT disable CUDA for the rest of INTEL-I.
            # -----------------------------------------------------

            sr.setPreferableBackend(
                cv2.dnn.DNN_BACKEND_OPENCV
            )

            sr.setPreferableTarget(
                cv2.dnn.DNN_TARGET_CPU
            )

            self.model = sr
            self.loaded_device = "cpu"

            if self.requested_device.startswith("cuda"):
                logger.info(
                    "EDSR requested CUDA but OpenCV "
                    "DNN Super Resolution is using CPU "
                    "to avoid an unsupported "
                    "DNN_BACKEND_OPENCV/CUDA target combination."
                )
            else:
                logger.info(
                    "EDSR loaded successfully on CPU: %s",
                    self.model_path,
                )

            return True

        except Exception:
            self.model = None
            self.loaded_device = None
            self._load_failed = True

            logger.exception(
                "Failed to load EDSR model: %s",
                self.model_path,
            )

            return False

    @staticmethod
    def _valid_image(
        image: np.ndarray,
    ) -> bool:
        if not isinstance(image, np.ndarray):
            return False

        if image.size == 0:
            return False

        if image.ndim not in (2, 3):
            return False

        if image.shape[0] <= 0 or image.shape[1] <= 0:
            return False

        return True

    @staticmethod
    def _prepare_image(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert the frame into a safe contiguous uint8 image
        accepted by OpenCV DNN Super Resolution.
        """

        if image.dtype != np.uint8:
            image = np.clip(
                image,
                0,
                255,
            ).astype(np.uint8)

        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        return image

    def upscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Upscale an image using EDSR.

        On any model/inference failure the original image is returned.
        The video analytics pipeline therefore continues operating.
        """

        if not self._valid_image(image):
            logger.warning(
                "EDSR received an invalid or empty image"
            )
            return image

        if not self.load():
            return image

        # If inference already failed, don't repeatedly execute the
        # same broken model for every CCTV frame.
        if self._inference_failed:
            return image

        try:
            prepared = self._prepare_image(image)

            enhanced = self.model.upsample(
                prepared
            )

            if enhanced is None:
                raise RuntimeError(
                    "EDSR returned no output"
                )

            if not isinstance(
                enhanced,
                np.ndarray,
            ):
                raise RuntimeError(
                    "EDSR returned an invalid output type"
                )

            if enhanced.size == 0:
                raise RuntimeError(
                    "EDSR returned an empty image"
                )

            enhanced = np.clip(
                enhanced,
                0,
                255,
            ).astype(
                np.uint8,
                copy=False,
            )

            return np.ascontiguousarray(
                enhanced
            )

        except cv2.error as exc:
            self._inference_failed = True

            logger.error(
                "EDSR OpenCV inference failed; "
                "EDSR is disabled for this process and "
                "the original frame will be used. Error: %s",
                exc,
            )

            return image

        except Exception:
            self._inference_failed = True

            logger.exception(
                "EDSR inference failed; "
                "EDSR is disabled for this process and "
                "the original frame will be used."
            )

            return image

    def status(self) -> Dict[str, Any]:
        return {
            "path": str(self.model_path),
            "exists": self.model_path.is_file(),
            "loaded": self.model is not None,
            "requested_device": self.requested_device,
            "loaded_device": self.loaded_device,
            "load_failed": self._load_failed,
            "inference_failed": self._inference_failed,
            "scale": 4,
            "model": "EDSR",
        }


class ONNXEnhancementModel:
    """
    Generic ONNX enhancement hook for a dedicated low-light/dehaze model.

    The exact preprocessing/postprocessing is model-specific.
    Provide a small adapter subclass for the selected model;
    do not guess tensor semantics.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.device = str(device or "cpu").lower()

        self.session = None
        self.loaded_provider = None
        self._load_failed = False

    def load(self) -> bool:
        if self.session is not None:
            return True

        if self._load_failed:
            return False

        if not self.model_path.is_file():
            logger.warning(
                "Enhancement model missing: %s",
                self.model_path,
            )
            self._load_failed = True
            return False

        try:
            import onnxruntime as ort

            available_providers = set(
                ort.get_available_providers()
            )

            providers = []

            if (
                self.device.startswith("cuda")
                and "CUDAExecutionProvider"
                in available_providers
            ):
                providers.append(
                    "CUDAExecutionProvider"
                )

            if (
                "CPUExecutionProvider"
                in available_providers
            ):
                providers.append(
                    "CPUExecutionProvider"
                )

            if not providers:
                logger.error(
                    "No usable ONNX Runtime execution "
                    "provider is available. Available: %s",
                    sorted(available_providers),
                )

                self._load_failed = True
                return False

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )

            active_providers = (
                self.session.get_providers()
            )

            self.loaded_provider = (
                active_providers[0]
                if active_providers
                else None
            )

            logger.info(
                "Enhancement ONNX model loaded | "
                "requested_device=%s | provider=%s | "
                "providers=%s",
                self.device,
                self.loaded_provider,
                active_providers,
            )

            return True

        except Exception:
            self.session = None
            self.loaded_provider = None
            self._load_failed = True

            logger.exception(
                "Failed to initialize ONNX "
                "enhancement model: %s",
                self.model_path,
            )

            return False

    def status(self) -> Dict[str, Any]:
        return {
            "path": str(self.model_path),
            "exists": self.model_path.is_file(),
            "loaded": self.session is not None,
            "device": self.device,
            "loaded_provider": self.loaded_provider,
            "load_failed": self._load_failed,
        }