from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


# ============================================================
# BASE DIRECTORY / ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = BASE_DIR / "models" / "EDSR_x4.pb"

# Supported values:
#
#   cuda / gpu / cuda:0
#       Require CUDA execution.
#
#   cpu
#       Force CPU execution.
#
#   auto
#       Use CUDA when available, otherwise CPU.
#
MODEL_ENHANCEMENT_DEVICE = (
    os.getenv(
        "MODEL_ENHANCEMENT_DEVICE",
        "auto",
    )
    .strip()
    .lower()
)

# CUDA device ID.
try:
    CUDA_DEVICE_ID = max(
        0,
        int(
            os.getenv(
                "MODEL_ENHANCEMENT_CUDA_DEVICE_ID",
                "0",
            ).strip()
        ),
    )
except (TypeError, ValueError):
    CUDA_DEVICE_ID = 0

# Whether CUDA should be treated as mandatory when requested.
MODEL_ENHANCEMENT_STRICT_GPU = (
    os.getenv(
        "MODEL_ENHANCEMENT_STRICT_GPU",
        "true",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


# ============================================================
# MODEL STATE
# ============================================================

_model_lock = threading.Lock()

_sr_model = None

_load_attempted = False

_load_error: Optional[str] = None

_actual_device = "unloaded"

_actual_backend = "unloaded"

_actual_target = "unloaded"


# ============================================================
# DEVICE NORMALIZATION
# ============================================================

def _normalize_device(value: str) -> str:
    """
    Normalize configuration values into:

        cpu
        cuda
        auto
    """

    device = str(value or "").strip().lower()

    if not device:
        return "auto"

    if device in {
        "cuda",
        "gpu",
        "cuda:0",
        "gpu:0",
    }:
        return "cuda"

    if device == "cpu":
        return "cpu"

    if device == "auto":
        return "auto"

    logger.warning(
        "[INTEL-I][SUPERRES] Unknown MODEL_ENHANCEMENT_DEVICE=%r. "
        "Using auto.",
        value,
    )

    return "auto"


# Normalize once at module load.
_REQUESTED_DEVICE = _normalize_device(
    MODEL_ENHANCEMENT_DEVICE
)


# ============================================================
# CUDA AVAILABILITY
# ============================================================

def _cuda_device_count() -> int:
    """
    Return the CUDA device count reported by OpenCV.

    Returns 0 if the current OpenCV build does not expose
    CUDA support or the CUDA query fails.
    """

    try:
        count = int(
            cv2.cuda.getCudaEnabledDeviceCount()
        )

        return max(0, count)

    except Exception:
        logger.debug(
            "[INTEL-I][SUPERRES] OpenCV CUDA query failed.",
            exc_info=True,
        )
        return 0


def _cuda_available() -> bool:
    """
    Check whether OpenCV can actually use a CUDA device.
    """

    count = _cuda_device_count()

    return (
        count > CUDA_DEVICE_ID
    )


def _select_runtime_device() -> str:
    """
    Determine the actual execution device.

    Returns:

        cuda
        cpu

    Rules:

        MODEL_ENHANCEMENT_DEVICE=cpu
            -> CPU

        MODEL_ENHANCEMENT_DEVICE=cuda
            -> CUDA if available
            -> raise if unavailable and strict mode is enabled
            -> otherwise CPU

        MODEL_ENHANCEMENT_DEVICE=auto
            -> CUDA if available
            -> otherwise CPU
    """

    global _load_error

    if _REQUESTED_DEVICE == "cpu":
        return "cpu"

    cuda_available = _cuda_available()

    if _REQUESTED_DEVICE == "cuda":
        if cuda_available:
            return "cuda"

        message = (
            "MODEL_ENHANCEMENT_DEVICE=cuda was requested, "
            "but the current OpenCV build does not expose "
            f"CUDA device {CUDA_DEVICE_ID}. "
            f"OpenCV CUDA device count={_cuda_device_count()}."
        )

        if MODEL_ENHANCEMENT_STRICT_GPU:
            raise RuntimeError(
                message
            )

        logger.warning(
            "[INTEL-I][SUPERRES] %s Falling back to CPU.",
            message,
        )

        return "cpu"

    # auto
    if cuda_available:
        return "cuda"

    return "cpu"


# ============================================================
# MODEL LOADING
# ============================================================

def load_superres_model() -> bool:
    """
    Load the EDSR x4 super-resolution model exactly once.

    The model is configured for:

        CUDA -> cv2.dnn.DNN_BACKEND_CUDA
                cv2.dnn.DNN_TARGET_CUDA

        CPU  -> cv2.dnn.DNN_BACKEND_OPENCV
                cv2.dnn.DNN_TARGET_CPU
    """

    global _sr_model
    global _load_attempted
    global _load_error
    global _actual_device
    global _actual_backend
    global _actual_target

    with _model_lock:

        # ----------------------------------------------------
        # Already loaded
        # ----------------------------------------------------

        if _sr_model is not None:
            return True

        # ----------------------------------------------------
        # Previous load failed
        # ----------------------------------------------------

        if _load_attempted:
            return False

        _load_attempted = True

        # ----------------------------------------------------
        # Validate model file
        # ----------------------------------------------------

        if not MODEL_PATH.is_file():

            _load_error = (
                f"Model not found: {MODEL_PATH}"
            )

            logger.error(
                "[INTEL-I][SUPERRES] %s",
                _load_error,
            )

            _actual_device = "unavailable"
            _actual_backend = "unavailable"
            _actual_target = "unavailable"

            return False

        # ----------------------------------------------------
        # Import dnn_superres
        # ----------------------------------------------------

        try:

            from cv2 import dnn_superres

        except Exception as exc:

            _load_error = (
                f"OpenCV dnn_superres unavailable: {exc}"
            )

            logger.exception(
                "[INTEL-I][SUPERRES] DNN_SUPERRES_IMPORT_FAILED"
            )

            _actual_device = "unavailable"
            _actual_backend = "unavailable"
            _actual_target = "unavailable"

            return False

        # ----------------------------------------------------
        # Select runtime device
        # ----------------------------------------------------

        try:

            runtime_device = _select_runtime_device()

        except Exception as exc:

            _load_error = str(exc)

            logger.exception(
                "[INTEL-I][SUPERRES] GPU_REQUEST_FAILED "
                "requested_device=%s",
                _REQUESTED_DEVICE,
            )

            _actual_device = "unavailable"
            _actual_backend = "unavailable"
            _actual_target = "unavailable"

            return False

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        try:

            model = (
                dnn_superres.DnnSuperResImpl_create()
            )

            model.readModel(
                str(MODEL_PATH)
            )

            model.setModel(
                "edsr",
                4,
            )

            # ------------------------------------------------
            # CUDA
            # ------------------------------------------------

            if runtime_device == "cuda":

                try:

                    model.setPreferableBackend(
                        cv2.dnn.DNN_BACKEND_CUDA
                    )

                    # Use device ID explicitly when supported.
                    try:
                        model.setPreferableTarget(
                            cv2.dnn.DNN_TARGET_CUDA,
                        )
                    except TypeError:
                        # Older OpenCV APIs may not accept
                        # a CUDA device argument here.
                        model.setPreferableTarget(
                            cv2.dnn.DNN_TARGET_CUDA
                        )

                    _actual_device = (
                        f"cuda:{CUDA_DEVICE_ID}"
                    )

                    _actual_backend = (
                        "DNN_BACKEND_CUDA"
                    )

                    _actual_target = (
                        "DNN_TARGET_CUDA"
                    )

                    logger.info(
                        "[INTEL-I][SUPERRES] "
                        "CUDA backend selected | "
                        "model=EDSR_x4 | "
                        "device=%s | "
                        "cuda_devices=%s",
                        _actual_device,
                        _cuda_device_count(),
                    )

                except Exception as exc:

                    message = (
                        "OpenCV CUDA backend could not be "
                        f"configured: {exc}"
                    )

                    if MODEL_ENHANCEMENT_STRICT_GPU:

                        _load_error = message

                        logger.exception(
                            "[INTEL-I][SUPERRES] "
                            "CUDA_BACKEND_INIT_FAILED"
                        )

                        _actual_device = "unavailable"
                        _actual_backend = "unavailable"
                        _actual_target = "unavailable"

                        return False

                    logger.warning(
                        "[INTEL-I][SUPERRES] %s "
                        "Falling back to CPU.",
                        message,
                    )

                    runtime_device = "cpu"

            # ------------------------------------------------
            # CPU
            # ------------------------------------------------

            if runtime_device == "cpu":

                model.setPreferableBackend(
                    cv2.dnn.DNN_BACKEND_OPENCV
                )

                model.setPreferableTarget(
                    cv2.dnn.DNN_TARGET_CPU
                )

                _actual_device = "cpu"

                _actual_backend = (
                    "DNN_BACKEND_OPENCV"
                )

                _actual_target = (
                    "DNN_TARGET_CPU"
                )

                logger.info(
                    "[INTEL-I][SUPERRES] "
                    "CPU backend selected | "
                    "model=EDSR_x4",
                )

            # ------------------------------------------------
            # Store model
            # ------------------------------------------------

            _sr_model = model

            _load_error = None

            logger.info(
                "[INTEL-I][SUPERRES] MODEL_READY | "
                "model=EDSR_x4 | "
                "path=%s | "
                "requested_device=%s | "
                "actual_device=%s | "
                "backend=%s | "
                "target=%s",
                MODEL_PATH,
                _REQUESTED_DEVICE,
                _actual_device,
                _actual_backend,
                _actual_target,
            )

            return True

        except Exception as exc:

            _load_error = str(exc)

            _actual_device = "unavailable"
            _actual_backend = "unavailable"
            _actual_target = "unavailable"

            logger.exception(
                "[INTEL-I][SUPERRES] MODEL_INIT_FAILED | "
                "model=%s | path=%s | "
                "requested_device=%s",
                "EDSR_x4",
                MODEL_PATH,
                _REQUESTED_DEVICE,
            )

            _sr_model = None

            return False


# ============================================================
# MODEL AVAILABILITY
# ============================================================

def superres_available() -> bool:
    """
    Return whether the EDSR model is loaded.
    """

    return _sr_model is not None


# ============================================================
# MODEL STATUS
# ============================================================

def superres_status() -> dict:
    """
    Return detailed runtime information.

    Important:
        'device' represents the device actually selected
        by the model loader, not merely the requested .env value.
    """

    return {
        "enabled": True,

        "model": "EDSR_x4",

        "model_path": str(
            MODEL_PATH
        ),

        "model_exists": MODEL_PATH.is_file(),

        "model_loaded": (
            _sr_model is not None
        ),

        "requested_device": (
            _REQUESTED_DEVICE
        ),

        "device": (
            _actual_device
        ),

        "backend": (
            _actual_backend
        ),

        "target": (
            _actual_target
        ),

        "cuda_device_id": (
            CUDA_DEVICE_ID
        ),

        "opencv_cuda_devices": (
            _cuda_device_count()
        ),

        "opencv_cuda_available": (
            _cuda_available()
        ),

        "strict_gpu": (
            MODEL_ENHANCEMENT_STRICT_GPU
        ),

        "load_attempted": (
            _load_attempted
        ),

        "load_error": (
            _load_error
        ),
    }


# ============================================================
# IMAGE UPSCALING
# ============================================================

def upscale_crop(
    image: np.ndarray,
    *,
    max_input_width: int = 320,
    max_input_height: int = 180,
) -> np.ndarray:
    """
    Neural 4x EDSR super-resolution for small regions.

    Large input crops are reduced before EDSR to avoid excessive
    GPU/CPU memory usage.
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not isinstance(
        image,
        np.ndarray,
    ):
        return image

    if image.size == 0:
        return image

    if image.ndim not in (
        2,
        3,
    ):
        return image

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not load_superres_model():
        return image

    # --------------------------------------------------------
    # Prepare working image
    # --------------------------------------------------------

    try:

        h, w = image.shape[:2]

        if (
            w > max_input_width
            or h > max_input_height
        ):

            scale = min(
                max_input_width / float(w),
                max_input_height / float(h),
            )

            new_w = max(
                1,
                int(
                    round(
                        w * scale
                    )
                ),
            )

            new_h = max(
                1,
                int(
                    round(
                        h * scale
                    )
                ),
            )

            working = cv2.resize(
                image,
                (
                    new_w,
                    new_h,
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:

            working = image

        # ----------------------------------------------------
        # Run EDSR
        # ----------------------------------------------------

        with _model_lock:

            if _sr_model is None:
                return image

            result = _sr_model.upsample(
                working
            )

        # ----------------------------------------------------
        # Validate result
        # ----------------------------------------------------

        if not isinstance(
            result,
            np.ndarray,
        ):
            return image

        if result.size == 0:
            return image

        return np.ascontiguousarray(
            np.clip(
                result,
                0,
                255,
            ).astype(
                np.uint8
            )
        )

    except Exception:

        logger.exception(
            "[INTEL-I][SUPERRES] "
            "INFERENCE_FAILED | "
            "device=%s | backend=%s | target=%s",
            _actual_device,
            _actual_backend,
            _actual_target,
        )

        return image


# ============================================================
# PLATE ENHANCEMENT
# ============================================================

def enhance_plate_crop(
    plate_crop: np.ndarray,
) -> np.ndarray:
    """
    Enhance an ANPR plate crop.

    Processing:

        1. EDSR 4x super-resolution
        2. Grayscale conversion
        3. CLAHE
        4. Return BGR image
    """

    if not isinstance(
        plate_crop,
        np.ndarray,
    ):
        return plate_crop

    if plate_crop.size == 0:
        return plate_crop

    # --------------------------------------------------------
    # EDSR
    # --------------------------------------------------------

    upscaled = upscale_crop(
        plate_crop,
        max_input_width=320,
        max_input_height=128,
    )

    # --------------------------------------------------------
    # CLAHE post-processing
    # --------------------------------------------------------

    try:

        if upscaled.ndim == 2:

            gray = upscaled

        else:

            gray = cv2.cvtColor(
                upscaled,
                cv2.COLOR_BGR2GRAY,
            )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(
                8,
                8,
            ),
        )

        enhanced = clahe.apply(
            gray
        )

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR,
        )

    except Exception:

        logger.exception(
            "[INTEL-I][SUPERRES] "
            "PLATE_POSTPROCESS_FAILED"
        )

        return upscaled


# ============================================================
# GENERIC VEHICLE ENHANCEMENT
# ============================================================

def enhance_vehicle_crop(
    vehicle_crop: np.ndarray,
) -> np.ndarray:
    """
    Enhance a vehicle crop using EDSR.

    This is intentionally bounded because EDSR x4 can become
    expensive on large vehicle crops.
    """

    if not isinstance(
        vehicle_crop,
        np.ndarray,
    ):
        return vehicle_crop

    if vehicle_crop.size == 0:
        return vehicle_crop

    return upscale_crop(
        vehicle_crop,
        max_input_width=320,
        max_input_height=180,
    )


# ============================================================
# GENERIC IMAGE ENHANCEMENT
# ============================================================

def enhance_image(
    image: np.ndarray,
    *,
    max_input_width: int = 320,
    max_input_height: int = 180,
) -> np.ndarray:
    """
    Generic EDSR enhancement API.
    """

    return upscale_crop(
        image,
        max_input_width=max_input_width,
        max_input_height=max_input_height,
    )


# ============================================================
# MODEL RESET
# ============================================================

def reset_superres_model() -> None:
    """
    Release the current EDSR model state.

    Useful during configuration changes or development reloads.
    """

    global _sr_model
    global _load_attempted
    global _load_error
    global _actual_device
    global _actual_backend
    global _actual_target

    with _model_lock:

        _sr_model = None

        _load_attempted = False

        _load_error = None

        _actual_device = "unloaded"

        _actual_backend = "unloaded"

        _actual_target = "unloaded"

    logger.info(
        "[INTEL-I][SUPERRES] MODEL_STATE_RESET"
    )