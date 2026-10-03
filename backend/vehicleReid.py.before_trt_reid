from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

# ============================================================
# RUNTIME LOG CONTROL
# Must be set before importing ONNX Runtime.
# ============================================================

os.environ.setdefault(
    "ORT_LOG_SEVERITY_LEVEL",
    "3",
)

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "VEHICLE_REID_MODEL",
    "models/osnet_ain_x1_0_vehicle_reid.onnx",
).strip()


# Default values match the current INTEL-I model.
DEFAULT_INPUT_WIDTH = 208
DEFAULT_INPUT_HEIGHT = 208
EXPECTED_EMBEDDING_DIM = 512


def _safe_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


INPUT_WIDTH = _safe_positive_int(
    "VEHICLE_REID_INPUT_WIDTH",
    DEFAULT_INPUT_WIDTH,
)

INPUT_HEIGHT = _safe_positive_int(
    "VEHICLE_REID_INPUT_HEIGHT",
    DEFAULT_INPUT_HEIGHT,
)

EMBEDDING_DIM = _safe_positive_int(
    "VEHICLE_REID_EMBEDDING_DIM",
    EXPECTED_EMBEDDING_DIM,
)


MAX_CROP_WIDTH = int(
    os.getenv(
        "VEHICLE_REID_MAX_CROP_WIDTH",
        "4096",
    )
)

MAX_CROP_HEIGHT = int(
    os.getenv(
        "VEHICLE_REID_MAX_CROP_HEIGHT",
        "4096",
    )
)


MIN_CROP_WIDTH = int(
    os.getenv(
        "VEHICLE_REID_MIN_CROP_WIDTH",
        "32",
    )
)

MIN_CROP_HEIGHT = int(
    os.getenv(
        "VEHICLE_REID_MIN_CROP_HEIGHT",
        "32",
    )
)


# ============================================================
# INFERENCE CONFIGURATION
# ============================================================

REID_PROVIDER = os.getenv(
    "VEHICLE_REID_PROVIDER",
    "auto",
).strip().lower()


REID_ENABLE_CUDA = (
    os.getenv(
        "VEHICLE_REID_ENABLE_CUDA",
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


REID_ENABLE_CPU = (
    os.getenv(
        "VEHICLE_REID_ENABLE_CPU",
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
# RE-ID THRESHOLDS
# ============================================================

DEFAULT_REID_THRESHOLD = float(
    os.getenv(
        "VEHICLE_REID_THRESHOLD",
        "0.75",
    )
)


REID_POSSIBLE_THRESHOLD = float(
    os.getenv(
        "VEHICLE_REID_POSSIBLE_THRESHOLD",
        "0.65",
    )
)


REID_PROBABLE_THRESHOLD = float(
    os.getenv(
        "VEHICLE_REID_PROBABLE_THRESHOLD",
        "0.75",
    )
)


REID_STRONG_THRESHOLD = float(
    os.getenv(
        "VEHICLE_REID_STRONG_THRESHOLD",
        "0.85",
    )
)


# ============================================================
# QUALITY CONFIGURATION
# ============================================================

MIN_BLUR_SCORE = float(
    os.getenv(
        "VEHICLE_REID_MIN_BLUR_SCORE",
        "15.0",
    )
)


MIN_BRIGHTNESS = float(
    os.getenv(
        "VEHICLE_REID_MIN_BRIGHTNESS",
        "15.0",
    )
)


MAX_BRIGHTNESS = float(
    os.getenv(
        "VEHICLE_REID_MAX_BRIGHTNESS",
        "245.0",
    )
)


# ============================================================
# INTERNAL STATE
# ============================================================

_session = None

_session_lock = threading.RLock()

_inference_lock = threading.RLock()

_model_metadata_lock = threading.RLock()

_model_metadata: Optional[
    dict[str, Any]
] = None


# ============================================================
# MODEL VALIDATION
# ============================================================

def _validate_model_path(
    model_path: str,
) -> str:

    if not model_path:

        raise RuntimeError(
            "VEHICLE_REID_MODEL is not configured"
        )

    path = (
        Path(
            model_path
        )
        .expanduser()
        .resolve()
    )

    if path.suffix.lower() != ".onnx":

        raise RuntimeError(
            "Vehicle Re-ID model must be an ONNX file"
        )

    if not path.exists():

        raise RuntimeError(
            "Vehicle Re-ID model file not found: "
            f"{path}"
        )

    if not path.is_file():

        raise RuntimeError(
            "Vehicle Re-ID model path is not a regular file"
        )

    return str(
        path
    )


# ============================================================
# PROVIDER SELECTION
# ============================================================

def _build_execution_providers(
    available_providers,
) -> list[str]:

    available = set(
        available_providers
    )

    providers = []

    if (
        REID_PROVIDER
        == "cpu"
    ):

        if (
            REID_ENABLE_CPU
            and "CPUExecutionProvider"
            in available
        ):

            providers.append(
                "CPUExecutionProvider"
            )

        return providers

    if (
        REID_PROVIDER
        == "cuda"
    ):

        if (
            REID_ENABLE_CUDA
            and "CUDAExecutionProvider"
            in available
        ):

            providers.append(
                "CUDAExecutionProvider"
            )

        if (
            REID_ENABLE_CPU
            and "CPUExecutionProvider"
            in available
        ):

            providers.append(
                "CPUExecutionProvider"
            )

        return providers

    # AUTO
    if (
        REID_ENABLE_CUDA
        and "CUDAExecutionProvider"
        in available
    ):

        providers.append(
            "CUDAExecutionProvider"
        )

    if (
        REID_ENABLE_CPU
        and "CPUExecutionProvider"
        in available
    ):

        providers.append(
            "CPUExecutionProvider"
        )

    return providers


# ============================================================
# ONNX SESSION
# ============================================================

def get_reid_session():

    global _session
    global _model_metadata

    if _session is not None:

        return _session

    with _session_lock:

        if _session is not None:

            return _session

        validated_model_path = (
            _validate_model_path(
                MODEL_PATH
            )
        )

        available_providers = (
            ort.get_available_providers()
        )

        providers = (
            _build_execution_providers(
                available_providers
            )
        )

        if not providers:

            raise RuntimeError(
                "No supported ONNX Runtime "
                "execution provider is available"
            )

        session_options = (
            ort.SessionOptions()
        )

        # Avoid unnecessary thread explosion
        # when many CCTV workers share the process.
        session_options.intra_op_num_threads = int(
            os.getenv(
                "VEHICLE_REID_INTRA_OP_THREADS",
                "0",
            )
        )

        session_options.inter_op_num_threads = int(
            os.getenv(
                "VEHICLE_REID_INTER_OP_THREADS",
                "0",
            )
        )

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:

            session = ort.InferenceSession(
                validated_model_path,
                sess_options=session_options,
                providers=providers,
            )

        except Exception as exc:

            logger.exception(
                "Vehicle Re-ID model initialization "
                "failed with providers=%s",
                providers,
            )

            # Safe CPU fallback.
            if (
                "CPUExecutionProvider"
                in available_providers
                and "CPUExecutionProvider"
                not in providers
            ):

                try:

                    session = (
                        ort.InferenceSession(
                            validated_model_path,
                            sess_options=session_options,
                            providers=[
                                "CPUExecutionProvider"
                            ],
                        )
                    )

                    logger.warning(
                        "Vehicle Re-ID fell back "
                        "to CPU execution"
                    )

                except Exception as cpu_exc:

                    raise RuntimeError(
                        "Vehicle Re-ID model "
                        "initialization failed"
                    ) from cpu_exc

            else:

                raise RuntimeError(
                    "Vehicle Re-ID model "
                    "initialization failed"
                ) from exc

        inputs = session.get_inputs()

        outputs = session.get_outputs()

        if not inputs:

            raise RuntimeError(
                "Vehicle Re-ID ONNX model "
                "has no inputs"
            )

        if not outputs:

            raise RuntimeError(
                "Vehicle Re-ID ONNX model "
                "has no outputs"
            )

        input_info = inputs[0]

        output_info = outputs[0]

        _session = session

        _model_metadata = {
            "model_path":
                validated_model_path,

            "providers":
                list(
                    session.get_providers()
                ),

            "input_name":
                input_info.name,

            "input_shape":
                list(
                    input_info.shape
                ),

            "input_type":
                input_info.type,

            "output_name":
                output_info.name,

            "output_shape":
                list(
                    output_info.shape
                ),

            "output_type":
                output_info.type,
        }

        logger.info(
            "Vehicle Re-ID model loaded "
            "providers=%s input=%s output=%s",
            session.get_providers(),
            input_info.name,
            output_info.name,
        )

        return _session


# ============================================================
# MODEL METADATA
# ============================================================

def get_reid_model_metadata() -> dict[str, Any]:

    if _model_metadata is None:

        get_reid_session()

    with _model_metadata_lock:

        return dict(
            _model_metadata
            or {}
        )


# ============================================================
# INPUT SHAPE RESOLUTION
# ============================================================

def _resolve_input_dimensions(
    session,
) -> tuple[int, int]:

    try:

        input_info = (
            session.get_inputs()[0]
        )

        shape = list(
            input_info.shape
        )

        # Typical NCHW:
        #
        # [1, 3, H, W]
        #
        if len(shape) == 4:

            height = shape[2]
            width = shape[3]

            if (
                isinstance(
                    height,
                    int,
                )
                and isinstance(
                    width,
                    int,
                )
                and height > 0
                and width > 0
            ):

                return (
                    height,
                    width,
                )

    except Exception:

        pass

    return (
        INPUT_HEIGHT,
        INPUT_WIDTH,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_crop(
    crop,
) -> bool:

    if crop is None:

        return False

    if not isinstance(
        crop,
        np.ndarray,
    ):

        return False

    if crop.size == 0:

        return False

    if crop.ndim != 3:

        return False

    if crop.shape[2] != 3:

        return False

    height, width = (
        crop.shape[:2]
    )

    if (
        width <= 0
        or height <= 0
    ):

        return False

    if (
        width < MIN_CROP_WIDTH
        or height < MIN_CROP_HEIGHT
    ):

        return False

    if (
        width > MAX_CROP_WIDTH
        or height > MAX_CROP_HEIGHT
    ):

        return False

    return True


# ============================================================
# CROP QUALITY
# ============================================================

def estimate_vehicle_crop_quality(
    crop,
) -> float:

    if not _validate_crop(
        crop
    ):

        return 0.0

    try:

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(
            np.mean(
                gray
            )
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        if not (
            np.isfinite(
                brightness
            )
            and np.isfinite(
                blur_score
            )
        ):

            return 0.0

        brightness_score = 1.0

        if (
            brightness
            < MIN_BRIGHTNESS
        ):

            brightness_score = (
                brightness
                / max(
                    MIN_BRIGHTNESS,
                    1.0,
                )
            )

        elif (
            brightness
            > MAX_BRIGHTNESS
        ):

            brightness_score = max(
                0.0,
                (
                    255.0
                    - brightness
                )
                / max(
                    255.0
                    - MAX_BRIGHTNESS,
                    1.0,
                ),
            )

        blur_score_normalized = min(
            1.0,
            max(
                0.0,
                blur_score
                / 300.0,
            ),
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    (
                        0.60
                        * blur_score_normalized
                    )
                    + (
                        0.40
                        * brightness_score
                    ),
                ),
            )
        )

    except Exception:

        return 0.0


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_vehicle(
    crop,
):

    if not _validate_crop(
        crop
    ):

        return None

    try:

        session = (
            get_reid_session()
        )

        input_height, input_width = (
            _resolve_input_dimensions(
                session
            )
        )

        if (
            input_height <= 0
            or input_width <= 0
        ):

            return None

        image = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (
                input_width,
                input_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        image = image.astype(
            np.float32,
            copy=False,
        )

        image /= 255.0

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        image = np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(
                image
            )
        ):

            return None

        return image

    except (
        cv2.error,
        ValueError,
        TypeError,
        RuntimeError,
    ):

        logger.debug(
            "Vehicle Re-ID preprocessing failed",
            exc_info=True,
        )

        return None


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(
    embedding,
):

    if embedding is None:

        return None

    try:

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(
            -1
        )

    except (
        ValueError,
        TypeError,
    ):

        return None

    if embedding.size == 0:

        return None

    if not np.all(
        np.isfinite(
            embedding
        )
    ):

        return None

    norm = float(
        np.linalg.norm(
            embedding
        )
    )

    if (
        not np.isfinite(
            norm
        )
        or norm <= 1e-12
    ):

        return None

    normalized = (
        embedding
        / norm
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.all(
        np.isfinite(
            normalized
        )
    ):

        return None

    return normalized


# ============================================================
# OUTPUT EXTRACTION
# ============================================================

def _extract_embedding_output(
    outputs,
):

    if not outputs:

        return None

    # Most OSNet ONNX exports place the embedding
    # in output[0].
    #
    # If an exporter returns [1, 512], flattening
    # produces the required 512-dimensional vector.

    try:

        candidate = outputs[0]

        array = np.asarray(
            candidate,
            dtype=np.float32,
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if array.size == 0:

        return None

    return normalize_embedding(
        array
    )


# ============================================================
# GET EMBEDDING
# ============================================================

def get_vehicle_embedding(
    crop,
):

    input_tensor = (
        preprocess_vehicle(
            crop
        )
    )

    if input_tensor is None:

        return None

    try:

        session = (
            get_reid_session()
        )

        inputs = (
            session.get_inputs()
        )

        if not inputs:

            return None

        input_name = (
            inputs[0].name
        )

        if not input_name:

            return None

        with _inference_lock:

            outputs = session.run(
                None,
                {
                    input_name:
                        input_tensor
                },
            )

    except Exception:

        logger.exception(
            "Vehicle Re-ID inference failed"
        )

        return None

    embedding = (
        _extract_embedding_output(
            outputs
        )
    )

    if embedding is None:

        return None

    if len(
        embedding
    ) != EMBEDDING_DIM:

        logger.warning(
            "Unexpected Re-ID embedding "
            "dimension=%s expected=%s",
            len(
                embedding
            ),
            EMBEDDING_DIM,
        )

        return None

    return embedding.tolist()



def get_vehicle_embedding_result(
    crop,
) -> dict[str, Any]:
    """
    Production-safe Re-ID inference result.

    Keeps the legacy get_vehicle_embedding() contract intact while
    exposing observation quality separately from identity similarity.
    """
    quality = estimate_vehicle_crop_quality(crop)

    if quality <= 0.0:
        return {
            "embedding": None,
            "embedding_quality": 0.0,
            "crop_quality": 0.0,
            "valid": False,
        }

    embedding = get_vehicle_embedding(crop)

    if embedding is None:
        return {
            "embedding": None,
            "embedding_quality": 0.0,
            "crop_quality": float(quality),
            "valid": False,
        }

    return {
        "embedding": embedding,
        "embedding_quality": 1.0,
        "crop_quality": float(quality),
        "valid": True,
        "dimension": len(embedding),
    }


# ============================================================
# EMBEDDING QUALITY
# ============================================================

def embedding_quality(
    embedding,
) -> float:

    normalized = (
        normalize_embedding(
            embedding
        )
    )

    if normalized is None:

        return 0.0

    if len(
        normalized
    ) != EMBEDDING_DIM:

        return 0.0

    # A valid normalized embedding is structurally sound.
    # This score intentionally does not claim identity confidence.
    return 1.0


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding_a,
    embedding_b,
):

    if (
        embedding_a is None
        or embedding_b is None
    ):

        return 0.0

    a = normalize_embedding(
        embedding_a
    )

    b = normalize_embedding(
        embedding_b
    )

    if (
        a is None
        or b is None
    ):

        return 0.0

    if len(a) != len(b):

        return 0.0

    if (
        len(a)
        != EMBEDDING_DIM
    ):

        return 0.0

    try:

        similarity = float(
            np.dot(
                a,
                b,
            )
        )

    except (
        ValueError,
        TypeError,
    ):

        return 0.0

    if not np.isfinite(
        similarity
    ):

        return 0.0

    return float(
        max(
            -1.0,
            min(
                1.0,
                similarity,
            ),
        )
    )


# ============================================================
# RE-ID SCORE NORMALIZATION
# ============================================================

def similarity_to_score(
    similarity: float,
) -> float:

    """
    Convert cosine similarity [-1, 1] to [0, 1].

    This is a mathematical score transformation, NOT a
    probability calibration.
    """

    try:

        similarity = float(
            similarity
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    if not np.isfinite(
        similarity
    ):

        return 0.0

    similarity = max(
        -1.0,
        min(
            1.0,
            similarity,
        ),
    )

    return float(
        (
            similarity
            + 1.0
        )
        / 2.0
    )


# ============================================================
# RE-ID DECISION
# ============================================================

def classify_reid_similarity(
    similarity: float,
) -> str:

    try:

        similarity = float(
            similarity
        )

    except (
        TypeError,
        ValueError,
    ):

        return "INVALID"

    if not np.isfinite(
        similarity
    ):

        return "INVALID"

    if (
        similarity
        >= REID_STRONG_THRESHOLD
    ):

        return "STRONG"

    if (
        similarity
        >= REID_PROBABLE_THRESHOLD
    ):

        return "PROBABLE"

    if (
        similarity
        >= REID_POSSIBLE_THRESHOLD
    ):

        return "POSSIBLE"

    return "WEAK"


# ============================================================
# MATCH VEHICLES
# ============================================================

def is_same_vehicle(
    embedding_a,
    embedding_b,
    threshold=DEFAULT_REID_THRESHOLD,
):

    try:

        threshold = float(
            threshold
        )

    except (
        TypeError,
        ValueError,
    ):

        threshold = (
            DEFAULT_REID_THRESHOLD
        )

    threshold = max(
        -1.0,
        min(
            1.0,
            threshold,
        ),
    )

    similarity = (
        cosine_similarity(
            embedding_a,
            embedding_b,
        )
    )

    return (
        similarity >= threshold,
        similarity,
    )


# ============================================================
# DETAILED RE-ID MATCH
# ============================================================

def compare_vehicle_embeddings(
    embedding_a,
    embedding_b,
    threshold=DEFAULT_REID_THRESHOLD,
) -> dict[str, Any]:

    similarity = (
        cosine_similarity(
            embedding_a,
            embedding_b,
        )
    )

    valid = (
        normalize_embedding(
            embedding_a
        )
        is not None
        and
        normalize_embedding(
            embedding_b
        )
        is not None
    )

    same_vehicle = (
        valid
        and similarity >= float(
            threshold
        )
    )

    return {

        "valid":
            bool(
                valid
            ),

        "similarity":
            float(
                similarity
            ),

        "score":
            similarity_to_score(
                similarity
            ),

        "threshold":
            float(
                threshold
            ),

        "same_vehicle":
            bool(
                same_vehicle
            ),

        "classification":
            classify_reid_similarity(
                similarity
            ),
    }


# ============================================================
# BATCH EMBEDDING
# ============================================================

def get_vehicle_embeddings_batch(
    crops,
) -> list[
    Optional[list[float]]
]:

    if crops is None:

        return []

    try:

        crops = list(
            crops
        )

    except TypeError:

        return []

    results = []

    for crop in crops:

        results.append(
            get_vehicle_embedding(
                crop
            )
        )

    return results


# ============================================================
# MODEL HEALTH
# ============================================================

def reid_health() -> dict[str, Any]:

    result = {

        "component":
            "vehicleReid",

        "model_path":
            MODEL_PATH,

        "model_exists":
            Path(
                MODEL_PATH
            ).expanduser().exists(),

        "loaded":
            _session is not None,

        "input_width":
            INPUT_WIDTH,

        "input_height":
            INPUT_HEIGHT,

        "embedding_dimension":
            EMBEDDING_DIM,

        "expected_embedding_dimension":
            EXPECTED_EMBEDDING_DIM,

        "threshold":
            DEFAULT_REID_THRESHOLD,

        "possible_threshold":
            REID_POSSIBLE_THRESHOLD,

        "probable_threshold":
            REID_PROBABLE_THRESHOLD,

        "strong_threshold":
            REID_STRONG_THRESHOLD,

    }

    if _session is not None:

        try:

            result[
                "providers"
            ] = list(
                _session.get_providers()
            )

            result[
                "model_metadata"
            ] = (
                get_reid_model_metadata()
            )

        except Exception:

            result[
                "providers"
            ] = []

    return result


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_reid_configuration() -> dict[str, Any]:

    errors = []

    if INPUT_WIDTH <= 0:

        errors.append(
            "VEHICLE_REID_INPUT_WIDTH "
            "must be positive"
        )

    if INPUT_HEIGHT <= 0:

        errors.append(
            "VEHICLE_REID_INPUT_HEIGHT "
            "must be positive"
        )

    if EMBEDDING_DIM <= 0:

        errors.append(
            "VEHICLE_REID_EMBEDDING_DIM "
            "must be positive"
        )

    if (
        MIN_CROP_WIDTH <= 0
        or MIN_CROP_HEIGHT <= 0
    ):

        errors.append(
            "Minimum crop dimensions "
            "must be positive"
        )

    if (
        MAX_CROP_WIDTH
        < MIN_CROP_WIDTH
        or MAX_CROP_HEIGHT
        < MIN_CROP_HEIGHT
    ):

        errors.append(
            "Maximum crop dimensions "
            "must exceed minimum dimensions"
        )

    thresholds = {
        "default":
            DEFAULT_REID_THRESHOLD,

        "possible":
            REID_POSSIBLE_THRESHOLD,

        "probable":
            REID_PROBABLE_THRESHOLD,

        "strong":
            REID_STRONG_THRESHOLD,
    }

    for name, value in thresholds.items():

        if (
            not np.isfinite(
                value
            )
            or value < -1.0
            or value > 1.0
        ):

            errors.append(
                f"Re-ID {name} threshold "
                "must be within [-1, 1]"
            )

    if (
        REID_POSSIBLE_THRESHOLD
        > REID_PROBABLE_THRESHOLD
    ):

        errors.append(
            "Possible threshold cannot "
            "exceed probable threshold"
        )

    if (
        REID_PROBABLE_THRESHOLD
        > REID_STRONG_THRESHOLD
    ):

        errors.append(
            "Probable threshold cannot "
            "exceed strong threshold"
        )

    if not MODEL_PATH:

        errors.append(
            "VEHICLE_REID_MODEL is empty"
        )

    return {

        "valid":
            len(
                errors
            ) == 0,

        "errors":
            errors,

        "model_path":
            MODEL_PATH,

        "input_width":
            INPUT_WIDTH,

        "input_height":
            INPUT_HEIGHT,

        "embedding_dimension":
            EMBEDDING_DIM,

        "thresholds":
            thresholds,
    }


# ============================================================
# SESSION RESET
# ============================================================

def reset_reid_session() -> None:

    global _session
    global _model_metadata

    with _session_lock:

        _session = None

        with _model_metadata_lock:

            _model_metadata = None


# ============================================================
# STARTUP SELF TEST
# ============================================================

def warmup_reid() -> bool:
    """
    Load the model and execute one deterministic inference.

    Useful during worker startup so model failures happen before
    the camera pipeline starts processing live frames.
    """

    try:

        session = (
            get_reid_session()
        )

        height, width = (
            _resolve_input_dimensions(
                session
            )
        )

        dummy = np.zeros(
            (
                1,
                3,
                height,
                width,
            ),
            dtype=np.float32,
        )

        input_name = (
            session
            .get_inputs()[0]
            .name
        )

        with _inference_lock:

            outputs = session.run(
                None,
                {
                    input_name:
                        dummy
                },
            )

        embedding = (
            _extract_embedding_output(
                outputs
            )
        )

        if embedding is None:

            logger.error(
                "Vehicle Re-ID warmup "
                "returned invalid embedding"
            )

            return False

        if len(
            embedding
        ) != EMBEDDING_DIM:

            logger.error(
                "Vehicle Re-ID warmup "
                "dimension mismatch: %s != %s",
                len(
                    embedding
                ),
                EMBEDDING_DIM,
            )

            return False

        logger.info(
            "Vehicle Re-ID warmup successful"
        )

        return True

    except Exception:

        logger.exception(
            "Vehicle Re-ID warmup failed"
        )

        return False


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def get_embedding(
    crop,
):

    return get_vehicle_embedding(
        crop
    )


def vehicle_embedding(
    crop,
):

    return get_vehicle_embedding(
        crop
    )


def compare_embeddings(
    embedding_a,
    embedding_b,
):

    return cosine_similarity(
        embedding_a,
        embedding_b,
    )


# ============================================================
# MODULE SELF CHECK
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    configuration = (
        validate_reid_configuration()
    )

    logger.info(
        "INTEL-I Vehicle Re-ID "
        "configuration: %s",
        configuration,
    )