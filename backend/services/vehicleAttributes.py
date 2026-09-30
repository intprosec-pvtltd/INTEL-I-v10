from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import cv2
import numpy as np
from dotenv import load_dotenv


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("vehicle-attributes")


# ============================================================
# BASE DIRECTORY / ENVIRONMENT
# ============================================================

# services/vehicleAttributes.py
#        |
#        └── backend/
BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"

# IMPORTANT:
# Load .env before reading os.getenv().
load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH_RAW = os.getenv(
    "VEHICLE_ATTRIBUTE_MODEL_PATH",
    "",
).strip()


MODEL_VERSION = (
    os.getenv(
        "VEHICLE_ATTRIBUTE_MODEL_VERSION",
        "1.0",
    ).strip()
    or "1.0"
)


def _get_float_env(
    name: str,
    default: float,
) -> float:

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)

    except (TypeError, ValueError):

        logger.warning(
            "Invalid float environment variable %s=%r. "
            "Using default=%s",
            name,
            value,
            default,
        )

        return default


ATTRIBUTE_CONFIDENCE_THRESHOLD = _get_float_env(
    "VEHICLE_ATTRIBUTE_CONFIDENCE_THRESHOLD",
    0.50,
)


VEHICLE_ATTRIBUTE_DEVICE = os.getenv(
    "VEHICLE_ATTRIBUTE_DEVICE",
    "cpu",
).strip()


# ============================================================
# MODEL PATH
# ============================================================

def resolve_model_path(
    value: str,
) -> Optional[Path]:
    """
    Resolve a model path.

    Relative paths are resolved against the INTEL-I backend root.

    Example:

        ./models/model

    becomes:

        backend/models/model
    """

    if not value:
        return None

    path = Path(value)

    if not path.is_absolute():

        path = BASE_DIR / path

    try:

        return path.resolve()

    except Exception:

        return path.absolute()


MODEL_PATH = resolve_model_path(
    MODEL_PATH_RAW
)


# ============================================================
# MODEL STATE
# ============================================================

_model = None

_model_lock = Lock()

_model_load_attempted = False

_model_load_error: Optional[str] = None


# ============================================================
# MODEL NAME
# ============================================================

MODEL_NAME = (
    "PP-LCNet_x1_0_vehicle_attribute"
)


# ============================================================
# MODEL LABELS
# ============================================================
#
# IMPORTANT:
#
# PP-LCNet output label_names should be preferred over
# hard-coded numeric mappings.
#
# These mappings are ONLY fallback mappings.
#
# If inference.yml provides the actual labels, the
# returned label_names are used first.
#


COLOR_LABELS = {
    0: "yellow",
    1: "orange",
    2: "green",
    3: "gray",
    4: "red",
    5: "blue",
    6: "white",
    7: "golden",
    8: "brown",
    9: "black",
}


VEHICLE_TYPE_LABELS = {
    10: "sedan",
    11: "suv",
    12: "van",
    13: "hatchback",
    14: "mpv",
    15: "pickup",
    16: "bus",
    17: "truck",
    18: "estate",
}


# ============================================================
# GENERIC LABEL NORMALIZATION
# ============================================================

def clean_label(
    label: Any,
) -> str:
    """
    Normalize PaddleX labels.

    Examples:

        yellow(黄色)
        ->
        yellow

        hatchback(掀背车)
        ->
        hatchback
    """

    if label is None:

        return ""

    value = str(label).strip()

    if not value:

        return ""

    # Remove parenthesized translations.
    if "(" in value:

        value = value.split(
            "(",
            1,
        )[0].strip()

    if "（" in value:

        value = value.split(
            "（",
            1,
        )[0].strip()

    return value.lower().strip()


# ============================================================
# LABEL CLASSIFICATION
# ============================================================

COLOR_NAMES = {
    "yellow",
    "orange",
    "green",
    "gray",
    "grey",
    "red",
    "blue",
    "white",
    "gold",
    "golden",
    "brown",
    "black",
    "silver",
    "purple",
}


VEHICLE_TYPE_NAMES = {
    "car",
    "sedan",
    "suv",
    "van",
    "hatchback",
    "mpv",
    "pickup",
    "bus",
    "truck",
    "estate",
    "wagon",
    "minivan",
    "motorcycle",
    "motorbike",
}


def _is_color_label(
    label: str,
) -> bool:

    return clean_label(label) in COLOR_NAMES


def _is_vehicle_type_label(
    label: str,
) -> bool:

    return clean_label(label) in VEHICLE_TYPE_NAMES


# ============================================================
# DEVICE
# ============================================================

def _resolve_device() -> str:
    """
    Convert VEHICLE_ATTRIBUTE_DEVICE to PaddleX format.

    Examples:

        VEHICLE_ATTRIBUTE_DEVICE=0
        -> gpu:0

        VEHICLE_ATTRIBUTE_DEVICE=1
        -> gpu:1

        VEHICLE_ATTRIBUTE_DEVICE=gpu:0
        -> gpu:0

        VEHICLE_ATTRIBUTE_DEVICE=cpu
        -> cpu
    """

    value = (
        VEHICLE_ATTRIBUTE_DEVICE
        .lower()
        .strip()
    )

    if not value:

        return "cpu"

    if value == "cpu":

        return "cpu"

    if value.startswith(
        "gpu:"
    ):

        return value

    if value.isdigit():

        return f"gpu:{value}"

    return "cpu"


# ============================================================
# MODEL DIRECTORY VALIDATION
# ============================================================

def _validate_model_directory(
    path: Optional[Path],
) -> bool:
    """
    Validate PP-LCNet Paddle inference directory.
    """

    if path is None:

        logger.error(
            "Vehicle attribute model path is not configured."
        )

        return False

    if not path.exists():

        logger.error(
            "Vehicle attribute model directory does not exist: %s",
            path,
        )

        return False

    if not path.is_dir():

        logger.error(
            "Vehicle attribute model path is not a directory: %s",
            path,
        )

        return False

    required_files = [
        "inference.json",
        "inference.pdiparams",
        "inference.yml",
    ]

    missing_files = []

    for filename in required_files:

        file_path = path / filename

        if not file_path.is_file():

            missing_files.append(
                filename
            )

    if missing_files:

        logger.error(
            "Invalid PP-LCNet model directory: %s | missing=%s",
            path,
            missing_files,
        )

        return False

    return True


# ============================================================
# MODEL LOADING
# ============================================================

def _load_model():
    """
    Lazy-load PaddleX PP-LCNet.

    The model is loaded only once and is protected by a lock
    because INTEL-I can have multiple camera workers.
    """

    global _model
    global _model_load_attempted
    global _model_load_error

    # Already loaded.
    if _model is not None:

        return _model

    # Previous attempt failed.
    if _model_load_attempted:

        return None

    with _model_lock:

        if _model is not None:

            return _model

        if _model_load_attempted:

            return None

        _model_load_attempted = True

        # ----------------------------------------------------
        # Validate model path
        # ----------------------------------------------------

        if not _validate_model_directory(
            MODEL_PATH
        ):

            _model_load_error = (
                "Invalid or missing model directory"
            )

            return None

        # ----------------------------------------------------
        # Import PaddleX
        # ----------------------------------------------------

        try:

            from paddlex import create_model

        except Exception as exc:

            _model_load_error = str(
                exc
            )

            logger.exception(
                "PaddleX is not installed. "
                "Install PaddlePaddle and PaddleX."
            )

            return None

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        try:

            device = _resolve_device()

            logger.info(
                "Loading vehicle attribute model | "
                "model=%s | path=%s | device=%s | version=%s",
                MODEL_NAME,
                MODEL_PATH,
                device,
                MODEL_VERSION,
            )

            _model = create_model(
                model_name=MODEL_NAME,
                model_dir=str(
                    MODEL_PATH
                ),
                device=device,
            )

            logger.info(
                "Vehicle attribute model loaded successfully | "
                "model=%s | device=%s",
                MODEL_NAME,
                device,
            )

            return _model

        except Exception as exc:

            _model_load_error = str(
                exc
            )

            logger.exception(
                "Failed to load vehicle attribute model | "
                "path=%s",
                MODEL_PATH,
            )

            _model = None

            return None


# ============================================================
# IMAGE PREPARATION
# ============================================================

def _prepare_crop(
    crop: Any,
) -> Optional[np.ndarray]:
    """
    Normalize vehicle image crop into BGR uint8 ndarray.
    """

    if crop is None:

        return None

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    if not isinstance(
        crop,
        np.ndarray,
    ):

        try:

            crop = np.asarray(
                crop
            )

        except Exception:

            logger.warning(
                "Unable to convert vehicle crop to numpy."
            )

            return None

    if crop.size == 0:

        return None

    # --------------------------------------------------------
    # Handle grayscale
    # --------------------------------------------------------

    if crop.ndim == 2:

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_GRAY2BGR,
        )

    # --------------------------------------------------------
    # Handle BGRA/RGBA
    # --------------------------------------------------------

    elif (
        crop.ndim == 3
        and crop.shape[2] == 4
    ):

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGRA2BGR,
        )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if (
        crop.ndim != 3
        or crop.shape[2] != 3
    ):

        logger.warning(
            "Invalid vehicle crop shape: %s",
            crop.shape,
        )

        return None

    height, width = crop.shape[:2]

    if (
        width < 20
        or height < 20
    ):

        logger.debug(
            "Vehicle crop too small: %sx%s",
            width,
            height,
        )

        return None

    # --------------------------------------------------------
    # Convert dtype
    # --------------------------------------------------------

    if crop.dtype != np.uint8:

        try:

            crop = np.clip(
                crop,
                0,
                255,
            ).astype(
                np.uint8
            )

        except Exception:

            logger.exception(
                "Failed to convert vehicle crop to uint8."
            )

            return None

    return np.ascontiguousarray(
        crop
    )


# ============================================================
# FALLBACK COLOR
# ============================================================

def _fallback_color(
    crop: Optional[np.ndarray],
) -> Optional[str]:
    """
    Conservative OpenCV color fallback.

    This is only used when PP-LCNet is unavailable or doesn't
    return a valid color.

    Fallback output is marked separately from model output.
    """

    if (
        crop is None
        or crop.size == 0
    ):

        return None

    try:

        image = cv2.resize(
            crop,
            (48, 48),
            interpolation=cv2.INTER_AREA,
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        pixels = hsv.reshape(
            -1,
            3,
        )

        saturation = pixels[:, 1]
        value = pixels[:, 2]

        mean_value = float(
            np.mean(value)
        )

        mean_saturation = float(
            np.mean(saturation)
        )

        # Black.
        if mean_value < 55:

            return "black"

        # White.
        if (
            mean_value > 190
            and mean_saturation < 50
        ):

            return "white"

        valid = pixels[
            (saturation > 45)
            & (value > 45)
        ]

        if len(valid) < 30:

            return "gray"

        hue = float(
            np.median(
                valid[:, 0]
            )
        )

        if (
            hue < 10
            or hue >= 170
        ):

            return "red"

        if hue < 25:

            return "orange"

        if hue < 35:

            return "yellow"

        if hue < 85:

            return "green"

        if hue < 125:

            return "blue"

        return "purple"

    except Exception:

        logger.exception(
            "Fallback vehicle color detection failed."
        )

        return None


# ============================================================
# RESULT OBJECT EXTRACTION
# ============================================================

def _extract_result_object(
    result: Any,
) -> Any:
    """
    Normalize PaddleX result wrapper.
    """

    if result is None:

        return None

    if isinstance(
        result,
        dict,
    ):

        if "res" in result:

            return result["res"]

        return result

    return result


# ============================================================
# RESULT FIELD READER
# ============================================================

def _get_result_field(
    result: Any,
    field: str,
    default: Any = None,
) -> Any:
    """
    Read PaddleX result fields from either dictionaries or
    result objects.
    """

    if result is None:

        return default

    if isinstance(
        result,
        dict,
    ):

        return result.get(
            field,
            default,
        )

    try:

        return getattr(
            result,
            field,
            default,
        )

    except Exception:

        return default


# ============================================================
# SAFE LIST CONVERSION
# ============================================================

def _safe_list(
    value: Any,
) -> list:
    """
    Convert PaddleX output into a normal Python list.
    """

    if value is None:

        return []

    try:

        return list(value)

    except Exception:

        return []


# ============================================================
# ATTRIBUTE PARSING
# ============================================================

def _parse_predictions(
    result: Any,
) -> Dict[str, Any]:
    """
    Parse PP-LCNet output.

    Preferred PaddleX fields:

        class_ids
        scores
        label_names

    Label names are preferred over hard-coded numeric IDs.
    """

    result = _extract_result_object(
        result
    )

    class_ids = _safe_list(
        _get_result_field(
            result,
            "class_ids",
            [],
        )
    )

    scores = _safe_list(
        _get_result_field(
            result,
            "scores",
            [],
        )
    )

    label_names = _safe_list(
        _get_result_field(
            result,
            "label_names",
            [],
        )
    )

    output: Dict[str, Any] = {

        "color": None,

        # Preserve the detector class separately; this is PP-LCNet body style.
        "vehicle_type": None,
        "vehicle_subtype": None,

        "attribute_confidence": 0.0,

        "attribute_predictions": [],
    }

    # --------------------------------------------------------
    # Determine prediction count
    # --------------------------------------------------------

    count = max(
        len(class_ids),
        len(scores),
        len(label_names),
    )

    confidences = []

    for index in range(
        count
    ):

        # ----------------------------------------------------
        # Class ID
        # ----------------------------------------------------

        class_id = None

        if index < len(
            class_ids
        ):

            try:

                class_id = int(
                    class_ids[index]
                )

            except Exception:

                class_id = None

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        score = 0.0

        if index < len(
            scores
        ):

            try:

                score = float(
                    scores[index]
                )

            except Exception:

                score = 0.0

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        raw_label = ""

        if index < len(
            label_names
        ):

            raw_label = (
                label_names[index]
            )

        label = clean_label(
            raw_label
        )

        if (
            not label
            and class_id is not None
        ):

            if class_id in COLOR_LABELS:

                label = COLOR_LABELS[
                    class_id
                ]

            elif class_id in VEHICLE_TYPE_LABELS:

                label = VEHICLE_TYPE_LABELS[
                    class_id
                ]

        if not label:

            continue

        # ----------------------------------------------------
        # Store raw prediction
        # ----------------------------------------------------

        prediction = {

            "class_id": class_id,

            "label": label,

            "confidence": score,
        }

        output[
            "attribute_predictions"
        ].append(
            prediction
        )

        # ----------------------------------------------------
        # Confidence collection
        # ----------------------------------------------------

        if (
            score
            >= ATTRIBUTE_CONFIDENCE_THRESHOLD
        ):

            confidences.append(
                score
            )

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        if _is_color_label(
            label
        ):

            # Normalize grey -> gray.
            if label == "grey":

                label = "gray"

            if label == "gold":

                label = "golden"

            output[
                "color"
            ] = label

            continue

        # ----------------------------------------------------
        # VEHICLE TYPE
        # ----------------------------------------------------

        if _is_vehicle_type_label(
            label
        ):

            output[
                "vehicle_type"
            ] = label

            output[
                "vehicle_subtype"
            ] = label

            continue

    # --------------------------------------------------------
    # Overall attribute confidence
    # --------------------------------------------------------

    if confidences:

        output[
            "attribute_confidence"
        ] = max(
            0.0,
            min(
                1.0,
                float(
                    np.mean(
                        confidences
                    )
                ),
            ),
        )

    return output


# ============================================================
# MODEL INFERENCE
# ============================================================

def _run_model(
    model: Any,
    crop: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Run PP-LCNet inference.

    PaddleX returns an iterable of result objects.
    """

    try:

        outputs = model.predict(
            crop,
            batch_size=1,
        )

        if outputs is None:

            return None

        # ----------------------------------------------------
        # Iterable result
        # ----------------------------------------------------

        if isinstance(
            outputs,
            (list, tuple),
        ):

            if not outputs:

                return None

            result = outputs[0]

        else:

            # PaddleX may return a generator.
            try:

                result = next(
                    iter(outputs)
                )

            except TypeError:

                result = outputs

            except StopIteration:

                return None

        return _parse_predictions(
            result
        )

    except Exception:

        logger.exception(
            "PP-LCNet vehicle attribute inference failed."
        )

        return None


# ============================================================
# PUBLIC INFERENCE API
# ============================================================

def infer_vehicle_attributes(
    crop: Any,
    vehicle_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main INTEL-I vehicle attribute inference API.

    Returns:

        {
            "vehicle_type": "...",
            "vehicle_subtype": "...",
            "vehicle_detection_type": "...",
            "color": "...",
            "make": None,
            "model": None,
            "attribute_confidence": 0.0,
            "attribute_model": "...",
            "attribute_model_version": "...",
            "attributes_source": "...",
            "attribute_predictions": [...]
        }

    PP-LCNet:
        - vehicle color
        - vehicle type

    Not provided reliably:
        - manufacturer
        - vehicle model
    """

    normalized_crop = _prepare_crop(
        crop
    )

    # --------------------------------------------------------
    # Default result
    # --------------------------------------------------------

    fallback_color = _fallback_color(
        normalized_crop
    )

    detector_vehicle_type = str(
        vehicle_type or "vehicle"
    )[:50]

    result: Dict[str, Any] = {

        # Keep YOLO's coarse class (car/truck/motorcycle) as vehicle_type.
        "vehicle_type": detector_vehicle_type,

        # PP-LCNet's body-style classification.
        "vehicle_subtype": None,

        "vehicle_detection_type": detector_vehicle_type,

        "color": fallback_color,

        "make": None,

        "model": None,

        "attribute_confidence": 0.0,

        "attribute_model": "fallback_color",

        "attribute_model_version": MODEL_VERSION,

        "attributes_source": "fallback",

        "attribute_predictions": [],
    }

    # --------------------------------------------------------
    # Invalid crop
    # --------------------------------------------------------

    if normalized_crop is None:

        return result

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = _load_model()

    if model is None:

        return result

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    prediction = _run_model(
        model,
        normalized_crop,
    )

    if not prediction:

        return result

    # --------------------------------------------------------
    # Vehicle type
    # --------------------------------------------------------

    predicted_type = prediction.get(
        "vehicle_type"
    )

    if predicted_type:

        result[
            "vehicle_subtype"
        ] = str(
            predicted_type
        )[:50]

    # --------------------------------------------------------
    # Color
    # --------------------------------------------------------

    predicted_color = prediction.get(
        "color"
    )

    if predicted_color:

        result[
            "color"
        ] = str(
            predicted_color
        )[:50]

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        prediction.get(
            "attribute_confidence",
            0.0,
        )
        or 0.0
    )

    result[
        "attribute_confidence"
    ] = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    result[
        "attribute_model"
    ] = MODEL_NAME

    result[
        "attribute_model_version"
    ] = MODEL_VERSION

    result[
        "attributes_source"
    ] = "model"

    result[
        "attribute_predictions"
    ] = prediction.get(
        "attribute_predictions",
        [],
    )

    return result


# ============================================================
# MODEL STATUS
# ============================================================

def vehicle_attribute_model_status() -> Dict[str, Any]:
    """
    Return safe model health/status information.
    """

    return {

        "enabled": bool(
            MODEL_PATH
        ),

        "model_loaded": (
            _model is not None
        ),

        "model_path_configured": bool(
            MODEL_PATH
        ),

        "model_path_exists": bool(
            MODEL_PATH
            and MODEL_PATH.exists()
        ),

        "model_path_is_directory": bool(
            MODEL_PATH
            and MODEL_PATH.is_dir()
        ),

        "required_files_present": bool(
            MODEL_PATH
            and _validate_model_directory(
                MODEL_PATH
            )
        ),

        "model_version": MODEL_VERSION,

        "device": _resolve_device(),

        "model_name": MODEL_NAME,

        "load_attempted": (
            _model_load_attempted
        ),

        "load_error": _model_load_error,
    }


# ============================================================
# PRELOAD MODEL
# ============================================================

def preload_vehicle_attribute_model() -> bool:
    """
    Explicitly load the vehicle attribute model.

    Returns:
        True  -> successfully loaded
        False -> failed/unavailable
    """

    return (
        _load_model()
        is not None
    )


def reset_vehicle_attribute_model() -> None:
    global _model
    global _model_load_attempted
    global _model_load_error

    with _model_lock:
        _model = None
        _model_load_attempted = False
        _model_load_error = None

    logger.info(
        "Vehicle attribute model state reset."
    )
