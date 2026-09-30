from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ai.tensorrt.runtime import runtime as tensorrt_runtime
from services.anprVoting import OCRCandidate, character_vote
from services.ollamaVision import ollama_plate_verifier
from services.plateFusion import fuse_plate_candidates
from services.anprAdaptivePreprocess import build_adaptive_variants
from services.awirosOcr import AwirosOCR
from services.anprOcrClient import RemoteAwirosOCR
try:
    from inference.model_metrics import MODEL_METRICS
except Exception:  # standalone ANPR compatibility
    MODEL_METRICS = None


# ============================================================
# ENVIRONMENT / LOGGING
# ============================================================

os.environ.setdefault(
    "PADDLE_DISABLE_ONEDNN",
    "1",
)

os.environ.setdefault(
    "GLOG_minloglevel",
    "2",
)

os.environ.setdefault(
    "FLAGS_log_level",
    "3",
)



try:
    import onnxruntime as ort

    _OCR_RUNTIME_IMPORT_OK = True
    _OCR_RUNTIME_VERSION = str(getattr(ort, "__version__", "unknown"))
except Exception as exc:
    ort = None
    _OCR_RUNTIME_IMPORT_OK = False
    _OCR_RUNTIME_VERSION = "unavailable"
    print(
        "[INTEL-I][ANPR][ERROR] ONNX Runtime initialization failed: "
        f"{exc}",
        flush=True,
    )


logger = logging.getLogger(__name__)

# ============================================================
# ANPR CONSOLE AUDIT LOGGING
# ============================================================

ANPR_CONSOLE_LOGS = (
    os.getenv(
        "ANPR_CONSOLE_LOGS",
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


def _anpr_log(message: str) -> None:
    if not ANPR_CONSOLE_LOGS:
        return

    try:
        print(
            f"[INTEL-I][ANPR] {message}",
            flush=True,
        )
    except Exception:
        pass


def _anpr_warning(message: str) -> None:
    if not ANPR_CONSOLE_LOGS:
        return

    try:
        print(
            f"[INTEL-I][ANPR][WARNING] {message}",
            flush=True,
        )
    except Exception:
        pass


def _anpr_error(message: str) -> None:
    if not ANPR_CONSOLE_LOGS:
        return

    try:
        print(
            f"[INTEL-I][ANPR][ERROR] {message}",
            flush=True,
        )
    except Exception:
        pass


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 100000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_csv_set(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip().lower() for item in str(raw).split(",") if item.strip()}


def _env_quad(name: str, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except (TypeError, ValueError):
        return default
    if len(values) != 4 or any((not math.isfinite(v) or v < 0.0 or v > 1.5) for v in values):
        return default
    return values  # left, top, right, bottom


DEFAULT_PLATE_MODEL = (
    "models/license_plate_yolov8m.pt"
)

PLATE_MODEL_PATH = Path(
    os.getenv(
        "ANPR_PLATE_MODEL",
        DEFAULT_PLATE_MODEL,
    )
).resolve()

ANPR_PLATE_DEVICE = os.getenv(
    "ANPR_PLATE_DEVICE",
    "auto",
).strip().lower()

PLATE_DETECTION_CONF = _env_float("ANPR_PLATE_DETECTION_CONF", 0.15)

PLATE_IOU = _env_float("ANPR_PLATE_IOU", 0.45)
ANPR_DETECT_IMAGE_SIZE = _env_int("ANPR_DETECT_IMAGE_SIZE", 1280, 320, 2048)
ANPR_DETECT_FALLBACK_IMAGE_SIZE = _env_int("ANPR_DETECT_FALLBACK_IMAGE_SIZE", 960, 320, 2048)

# Mixed-vehicle support. Existing callers may omit vehicle_class; when supplied,
# aliases are normalized and unsupported classes are skipped safely.
ANPR_SUPPORTED_VEHICLES = _env_csv_set(
    "ANPR_VEHICLE_CLASSES",
    "car,motorcycle,motorbike,bike,scooter,bus,truck,van,autorickshaw,auto-rickshaw",
)
VEHICLE_CLASS_ALIASES = {
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "scooter": "motorcycle",
    "auto-rickshaw": "autorickshaw",
    "auto_rickshaw": "autorickshaw",
    "threewheeler": "autorickshaw",
    "three-wheeler": "autorickshaw",
}

ANPR_TWO_LINE_PLATES = _env_bool("ANPR_TWO_LINE_PLATES", True)
ANPR_PERSPECTIVE_CORRECTION = _env_bool("ANPR_PERSPECTIVE_CORRECTION", True)
ANPR_PARTIAL_RESULTS = _env_bool("ANPR_PARTIAL_RESULTS", True)
ANPR_CHARACTER_VOTING_ENABLED = _env_bool("ANPR_CHARACTER_VOTING_ENABLED", True)
ANPR_VEHICLE_CROP_EXPANSION_ENABLED = _env_bool("ANPR_VEHICLE_CROP_EXPANSION_ENABLED", True)
ANPR_TWO_LINE_MAX_ASPECT = _env_float("ANPR_TWO_LINE_MAX_ASPECT", 2.4, 0.8, 4.0)
ANPR_TWO_LINE_SPLIT_OVERLAP = _env_float("ANPR_TWO_LINE_SPLIT_OVERLAP", 0.12, 0.0, 0.35)
ANPR_PARTIAL_CHAR_MIN_AGREEMENT = _env_float("ANPR_PARTIAL_CHAR_MIN_AGREEMENT", 0.60, 0.5, 1.0)
ANPR_PARTIAL_MIN_KNOWN_RATIO = _env_float("ANPR_PARTIAL_MIN_KNOWN_RATIO", 0.65, 0.3, 1.0)
ANPR_CONFIRMED_MIN_TEMPORAL = _env_float("ANPR_CONFIRMED_MIN_TEMPORAL", 0.75, 0.0, 1.0)
ANPR_CONFIRMED_MIN_FORMAT_SCORE = _env_float("ANPR_CONFIRMED_MIN_FORMAT_SCORE", 0.90, 0.0, 1.0)

VEHICLE_CROP_EXPANSION = {
    "car": _env_quad("ANPR_CROP_EXPANSION_CAR", (0.10, 0.10, 0.10, 0.20)),
    "motorcycle": _env_quad("ANPR_CROP_EXPANSION_MOTORCYCLE", (0.25, 0.15, 0.25, 0.40)),
    "bus": _env_quad("ANPR_CROP_EXPANSION_BUS", (0.08, 0.08, 0.08, 0.20)),
    "truck": _env_quad("ANPR_CROP_EXPANSION_TRUCK", (0.12, 0.10, 0.12, 0.25)),
    "van": _env_quad("ANPR_CROP_EXPANSION_VAN", (0.10, 0.10, 0.10, 0.20)),
    "autorickshaw": _env_quad("ANPR_CROP_EXPANSION_AUTORICKSHAW", (0.15, 0.10, 0.15, 0.30)),
}

# SAHI wraps only the configured plate weights; the general YOLO detector is untouched.
ANPR_SAHI_ENABLED = os.getenv("ANPR_SAHI_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ANPR_SAHI_REQUIRED = os.getenv("ANPR_SAHI_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
ANPR_SAHI_SLICE_WIDTH = _env_int("ANPR_SAHI_SLICE_WIDTH", 640, 256, 1280)
ANPR_SAHI_SLICE_HEIGHT = _env_int("ANPR_SAHI_SLICE_HEIGHT", 640, 256, 1280)
ANPR_SAHI_OVERLAP_WIDTH = _env_float("ANPR_SAHI_OVERLAP_WIDTH", 0.25, 0.0, 0.50)
ANPR_SAHI_OVERLAP_HEIGHT = _env_float("ANPR_SAHI_OVERLAP_HEIGHT", 0.25, 0.0, 0.50)
ANPR_SAHI_MIN_IMAGE_WIDTH = _env_int("ANPR_SAHI_MIN_IMAGE_WIDTH", 720, 256, 4096)
ANPR_SAHI_MIN_IMAGE_HEIGHT = _env_int("ANPR_SAHI_MIN_IMAGE_HEIGHT", 480, 256, 4096)
ANPR_SAHI_POSTPROCESS_MATCH_THRESHOLD = _env_float("ANPR_SAHI_POSTPROCESS_MATCH_THRESHOLD", 0.50)
ANPR_SAHI_FRAME_INTERVAL = _env_int("ANPR_SAHI_FRAME_INTERVAL", 2, 1, 30)
ANPR_CROP_BUFFER_SIZE = _env_int("ANPR_CROP_BUFFER_SIZE", 30, 3, 90)
ANPR_OCR_TOP_K_CROPS = _env_int("ANPR_OCR_TOP_K_CROPS", 8, 1, 12)
ANPR_CROP_BUFFER_TTL_SECONDS = _env_float("ANPR_CROP_BUFFER_TTL_SECONDS", 5.0, 0.5, 30.0)

MAX_PLATES_PER_VEHICLE = max(
    1,
    min(
        10,
        int(
            os.getenv(
                "ANPR_MAX_PLATES_PER_VEHICLE",
                "3",
            )
        ),
    ),
)


# ============================================================
# OCR CONFIGURATION
# ============================================================

OCR_ENGINE = os.getenv("ANPR_OCR_ENGINE", "awiros_paddle").strip().lower()
if OCR_ENGINE not in {"awiros_paddle", "ppocrv5_onnx"}:
    raise RuntimeError(
        "ANPR_OCR_ENGINE must be 'awiros_paddle' or 'ppocrv5_onnx'"
    )

OCR_LANGUAGE = os.getenv(
    "ANPR_OCR_LANGUAGE",
    "en",
).strip().lower()

OCR_DEVICE = os.getenv(
    "ANPR_OCR_DEVICE",
    "auto",
).strip().lower()

if OCR_DEVICE not in {"auto", "cuda", "cpu"}:
    OCR_DEVICE = "auto"

OCR_CUDA_DEVICE_ID = max(
    0,
    _env_int("ANPR_OCR_CUDA_DEVICE_ID", 0, 0, 16),
)

OCR_MODEL_NAME = os.getenv(
    "ANPR_OCR_MODEL",
    "Awiros-ANPR-OCR",
).strip()

DEFAULT_OCR_ONNX_MODEL = (
    "models/en_PP-OCRv5_rec_mobile.onnx"
)

OCR_ONNX_MODEL_PATH = Path(
    os.getenv(
        "ANPR_OCR_ONNX_MODEL",
        DEFAULT_OCR_ONNX_MODEL,
    )
).resolve()

DEFAULT_OCR_DICT_PATH = (
    "models/ppocrv5_en_dict.txt"
)

OCR_DICT_PATH = Path(
    os.getenv(
        "ANPR_OCR_DICT",
        DEFAULT_OCR_DICT_PATH,
    )
).resolve()

AWIROS_MODEL_PATH = Path(
    os.getenv(
        "ANPR_AWIROS_MODEL_PATH",
        "models/awiros_anpr_ocr/model.safetensors",
    )
).resolve()
AWIROS_DICT_PATH = Path(
    os.getenv(
        "ANPR_AWIROS_DICT_PATH",
        "models/awiros_anpr_ocr/en_dict.txt",
    )
).resolve()
AWIROS_PADDLEOCR_DIR = Path(
    os.getenv("ANPR_AWIROS_PADDLEOCR_DIR", "vendor/PaddleOCR")
).resolve()
AWIROS_STRICT_DEVICE = os.getenv(
    "ANPR_AWIROS_STRICT_DEVICE", "false"
).strip().lower() in {"1", "true", "yes", "on"}

ANPR_OCR_REMOTE_ENABLED = os.getenv(
    "ANPR_OCR_REMOTE_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
ANPR_OCR_REMOTE_URL = os.getenv(
    "ANPR_OCR_REMOTE_URL", "http://127.0.0.1:9201"
).strip().rstrip("/")
ANPR_OCR_REMOTE_TIMEOUT_SECONDS = _env_float(
    "ANPR_OCR_REMOTE_TIMEOUT_SECONDS", 2.0, 0.1, 30.0
)
ANPR_OCR_INTERNAL_TOKEN = os.getenv(
    "ANPR_OCR_INTERNAL_TOKEN", ""
).strip()

OCR_INPUT_HEIGHT = max(32, _env_int("ANPR_OCR_INPUT_HEIGHT", 48, 32, 128))
OCR_INPUT_WIDTH = max(128, _env_int("ANPR_OCR_INPUT_WIDTH", 320, 128, 1024))
OCR_CPU_THREADS = max(1, _env_int("ANPR_OCR_CPU_THREADS", 4, 1, 32))

ANPR_PRINT_ALL_OCR_CANDIDATES = (
    os.getenv(
        "ANPR_PRINT_ALL_OCR_CANDIDATES",
        "true",
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

OCR_ALLOWLIST = os.getenv(
    "ANPR_OCR_ALLOWLIST",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
).strip()

OCR_UPSCALE_MIN_HEIGHT = max(
    48,
    int(
        os.getenv(
            "ANPR_OCR_UPSCALE_MIN_HEIGHT",
            "96",
        )
    ),
)

OCR_USE_PREPROCESSING = (
    os.getenv(
        "ANPR_OCR_USE_PREPROCESSING",
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

PLATE_OCR_CONF = _env_float("ANPR_PLATE_OCR_CONF", 0.45)


# ============================================================
# OCR STARTUP / PREWARM
# ============================================================

ANPR_OCR_PREWARM = (
    os.getenv(
        "ANPR_OCR_PREWARM",
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
# MODEL VERSIONING
# ============================================================

ANPR_MODEL_VERSION = (
    os.getenv(
        "ANPR_MODEL_VERSION",
        "2.0",
    ).strip()
    or "2.0"
)

ANPR_DETECTOR_NAME = os.getenv(
    "ANPR_DETECTOR_NAME",
    "YOLOv8m-license-plate",
).strip()

ANPR_RECOGNIZER_NAME = os.getenv(
    "ANPR_RECOGNIZER_NAME",
    "Awiros-ANPR-OCR",
).strip()


# ============================================================
# PERSISTENCE
# ============================================================

ANPR_PERSISTENCE_ENABLED = (
    os.getenv(
        "ANPR_PERSISTENCE_ENABLED",
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

ANPR_PERSIST_MIN_INTERVAL_SECONDS = max(
    0.0,
    float(
        os.getenv(
            "ANPR_PERSIST_MIN_INTERVAL_SECONDS",
            "5.0",
        )
    ),
)

ANPR_PERSIST_CONFIDENCE_DELTA = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "ANPR_PERSIST_CONFIDENCE_DELTA",
                "0.10",
            )
        ),
    ),
)


# ============================================================
# INPUT LIMITS
# ============================================================

MAX_INPUT_WIDTH = max(
    320,
    int(
        os.getenv(
            "ANPR_MAX_INPUT_WIDTH",
            "1920",
        )
    ),
)

MAX_INPUT_HEIGHT = max(
    240,
    int(
        os.getenv(
            "ANPR_MAX_INPUT_HEIGHT",
            "1080",
        )
    ),
)


# ============================================================
# PLATE CROP LIMITS
# ============================================================

MIN_PLATE_WIDTH = max(
    10,
    int(
        os.getenv(
            "ANPR_MIN_PLATE_WIDTH",
            "18",
        )
    ),
)

MIN_PLATE_HEIGHT = max(
    5,
    int(
        os.getenv(
            "ANPR_MIN_PLATE_HEIGHT",
            "8",
        )
    ),
)

# Candidate crops smaller than the readable threshold are allowed into the
# temporal buffer so a later, sharper frame can rescue the track. They cannot
# become CONFIRMED solely from a tiny crop.
ANPR_MIN_READABLE_PLATE_WIDTH = _env_int(
    "ANPR_MIN_READABLE_PLATE_WIDTH", 40, MIN_PLATE_WIDTH, 512
)
ANPR_MIN_READABLE_PLATE_HEIGHT = _env_int(
    "ANPR_MIN_READABLE_PLATE_HEIGHT", 14, MIN_PLATE_HEIGHT, 256
)
ANPR_TARGET_PLATE_WIDTH = _env_int(
    "ANPR_TARGET_PLATE_WIDTH", 80, ANPR_MIN_READABLE_PLATE_WIDTH, 512
)

MAX_PLATE_WIDTH = max(
    MIN_PLATE_WIDTH,
    int(
        os.getenv(
            "ANPR_MAX_PLATE_WIDTH",
            "800",
        )
    ),
)

MAX_PLATE_HEIGHT = max(
    MIN_PLATE_HEIGHT,
    int(
        os.getenv(
            "ANPR_MAX_PLATE_HEIGHT",
            "300",
        )
    ),
)


# ============================================================
# PLATE TEXT LIMITS
# ============================================================

MIN_PLATE_LENGTH = max(
    1,
    int(
        os.getenv(
            "ANPR_MIN_PLATE_LENGTH",
            "4",
        )
    ),
)

MAX_PLATE_LENGTH = max(
    MIN_PLATE_LENGTH,
    int(
        os.getenv(
            "ANPR_MAX_PLATE_LENGTH",
            "12",
        )
    ),
)


PLATE_ALLOWED_PATTERN = re.compile(
    r"^[A-Z0-9]+$"
)


# ============================================================
# TEMPORAL AGGREGATION
# ============================================================

ANPR_ENABLE_TEMPORAL_AGGREGATION = (
    os.getenv(
        "ANPR_ENABLE_TEMPORAL_AGGREGATION",
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

ANPR_AGGREGATOR_MAX_OBSERVATIONS = max(
    3,
    min(
        100,
        int(
            os.getenv(
                "ANPR_AGGREGATOR_MAX_OBSERVATIONS",
                "20",
            )
        ),
    ),
)

ANPR_AGGREGATOR_MIN_OBSERVATIONS = max(
    1,
    min(
        ANPR_AGGREGATOR_MAX_OBSERVATIONS,
        int(
            os.getenv(
                "ANPR_AGGREGATOR_MIN_OBSERVATIONS",
                "3",
            )
        ),
    ),
)

ANPR_AGGREGATOR_MIN_SUPPORT = max(
    1,
    min(
        ANPR_AGGREGATOR_MAX_OBSERVATIONS,
        int(
            os.getenv(
                "ANPR_AGGREGATOR_MIN_SUPPORT",
                "2",
            )
        ),
    ),
)

ANPR_AGGREGATOR_MIN_CONFIDENCE = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "ANPR_AGGREGATOR_MIN_CONFIDENCE",
                "0.80",
            )
        ),
    ),
)

ANPR_AGGREGATOR_MIN_MARGIN = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "ANPR_AGGREGATOR_MIN_MARGIN",
                "0.10",
            )
        ),
    ),
)

ANPR_AGGREGATOR_TTL_SECONDS = max(
    1.0,
    min(
        600.0,
        float(
            os.getenv(
                "ANPR_AGGREGATOR_TTL_SECONDS",
                "30.0",
            )
        ),
    ),
)

ANPR_AGGREGATOR_DECAY_SECONDS = max(
    1.0,
    min(
        600.0,
        float(
            os.getenv(
                "ANPR_AGGREGATOR_DECAY_SECONDS",
                "15.0",
            )
        ),
    ),
)

ANPR_MAX_TRACK_STATES = max(
    100,
    min(
        100000,
        int(
            os.getenv(
                "ANPR_MAX_TRACK_STATES",
                "10000",
            )
        ),
    ),
)


# ============================================================
# PLATE CONFIDENCE LEVELS
# ============================================================

PLATE_PROVISIONAL_THRESHOLD = float(
    os.getenv(
        "ANPR_PLATE_PROVISIONAL_THRESHOLD",
        "0.60",
    )
)

PLATE_STABLE_THRESHOLD = float(
    os.getenv(
        "ANPR_PLATE_STABLE_THRESHOLD",
        "0.80",
    )
)

PLATE_STRONG_THRESHOLD = float(
    os.getenv(
        "ANPR_PLATE_STRONG_THRESHOLD",
        "0.92",
    )
)


# ============================================================
# MODEL OBJECTS
# ============================================================

_plate_model: Any = None
_plate_tensorrt_model: Any = None
_sahi_plate_model: Any = None
_sahi_initialization_failed = False
_ocr_engine: Any = None
_ocr_engine_version: str = "unknown"
_ocr_dictionary: list[str] = []
_ocr_initialization_attempted = False
_ocr_initialization_failed = False

_plate_model_lock = threading.RLock()
_ocr_lock = threading.RLock()


# ============================================================
# PERSISTENCE STATE
# ============================================================

_persistence_lock = threading.RLock()

_last_persisted_plate: dict[
    tuple[str, str, str],
    tuple[float, float],
] = {}


# ============================================================
# TEMPORAL STATE
# ============================================================

class _PlateEvidenceAggregator:

    __slots__ = (
        "observations",
        "last_seen",
    )

    def __init__(self) -> None:

        self.observations = deque(
            maxlen=ANPR_AGGREGATOR_MAX_OBSERVATIONS
        )

        self.last_seen = 0.0


_aggregator_lock = threading.RLock()

_plate_track_states: dict[
    tuple[str, str],
    _PlateEvidenceAggregator,
] = {}

_plate_crop_buffer_lock = threading.RLock()
_plate_crop_buffers: dict[tuple[str, str], deque] = {}
_plate_sahi_call_counts: dict[tuple[str, str], int] = {}


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return default

    if not math.isfinite(value):

        return default

    return value


def _safe_confidence(
    value: Any,
    default: float = 0.0,
) -> float:

    value = _safe_float(
        value,
        default,
    )

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def _safe_string(
    value: Any,
    max_length: int = 256,
) -> Optional[str]:

    if value is None:
        return None

    try:

        value = str(
            value
        ).strip()

    except Exception:

        return None

    if not value:
        return None

    value = value[
        :max_length
    ]

    if any(
        ord(char) < 32
        for char in value
    ):

        return None

    return value


def _utc_timestamp(
    value: Optional[datetime] = None,
) -> float:

    if value is None:

        return datetime.now(
            timezone.utc
        ).timestamp()

    try:

        if value.tzinfo is None:

            value = value.replace(
                tzinfo=timezone.utc
            )

        return value.timestamp()

    except Exception:

        return datetime.now(
            timezone.utc
        ).timestamp()


# ============================================================
# MODEL PATH VALIDATION
# ============================================================

def _validate_local_model_path(
    model_path: Path,
) -> Path:

    model_path = model_path.resolve()

    if not model_path.exists():

        raise FileNotFoundError(
            "ANPR plate model not found: "
            f"{model_path}"
        )

    if not model_path.is_file():

        raise RuntimeError(
            "ANPR plate model path is not a file"
        )

    if model_path.suffix.lower() not in {
        ".pt",
        ".onnx",
    }:

        raise RuntimeError(
            "Unsupported ANPR model format"
        )

    return model_path


# ============================================================
# LOAD PLATE MODEL
# ============================================================

def _tensor_rt_plate_enabled() -> bool:
    return os.getenv("TENSORRT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

def _get_plate_tensorrt_model() -> Any:
    global _plate_tensorrt_model
    if _plate_tensorrt_model is not None:
        return _plate_tensorrt_model
    with _plate_model_lock:
        if _plate_tensorrt_model is not None:
            return _plate_tensorrt_model
        state = tensorrt_runtime.status(validate_files=True)
        if not state.get("ready"):
            raise RuntimeError("validated TensorRT manifest is not ready for plate detector")
        tensorrt_runtime.load_engine("plate_detector")
        manifest_path = Path(state["manifest"])
        import json as _json
        data = _json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next((e for e in data.get("engines", []) if e.get("role") == "plate_detector"), None)
        if not entry:
            raise RuntimeError("plate_detector engine is not registered")
        engine_path = Path(str(entry["engine_path"]))
        if not engine_path.is_file():
            raise RuntimeError("plate_detector TensorRT engine is missing")
        from ultralytics import YOLO
        _plate_tensorrt_model = YOLO(str(engine_path))
        logger.info("ANPR TensorRT plate detector loaded: %s", engine_path.name)
        return _plate_tensorrt_model

def get_plate_model() -> Any:

    global _plate_model

    if _plate_model is not None:

        return _plate_model

    with _plate_model_lock:

        if _plate_model is not None:

            return _plate_model

        model_path = (
            _validate_local_model_path(
                PLATE_MODEL_PATH
            )
        )

        try:

            from ultralytics import YOLO
            model = YOLO(
                str(model_path)
            )

            names = getattr(
                model,
                "names",
                None,
            )

            if not names:

                raise RuntimeError(
                    "ANPR YOLO model has no class metadata"
                )

            _plate_model = model

            logger.info(
                "ANPR plate detector loaded: %s",
                model_path.name,
            )

            _anpr_log(
                f"PLATE_MODEL_READY "
                f"model={model_path.name} "
                f"classes={names}"
            )

            return model

        except Exception:

            logger.exception(
                "Failed to load ANPR plate detector"
            )

            raise


def _get_sahi_plate_model():
    """Load a SAHI adapter around the same configured ANPR weights."""
    global _sahi_plate_model, _sahi_initialization_failed
    if _sahi_plate_model is not None:
        return _sahi_plate_model
    if _sahi_initialization_failed and not ANPR_SAHI_REQUIRED:
        return None
    with _plate_model_lock:
        if _sahi_plate_model is not None:
            return _sahi_plate_model
        try:
            from sahi import AutoDetectionModel
            device = _resolve_plate_device()
            device_name = "cpu" if device == "cpu" else f"cuda:{int(device)}"
            _sahi_plate_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(_validate_local_model_path(PLATE_MODEL_PATH)),
                model=get_plate_model(),
                confidence_threshold=PLATE_DETECTION_CONF,
                device=device_name,
            )
            _anpr_log(
                f"SAHI_READY slices={ANPR_SAHI_SLICE_WIDTH}x{ANPR_SAHI_SLICE_HEIGHT} "
                f"overlap={ANPR_SAHI_OVERLAP_WIDTH:.2f}x{ANPR_SAHI_OVERLAP_HEIGHT:.2f}"
            )
            return _sahi_plate_model
        except Exception:
            _sahi_initialization_failed = True
            logger.exception("SAHI ANPR initialization failed")
            if ANPR_SAHI_REQUIRED:
                raise RuntimeError("SAHI is required but could not be initialized")
            return None


# ============================================================
# LOAD PP-OCRv5 ONNX ENGINE
# ============================================================

def _load_ocr_dictionary() -> list[str]:
    if not OCR_DICT_PATH.exists():
        raise FileNotFoundError(
            "PP-OCRv5 dictionary not found: "
            f"{OCR_DICT_PATH}. Download ppocrv5_en_dict.txt and place it there."
        )

    characters: list[str] = []
    with OCR_DICT_PATH.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.rstrip("\r\n")
            if value:
                characters.append(value[0] if len(value) > 1 else value)

    if not characters:
        raise RuntimeError(
            f"PP-OCRv5 dictionary is empty: {OCR_DICT_PATH}"
        )

    return characters


def _select_ocr_output(
    output_arrays: list[np.ndarray],
    dictionary_size: int,
) -> np.ndarray:
    """
    Select and orient the PP-OCRv5 CTC output tensor.

    The English PP-OCRv5 model uses ``use_space_char=True``. With a 436
    character dictionary, that gives 437 character tokens plus one CTC blank,
    i.e. 438 output classes. PaddleOCR's config explicitly enables the space
    character for PP-OCRv5 recognition.

    We therefore accept both common exports:
      * dictionary + blank                      -> 437 classes
      * dictionary + space + blank             -> 438 classes

    The current ONNX model exposes 438 classes.
    """
    if not output_arrays:
        raise RuntimeError("PP-OCRv5 ONNX model returned no outputs")

    dictionary_size = int(dictionary_size)
    supported_classes = {dictionary_size + 1, dictionary_size + 2}
    candidates: list[np.ndarray] = []

    for raw in output_arrays:
        value = np.asarray(raw)
        if value.ndim == 3 and value.shape[0] >= 1:
            value = value[0]
        if value.ndim != 2:
            continue

        if int(value.shape[-1]) in supported_classes:
            return value
        if int(value.shape[0]) in supported_classes:
            return value.T
        candidates.append(value)

    if not candidates:
        raise RuntimeError(
            "PP-OCRv5 ONNX output shape is unsupported: "
            + ", ".join(str(np.asarray(x).shape) for x in output_arrays)
        )

    def orientation_score(x: np.ndarray) -> tuple[int, int]:
        d0 = min(abs(int(x.shape[0]) - n) for n in supported_classes)
        d1 = min(abs(int(x.shape[1]) - n) for n in supported_classes)
        return (min(d0, d1), -max(x.shape))

    chosen = min(candidates, key=orientation_score)
    d0 = min(abs(int(chosen.shape[0]) - n) for n in supported_classes)
    d1 = min(abs(int(chosen.shape[1]) - n) for n in supported_classes)
    return chosen.T if d0 <= d1 else chosen


def _decode_ppocrv5_ctc(
    output: np.ndarray,
    dictionary: list[str],
) -> tuple[str, float]:
    """Decode PP-OCRv5 CTC probabilities into text.

    Class layout used by the current English ONNX export:
        0                    -> CTC blank
        1..len(dictionary)  -> dictionary characters
        len(dictionary)+1    -> optional space token

    This fixes the previous ``438 vs 437`` failure without silently shifting
    character indexes. The optional space token is ignored for license plates.
    """
    array = np.asarray(output, dtype=np.float32)
    if array.ndim != 2:
        raise RuntimeError(
            f"PP-OCRv5 CTC output must be 2D; got {array.shape}"
        )

    dictionary_size = len(dictionary)
    class_count = int(array.shape[-1])

    if class_count not in {dictionary_size + 1, dictionary_size + 2}:
        if int(array.shape[0]) in {dictionary_size + 1, dictionary_size + 2}:
            array = array.T
            class_count = int(array.shape[-1])

    if class_count not in {dictionary_size + 1, dictionary_size + 2}:
        raise RuntimeError(
            "PP-OCRv5 class dimension mismatch: "
            f"output={array.shape} dictionary={dictionary_size} "
            f"supported_classes={dictionary_size + 1},{dictionary_size + 2}"
        )

    min_value = float(np.min(array)) if array.size else 0.0
    max_value = float(np.max(array)) if array.size else 0.0
    row_sums = np.sum(array, axis=-1)
    already_probabilities = (
        min_value >= -1e-5
        and max_value <= 1.00001
        and np.mean(np.abs(row_sums - 1.0)) < 1e-2
    )

    if already_probabilities:
        probabilities = np.clip(array, 0.0, 1.0)
        probabilities = probabilities / np.maximum(
            np.sum(probabilities, axis=-1, keepdims=True),
            1e-12,
        )
    else:
        shifted = array - np.max(array, axis=-1, keepdims=True)
        exp_value = np.exp(np.clip(shifted, -80.0, 80.0))
        probabilities = exp_value / np.maximum(
            np.sum(exp_value, axis=-1, keepdims=True),
            1e-12,
        )

    classes = np.argmax(probabilities, axis=-1)
    scores = np.max(probabilities, axis=-1)

    chars: list[str] = []
    char_scores: list[float] = []
    previous = -1
    space_index = dictionary_size + 1 if class_count == dictionary_size + 2 else None

    for class_id, score in zip(classes.tolist(), scores.tolist()):
        class_id = int(class_id)

        # CTC blank.
        if class_id == 0:
            previous = class_id
            continue

        # Collapse repeated adjacent CTC symbols.
        if class_id == previous:
            continue

        # Optional space token used by the PP-OCRv5 English export.
        if space_index is not None and class_id == space_index:
            previous = class_id
            continue

        dictionary_index = class_id - 1
        if 0 <= dictionary_index < dictionary_size:
            char = dictionary[dictionary_index]
            if char != " ":
                chars.append(char)
                char_scores.append(float(score))

        previous = class_id

    text = "".join(chars)
    confidence = float(np.mean(char_scores)) if char_scores else 0.0

    return text, _safe_confidence(confidence, 0.0)


def get_ocr_engine():
    """Return the configured singleton OCR recognizer.

    Awiros is the production default. The previous ONNX recognizer remains an
    explicit rollback engine and is never selected silently on Awiros failure.
    """

    global _ocr_engine
    global _ocr_engine_version
    global _ocr_initialization_attempted
    global _ocr_initialization_failed
    global _ocr_dictionary

    if _ocr_engine is not None:
        return _ocr_engine

    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine

        if _ocr_initialization_failed:
            return None

        if _ocr_initialization_attempted:
            return None

        _ocr_initialization_attempted = True

        try:
            if OCR_ENGINE == "awiros_paddle":
                if os.getenv("INTEL_I_INFERENCE_MODE", "local") == "triton":
                    from inference.triton_ocr import TritonOCR
                    engine = TritonOCR().load()
                    model_path_label = "triton-awiros"
                elif ANPR_OCR_REMOTE_ENABLED:
                    engine = RemoteAwirosOCR(
                        base_url=ANPR_OCR_REMOTE_URL,
                        token=ANPR_OCR_INTERNAL_TOKEN,
                        timeout_seconds=ANPR_OCR_REMOTE_TIMEOUT_SECONDS,
                    ).load()
                    model_path_label = "remote-worker"
                else:
                    engine = AwirosOCR(
                        weights_path=AWIROS_MODEL_PATH,
                        dictionary_path=AWIROS_DICT_PATH,
                        paddleocr_dir=AWIROS_PADDLEOCR_DIR,
                        device=OCR_DEVICE,
                        strict_device=AWIROS_STRICT_DEVICE,
                    ).load()
                    model_path_label = AWIROS_MODEL_PATH.name

                _ocr_engine = engine
                _ocr_engine_version = engine.version + "/" + engine.actual_device
                _anpr_log(
                    "OCR_MODEL_READY engine=Awiros-ANPR-OCR "
                    f"device={engine.actual_device} "
                    f"model_path={model_path_label} "
                    f"remote={ANPR_OCR_REMOTE_ENABLED}"
                )
                return engine

            if not _OCR_RUNTIME_IMPORT_OK or ort is None:
                raise RuntimeError(
                    "onnxruntime is unavailable"
                )

            model_name = (
                OCR_MODEL_NAME.strip()
                or "en_PP-OCRv5_mobile_rec"
            )

            if not OCR_ONNX_MODEL_PATH.exists():
                raise FileNotFoundError(
                    "PP-OCRv5 ONNX model not found: "
                    f"{OCR_ONNX_MODEL_PATH}"
                )

            if not OCR_DICT_PATH.exists():
                raise FileNotFoundError(
                    "PP-OCRv5 dictionary not found: "
                    f"{OCR_DICT_PATH}"
                )

            _ocr_dictionary = _load_ocr_dictionary()

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = OCR_CPU_THREADS
            session_options.inter_op_num_threads = 1
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            available_providers = list(
                ort.get_available_providers()
            )

            cuda_available = (
                "CUDAExecutionProvider" in available_providers
            )

            if OCR_DEVICE == "cpu":
                selected_device = "cpu"
                providers = ["CPUExecutionProvider"]
                provider_options = [{}]

            elif OCR_DEVICE == "cuda":
                if not cuda_available:
                    raise RuntimeError(
                        "ANPR_OCR_DEVICE=cuda was requested, but "
                        "CUDAExecutionProvider is not available. "
                        f"Available providers: {available_providers}"
                    )

                selected_device = "cuda"
                providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
                provider_options = [
                    {
                        "device_id": str(OCR_CUDA_DEVICE_ID),
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": "1",
                    },
                    {},
                ]

            else:
                if cuda_available:
                    selected_device = "cuda"
                    providers = [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ]
                    provider_options = [
                        {
                            "device_id": str(OCR_CUDA_DEVICE_ID),
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": "1",
                        },
                        {},
                    ]
                else:
                    selected_device = "cpu"
                    providers = ["CPUExecutionProvider"]
                    provider_options = [{}]

                    _anpr_warning(
                        "OCR_CUDA_UNAVAILABLE "
                        "falling_back_to_cpu "
                        f"available_providers={available_providers}"
                    )

            _anpr_log(
                "OCR_INIT "
                "engine=PP-OCRv5-ONNX "
                f"model={model_name} "
                f"device={selected_device} "
                f"requested_device={OCR_DEVICE} "
                f"runtime=onnxruntime/{_OCR_RUNTIME_VERSION} "
                f"available_providers={available_providers} "
                f"providers={providers} "
                f"model_path={OCR_ONNX_MODEL_PATH.name}"
            )

            engine = ort.InferenceSession(
                str(OCR_ONNX_MODEL_PATH),
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options,
            )

            if not engine.get_inputs():
                raise RuntimeError(
                    "PP-OCRv5 ONNX model has no inputs"
                )

            actual_providers = list(
                engine.get_providers()
            )

            actual_device = (
                "cuda"
                if "CUDAExecutionProvider" in actual_providers
                else "cpu"
            )

            _ocr_engine = engine
            _ocr_engine_version = (
                f"ONNXRuntime/{_OCR_RUNTIME_VERSION}/"
                f"{model_name}/"
                f"{actual_device}"
            )

            logger.info(
                "ANPR OCR model loaded: %s",
                OCR_ONNX_MODEL_PATH.name,
            )

            _anpr_log(
                "OCR_MODEL_READY "
                "engine=PP-OCRv5-ONNX "
                f"model={model_name} "
                f"device={actual_device} "
                f"dict={OCR_DICT_PATH.name} "
                f"providers={actual_providers}"
            )

            return engine

        except Exception:
            _ocr_initialization_failed = True

            logger.exception("Failed to initialize ANPR OCR engine=%s", OCR_ENGINE)

            _anpr_error(
                "OCR_INIT_FAILED "
                f"engine={OCR_ENGINE} "
                f"model={OCR_MODEL_NAME} "
                f"requested_device={OCR_DEVICE}"
            )

            return None


# ============================================================
# VEHICLE IMAGE VALIDATION
# ============================================================



def _validate_vehicle_crop(
    crop: Optional[np.ndarray],
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

    if width <= 0 or height <= 0:

        return False

    if width > MAX_INPUT_WIDTH:

        return False

    if height > MAX_INPUT_HEIGHT:

        return False

    return True


# ============================================================
# MIXED-VEHICLE / ROI HELPERS
# ============================================================

def normalize_vehicle_class(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "-")
    if not normalized:
        return None
    return VEHICLE_CLASS_ALIASES.get(normalized, normalized)


def is_anpr_supported_vehicle(value: Optional[str]) -> bool:
    normalized = normalize_vehicle_class(value)
    if normalized is None:
        # Backward compatibility: older callers only pass a crop.
        return True
    supported = {
        VEHICLE_CLASS_ALIASES.get(item, item)
        for item in ANPR_SUPPORTED_VEHICLES
    }
    return normalized in supported


def _coerce_bbox(bbox: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _expanded_vehicle_crop_from_frame(
    source_frame: Optional[np.ndarray],
    vehicle_bbox: Any,
    vehicle_class: Optional[str],
) -> tuple[Optional[np.ndarray], Optional[list[int]]]:
    """Return a vehicle-class-aware crop from the original CCTV frame.

    Motorcycle crops deliberately extend farther downward because front/rear
    plates frequently sit below the detector's nominal motorcycle box.
    """
    if not ANPR_VEHICLE_CROP_EXPANSION_ENABLED:
        return None, None
    if source_frame is None or not isinstance(source_frame, np.ndarray) or source_frame.size == 0:
        return None, None
    parsed = _coerce_bbox(vehicle_bbox)
    if parsed is None:
        return None, None

    frame_h, frame_w = source_frame.shape[:2]
    x1, y1, x2, y2 = parsed
    width = x2 - x1
    height = y2 - y1
    normalized_class = normalize_vehicle_class(vehicle_class) or "car"
    left, top, right, bottom = VEHICLE_CROP_EXPANSION.get(
        normalized_class,
        (0.12, 0.10, 0.12, 0.25),
    )

    ex1 = max(0, int(math.floor(x1 - width * left)))
    ey1 = max(0, int(math.floor(y1 - height * top)))
    ex2 = min(frame_w, int(math.ceil(x2 + width * right)))
    ey2 = min(frame_h, int(math.ceil(y2 + height * bottom)))
    if ex2 <= ex1 or ey2 <= ey1:
        return None, None

    crop = np.ascontiguousarray(source_frame[ey1:ey2, ex1:ex2])
    if not _validate_vehicle_crop(crop):
        return None, None
    return crop, [ex1, ey1, ex2, ey2]


def _perspective_rectify_plate(plate_crop: np.ndarray) -> Optional[np.ndarray]:
    """Conservative contour-based deskew for tilted plate crops.

    The function returns None when a reliable quadrilateral cannot be found;
    it never replaces the original crop, it only contributes an OCR variant.
    """
    if not ANPR_PERSPECTIVE_CORRECTION:
        return None
    if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        return None
    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(max(1, plate_crop.shape[0] * plate_crop.shape[1]))
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.25:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(approx) != 4:
                continue
            pts = approx.reshape(4, 2).astype(np.float32)
            sums = pts.sum(axis=1)
            diffs = np.diff(pts, axis=1).reshape(-1)
            ordered = np.array(
                [
                    pts[np.argmin(sums)],
                    pts[np.argmin(diffs)],
                    pts[np.argmax(sums)],
                    pts[np.argmax(diffs)],
                ],
                dtype=np.float32,
            )
            tl, tr, br, bl = ordered
            width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if width < MIN_PLATE_WIDTH or height < MIN_PLATE_HEIGHT:
                continue
            destination = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(ordered, destination)
            rectified = cv2.warpPerspective(plate_crop, matrix, (width, height))
            if rectified.size > 0:
                return np.ascontiguousarray(rectified)
    except Exception:
        logger.debug("Plate perspective correction failed", exc_info=True)
    return None


def _indian_plate_format_score(text: Optional[str]) -> float:
    """Score format plausibility without rewriting uncertain OCR characters."""
    normalized = normalize_plate_text(text)
    if not normalized:
        return 0.0

    patterns = (
        # Most state registrations, e.g. GJ01AB1234 / KA1A1234.
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$",
        # Bharat Series, e.g. 22BH1234AA.
        r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$",
    )
    if any(re.fullmatch(pattern, normalized) for pattern in patterns):
        return 1.0

    letters = sum(ch.isalpha() for ch in normalized)
    digits = sum(ch.isdigit() for ch in normalized)
    if 6 <= len(normalized) <= 12 and letters >= 2 and digits >= 2:
        return 0.65
    if 4 <= len(normalized) <= 12 and (letters > 0 and digits > 0):
        return 0.40
    return 0.15


def _partial_character_consensus(observations: list[dict[str, Any]]) -> Optional[str]:
    """Return a conservative partial plate such as GJ01AB12??.

    Only same-length OCR observations are combined. Unknown positions are
    represented with '?'; this string is never persisted as a confirmed plate.
    """
    if not ANPR_PARTIAL_RESULTS or not observations:
        return None

    grouped: dict[int, list[tuple[str, float]]] = {}
    for observation in observations:
        text = normalize_plate_text(observation.get("plate"))
        if not text:
            continue
        weight = max(
            0.05,
            0.65 * _safe_confidence(observation.get("ocr_confidence", 0.0), 0.0)
            + 0.25 * _safe_confidence(observation.get("detection_confidence", 0.0), 0.0)
            + 0.10 * _safe_confidence(observation.get("quality_score", 0.0), 0.0),
        )
        grouped.setdefault(len(text), []).append((text, weight))

    if not grouped:
        return None
    length, candidates = max(
        grouped.items(),
        key=lambda item: (len(item[1]), sum(weight for _, weight in item[1])),
    )
    if len(candidates) < 2:
        return None

    chars: list[str] = []
    known = 0
    for index in range(length):
        counts: dict[str, float] = {}
        total = 0.0
        for text, weight in candidates:
            char = text[index]
            counts[char] = counts.get(char, 0.0) + weight
            total += weight
        winner, winner_weight = max(counts.items(), key=lambda item: item[1])
        agreement = winner_weight / total if total > 0.0 else 0.0
        if agreement >= ANPR_PARTIAL_CHAR_MIN_AGREEMENT:
            chars.append(winner)
            known += 1
        else:
            chars.append("?")

    if known / float(max(1, length)) < ANPR_PARTIAL_MIN_KNOWN_RATIO:
        return None
    partial = "".join(chars)
    return partial if "?" in partial else None


def _end_to_end_anpr_confidence(
    *,
    ocr_confidence: float,
    detection_confidence: float,
    quality_score: float,
    temporal_consistency: float,
    format_score: float,
    vehicle_confidence: Optional[float] = None,
    plate_width: int = 0,
    plate_height: int = 0,
) -> float:
    """Calibrated end-to-end ANPR confidence; never aliases vehicle confidence."""
    components = [
        (0.30, _safe_confidence(ocr_confidence, 0.0)),
        (0.25, _safe_confidence(detection_confidence, 0.0)),
        (0.20, _safe_confidence(temporal_consistency, 0.0)),
        (0.15, _safe_confidence(quality_score, 0.0)),
        (0.10, _safe_confidence(format_score, 0.0)),
    ]
    if vehicle_confidence is not None:
        # Preserve the relative ratios of the ANPR terms and reserve only a
        # small contribution for vehicle detection confidence.
        components = [(weight * 0.90, value) for weight, value in components]
        components.append((0.10, _safe_confidence(vehicle_confidence, 0.0)))

    score = sum(weight * value for weight, value in components) / max(
        1e-9, sum(weight for weight, _ in components)
    )

    # A tiny crop can be evidence but cannot honestly be called near-certain.
    if plate_width and plate_height:
        if plate_width < ANPR_MIN_READABLE_PLATE_WIDTH or plate_height < ANPR_MIN_READABLE_PLATE_HEIGHT:
            score = min(score, 0.54)
        elif plate_width < ANPR_TARGET_PLATE_WIDTH:
            size_ratio = plate_width / float(max(1, ANPR_TARGET_PLATE_WIDTH))
            score *= 0.80 + 0.20 * min(1.0, max(0.0, size_ratio))

    return _safe_confidence(score, 0.0)


# ============================================================
# PLATE NORMALIZATION
# ============================================================

def normalize_plate_text(
    text: Optional[str],
) -> Optional[str]:

    if text is None:

        return None

    if not isinstance(
        text,
        str,
    ):

        return None

    value = (
        text
        .strip()
        .upper()
    )

    if not value:

        return None

    # Common OCR separators.
    value = re.sub(
        r"[\s\-_.:/\\|]+",
        "",
        value,
    )

    # Only alphanumeric characters.
    value = re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )

    if not value:

        return None

    if len(value) < MIN_PLATE_LENGTH:

        return None

    if len(value) > MAX_PLATE_LENGTH:

        return None

    if not PLATE_ALLOWED_PATTERN.fullmatch(
        value
    ):

        return None

    return value


# ============================================================
# OPTIONAL OCR CHARACTER CORRECTION
# ============================================================

def _correct_common_ocr_confusions(
    text: Optional[str],
) -> Optional[str]:

    normalized = normalize_plate_text(
        text
    )

    if not normalized:

        return None

    # Conservative substitutions only.
    #
    # We deliberately DO NOT blindly convert every O -> 0
    # or I -> 1 because Indian registration plates can contain
    # both letters and numbers.
    #
    # This function therefore only removes OCR artifacts and
    # preserves the recognized alphanumeric sequence.

    return normalized


# ============================================================
# SAFE PLATE CROP
# ============================================================

def _safe_plate_crop(
    image: np.ndarray,
    bbox: Any,
) -> Optional[np.ndarray]:

    if not isinstance(
        image,
        np.ndarray,
    ):

        return None

    if image.size == 0:

        return None

    if not isinstance(
        bbox,
        (list, tuple, np.ndarray),
    ):

        return None

    if len(bbox) < 4:

        return None

    height, width = (
        image.shape[:2]
    )

    try:

        x1, y1, x2, y2 = [
            int(
                float(value)
            )
            for value in bbox[:4]
        ]

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None

    x1 = max(
        0,
        min(
            x1,
            width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            height - 1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            x2,
            width,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            y2,
            height,
        ),
    )

    if x2 <= x1 or y2 <= y1:

        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None

    crop_height, crop_width = (
        crop.shape[:2]
    )

    if crop_width < MIN_PLATE_WIDTH:

        return None

    if crop_height < MIN_PLATE_HEIGHT:

        return None

    # Bound dimensions.
    if crop_width > MAX_PLATE_WIDTH:

        new_width = MAX_PLATE_WIDTH

        new_height = max(
            MIN_PLATE_HEIGHT,
            int(
                crop_height
                * new_width
                / crop_width
            ),
        )

        crop = cv2.resize(
            crop,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    if crop.shape[0] > MAX_PLATE_HEIGHT:

        new_height = MAX_PLATE_HEIGHT

        new_width = max(
            MIN_PLATE_WIDTH,
            int(
                crop.shape[1]
                * new_height
                / crop.shape[0]
            ),
        )

        crop = cv2.resize(
            crop,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    return crop


# ============================================================
# PLATE IMAGE QUALITY
# ============================================================

def estimate_plate_quality(
    plate_crop: Optional[np.ndarray],
) -> float:

    if (
        plate_crop is None
        or not isinstance(
            plate_crop,
            np.ndarray,
        )
        or plate_crop.size == 0
    ):

        return 0.0

    try:

        gray = cv2.cvtColor(
            plate_crop,
            cv2.COLOR_BGR2GRAY,
        )

        height, width = (
            gray.shape[:2]
        )

        if (
            width < MIN_PLATE_WIDTH
            or height < MIN_PLATE_HEIGHT
        ):

            return 0.0

        brightness = float(
            np.mean(gray)
        )

        blur_score_raw = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        if not (
            math.isfinite(
                brightness
            )
            and math.isfinite(
                blur_score_raw
            )
        ):

            return 0.0

        brightness_score = max(
            0.0,
            1.0
            - abs(
                brightness
                - 128.0
            )
            / 128.0,
        )

        blur_score = min(
            1.0,
            max(
                0.0,
                blur_score_raw
                / 300.0,
            ),
        )

        aspect = (
            width / max(
                height,
                1,
            )
        )

        # Indian motorcycle plates may be close to square/two-line, while car
        # plates are usually wider. Do not punish a valid bike plate simply
        # because it is not a 2:1+ rectangle.
        if 0.80 <= aspect <= 6.50:
            aspect_score = 1.0
        elif 0.60 <= aspect <= 8.00:
            aspect_score = 0.65
        else:
            aspect_score = 0.30

        size_score = min(
            1.0,
            max(0.0, width / float(max(1, ANPR_TARGET_PLATE_WIDTH))),
        )

        return _safe_confidence(
            (0.25 * brightness_score)
            + (0.35 * blur_score)
            + (0.20 * aspect_score)
            + (0.20 * size_score),
            0.0,
        )

    except Exception:

        return 0.0


# ============================================================
# PLATE DETECTION
# ============================================================


def _resolve_plate_device():
    """
    Resolve a safe Ultralytics plate-detector device.

    CUDA_VISIBLE_DEVICES remaps the visible GPUs, so the application
    must select an index within torch.cuda.device_count().
    """
    import torch

    requested = str(
        os.getenv(
            "ANPR_PLATE_DEVICE",
            "auto",
        )
    ).strip().lower()

    if requested == "cpu":
        return "cpu"

    cuda_available = (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    )

    if not cuda_available:
        if requested not in {
            "auto",
            "cpu",
        }:
            logger.warning(
                "ANPR plate CUDA device requested but CUDA is unavailable; "
                "using CPU"
            )

        return "cpu"

    if requested in {
        "",
        "auto",
        "cuda",
    }:
        return 0

    try:
        device_id = int(requested)
    except ValueError:
        logger.warning(
            "Invalid ANPR_PLATE_DEVICE=%r; using visible CUDA device 0",
            requested,
        )
        return 0

    device_count = torch.cuda.device_count()

    if device_id < 0 or device_id >= device_count:
        logger.warning(
            "ANPR_PLATE_DEVICE=%s is outside visible CUDA range 0..%s; "
            "using device 0",
            device_id,
            device_count - 1,
        )
        return 0

    return device_id


def detect_plate_crops(
    vehicle_crop: np.ndarray,
    *,
    use_sahi: bool = False,
) -> list[
    tuple[
        np.ndarray,
        float,
        list[float],
    ]
]:

    if not _validate_vehicle_crop(
        vehicle_crop
    ):

        return []

    # Make an isolated contiguous read-only-ish inference buffer so
    # downstream model code cannot observe a mutable/non-contiguous ROI.
    vehicle_crop = np.ascontiguousarray(vehicle_crop)

    if use_sahi and ANPR_SAHI_ENABLED:
        height, width = vehicle_crop.shape[:2]
        if width >= ANPR_SAHI_MIN_IMAGE_WIDTH or height >= ANPR_SAHI_MIN_IMAGE_HEIGHT:
            try:
                from sahi.predict import get_sliced_prediction
                adapter = _get_sahi_plate_model()
                if adapter is not None:
                    with _plate_model_lock:
                        sliced = get_sliced_prediction(
                            vehicle_crop,
                            adapter,
                            slice_height=min(ANPR_SAHI_SLICE_HEIGHT, height),
                            slice_width=min(ANPR_SAHI_SLICE_WIDTH, width),
                            overlap_height_ratio=ANPR_SAHI_OVERLAP_HEIGHT,
                            overlap_width_ratio=ANPR_SAHI_OVERLAP_WIDTH,
                            perform_standard_pred=False,
                            postprocess_type="GREEDYNMM",
                            postprocess_match_metric="IOS",
                            postprocess_match_threshold=ANPR_SAHI_POSTPROCESS_MATCH_THRESHOLD,
                            postprocess_class_agnostic=True,
                            verbose=0,
                        )
                    candidates = []
                    ordered = sorted(
                        sliced.object_prediction_list,
                        key=lambda item: float(item.score.value),
                        reverse=True,
                    )
                    for prediction in ordered[:MAX_PLATES_PER_VEHICLE]:
                        confidence = _safe_confidence(prediction.score.value, 0.0)
                        box = [float(value) for value in prediction.bbox.to_xyxy()]
                        plate_crop = _safe_plate_crop(vehicle_crop, box)
                        if plate_crop is not None and confidence >= PLATE_DETECTION_CONF:
                            candidates.append((plate_crop, confidence, box))
                    if candidates:
                        _anpr_log(f"SAHI_PLATES count={len(candidates)} image={width}x{height}")
                        return candidates
            except Exception:
                logger.exception("SAHI plate detection failed")
                if ANPR_SAHI_REQUIRED:
                    raise

    try:
        model = _get_plate_tensorrt_model() if _tensor_rt_plate_enabled() else get_plate_model()
    except Exception:
        if os.getenv("TENSORRT_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        logger.warning("TensorRT plate detector unavailable; using configured ANPR fallback model")
        model = get_plate_model()

    try:

        with _plate_model_lock:
            plate_device = _resolve_plate_device()
            predict_kwargs = {
                "source": vehicle_crop,
                "conf": PLATE_DETECTION_CONF,
                "iou": PLATE_IOU,
                "max_det": MAX_PLATES_PER_VEHICLE,
                "verbose": False,
            }
            # PyTorch/ONNX Ultralytics models can use the larger ANPR input and
            # explicit device. TensorRT engines may be built with a static shape,
            # so do not override their engine dimensions here.
            if not _tensor_rt_plate_enabled():
                predict_kwargs["imgsz"] = ANPR_DETECT_IMAGE_SIZE
                predict_kwargs["device"] = plate_device
            try:
                results = model.predict(**predict_kwargs)
            except RuntimeError as exc:
                # 1280 gives the strongest small-plate recall, but a 4 GB GPU
                # can occasionally run out of memory. Retry once at the configured
                # fallback size instead of dropping the whole ANPR observation.
                is_oom = "out of memory" in str(exc).lower()
                if is_oom and not _tensor_rt_plate_enabled():
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    fallback_size = min(
                        ANPR_DETECT_IMAGE_SIZE,
                        ANPR_DETECT_FALLBACK_IMAGE_SIZE,
                    )
                    if fallback_size >= ANPR_DETECT_IMAGE_SIZE:
                        raise
                    _anpr_warning(
                        "PLATE_DETECT_OOM_RETRY "
                        f"imgsz={ANPR_DETECT_IMAGE_SIZE}->{fallback_size}"
                    )
                    predict_kwargs["imgsz"] = fallback_size
                    results = model.predict(**predict_kwargs)
                else:
                    raise

        if not results:

            return []

        result = results[0]

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:

            return []

        xyxy = getattr(
            boxes,
            "xyxy",
            None,
        )

        confidence = getattr(
            boxes,
            "conf",
            None,
        )

        if xyxy is None or confidence is None:

            return []

        xyxy = (
            xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confidence = (
            confidence
            .detach()
            .cpu()
            .numpy()
        )

        candidates = []

        for (
            box,
            detection_confidence,
        ) in zip(
            xyxy,
            confidence,
        ):

            if (
                len(candidates)
                >= MAX_PLATES_PER_VEHICLE
            ):

                break

            detection_confidence = (
                _safe_confidence(
                    detection_confidence,
                    0.0,
                )
            )

            if (
                detection_confidence
                < PLATE_DETECTION_CONF
            ):

                continue

            plate_crop = (
                _safe_plate_crop(
                    vehicle_crop,
                    box,
                )
            )

            if plate_crop is None:

                continue

            candidates.append(
                (
                    plate_crop,
                    detection_confidence,
                    [
                        float(
                            value
                        )
                        for value
                        in box[:4]
                    ],
                )
            )

            _anpr_log(
                f"PLATE_DETECTED "
                f"confidence={detection_confidence:.3f} "
                f"bbox={[round(float(v), 1) for v in box[:4]]} "
                f"crop_size={plate_crop.shape[1]}x{plate_crop.shape[0]}"
            )

        return candidates

    except Exception:

        logger.exception(
            "ANPR plate detection failed"
        )

        return []


# ============================================================
# OCR RECOGNITION
# ============================================================


def _preprocess_ppocrv5(plate_crop: np.ndarray) -> np.ndarray:
    """Convert a BGR plate crop to the [1,3,48,320] PP-OCRv5 input."""

    if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        raise ValueError("Invalid plate crop")

    image = np.ascontiguousarray(plate_crop)

    target_h = OCR_INPUT_HEIGHT
    target_w = OCR_INPUT_WIDTH

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid plate dimensions")

    scale = target_h / float(height)
    resized_w = max(1, min(target_w, int(round(width * scale))))

    resized = cv2.resize(
        image,
        (resized_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    # PaddleOCR's recognition preprocessing uses BGR input, normalization to
    # [-1,1], CHW layout and right padding.
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5

    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.float32,
    )
    canvas[:, :resized_w, :] = normalized

    tensor = np.transpose(
        canvas,
        (2, 0, 1),
    )[None, ...]

    return np.ascontiguousarray(tensor, dtype=np.float32)


def _resize_plate_for_ocr(
    plate_crop: np.ndarray,
) -> np.ndarray:
    """Upscale a small plate crop while preserving aspect ratio."""
    if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        raise ValueError("Invalid plate crop")

    height, width = plate_crop.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid plate dimensions")

    target_h = max(48, min(256, OCR_UPSCALE_MIN_HEIGHT))
    if height >= target_h:
        return np.ascontiguousarray(plate_crop)

    scale = target_h / float(height)
    new_width = max(1, int(round(width * scale)))

    return np.ascontiguousarray(
        cv2.resize(
            plate_crop,
            (new_width, target_h),
            interpolation=cv2.INTER_CUBIC,
        )
    )


def _build_ocr_variants(
    plate_crop: np.ndarray,
) -> list[np.ndarray]:
    """Build original/enhanced/thresholded CPU OCR variants."""
    base = _resize_plate_for_ocr(plate_crop)
    variants: list[np.ndarray] = [base]

    if not OCR_USE_PREPROCESSING:
        return variants

    try:
        rectified = _perspective_rectify_plate(base)
        if rectified is not None and rectified.size > 0:
            variants.append(_resize_plate_for_ocr(rectified))

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        enhanced = clahe.apply(gray)

        variants.append(
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        )

        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        variants.append(
            cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
        )

        adaptive_block = 31
        if min(enhanced.shape[:2]) < adaptive_block:
            adaptive_block = max(3, min(enhanced.shape[:2]) | 1)
            if adaptive_block % 2 == 0:
                adaptive_block -= 1
            adaptive_block = max(3, adaptive_block)

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            8,
        )
        variants.append(
            cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        )

        # Difficult Indian CCTV conditions: bounded low-light/blur/glare/roll
        # variants. These are OCR candidates only; temporal voting/fusion remains
        # authoritative and no single enhanced frame can finalize an identity.
        adaptive_variants, _conditions = build_adaptive_variants(base)
        variants.extend(adaptive_variants)

    except Exception:
        logger.exception(
            "Failed to build PP-OCRv5 preprocessing variants"
        )

    max_variants = max(1, min(16, int(os.getenv("ANPR_MAX_OCR_VARIANTS", "10"))))
    return [
        np.ascontiguousarray(item)
        for item in variants
        if isinstance(item, np.ndarray) and item.size > 0
    ][:max_variants]


def _build_two_line_row_pairs(plate_crop: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build top/bottom OCR row pairs for square/two-line motorcycle plates."""
    if not ANPR_TWO_LINE_PLATES:
        return []
    if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        return []

    base = _resize_plate_for_ocr(plate_crop)
    height, width = base.shape[:2]
    aspect = width / float(max(1, height))
    if aspect > ANPR_TWO_LINE_MAX_ASPECT or height < 24:
        return []

    overlap_px = int(round(height * ANPR_TWO_LINE_SPLIT_OVERLAP))
    split = height // 2
    top_end = min(height, split + overlap_px)
    bottom_start = max(0, split - overlap_px)
    if top_end <= 0 or bottom_start >= height:
        return []

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    top = np.ascontiguousarray(base[:top_end, :])
    bottom = np.ascontiguousarray(base[bottom_start:, :])
    if top.size and bottom.size:
        pairs.append((_resize_plate_for_ocr(top), _resize_plate_for_ocr(bottom)))

    # A CLAHE pair helps faded/dirty bike plates but remains only an OCR
    # candidate; it does not override temporal confirmation.
    try:
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        enhanced = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
        top_e = np.ascontiguousarray(enhanced[:top_end, :])
        bottom_e = np.ascontiguousarray(enhanced[bottom_start:, :])
        if top_e.size and bottom_e.size:
            pairs.append((_resize_plate_for_ocr(top_e), _resize_plate_for_ocr(bottom_e)))
    except Exception:
        logger.debug("Two-line OCR enhancement failed", exc_info=True)

    return pairs[:2]


def _plate_text_is_plausible(
    text: Optional[str],
) -> bool:
    normalized = normalize_plate_text(text)

    if not normalized:
        return False

    if len(normalized) < MIN_PLATE_LENGTH:
        return False

    if len(normalized) > MAX_PLATE_LENGTH:
        return False

    letters = sum(char.isalpha() for char in normalized)
    digits = sum(char.isdigit() for char in normalized)

    if letters < 2 or digits < 2:
        return False

    return True


def _run_ppocrv5_onnx(
    ocr_engine: Any,
    plate_crop: np.ndarray,
) -> tuple[Optional[str], float]:
    input_meta = ocr_engine.get_inputs()[0]
    input_name = input_meta.name

    image_tensor = _preprocess_ppocrv5(plate_crop)
    outputs = ocr_engine.run(
        None,
        {input_name: image_tensor},
    )

    output = _select_ocr_output(
        outputs,
        dictionary_size=len(_ocr_dictionary),
    )

    if ANPR_PRINT_ALL_OCR_CANDIDATES:
        _anpr_log(
            "OCR_OUTPUT_SHAPE "
            f"shape={tuple(output.shape)} "
            f"dictionary={len(_ocr_dictionary)}"
        )

    text, confidence = _decode_ppocrv5_ctc(
        output,
        _ocr_dictionary,
    )

    normalized = normalize_plate_text(text)
    if normalized and _plate_text_is_plausible(normalized):
        return normalized, confidence

    return normalized, confidence


def _run_selected_ocr(
    ocr_engine: Any,
    plate_crop: np.ndarray,
) -> tuple[Optional[str], float]:
    if OCR_ENGINE == "awiros_paddle":
        text, confidence = ocr_engine.recognize(plate_crop)
        return normalize_plate_text(text), _safe_confidence(confidence, 0.0)
    return _run_ppocrv5_onnx(ocr_engine, plate_crop)


def read_plate(
    plate_crop: np.ndarray,
) -> tuple[Optional[str], float]:
    """Read a plate from original/enhanced and optional two-line variants."""
    if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        return None, 0.0

    try:
        ocr = get_ocr_engine()
        if ocr is None:
            _anpr_warning(
                "OCR_UNAVAILABLE "
                f"model={OCR_MODEL_NAME} device={OCR_DEVICE}"
            )
            return None, 0.0

        variants = _build_ocr_variants(plate_crop)
        ranked_candidates: list[tuple[float, str, float, str]] = []
        candidate_count = 0

        with _ocr_lock:
            for variant_index, variant in enumerate(variants):
                try:
                    text, score = _run_selected_ocr(ocr, variant)
                except Exception as exc:
                    logger.exception(
                        "%s OCR inference failed on variant=%s",
                        OCR_ENGINE,
                        variant_index,
                    )
                    _anpr_error(
                        "OCR_VARIANT_FAILED "
                        f"variant={variant_index} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    continue

                normalized = normalize_plate_text(text)
                if not normalized:
                    continue

                candidate_count += 1
                score = _safe_confidence(score, 0.0)
                plausible = _plate_text_is_plausible(normalized)
                format_score = _indian_plate_format_score(normalized)
                rank_score = _safe_confidence(
                    score
                    + (0.05 if plausible else -0.20)
                    + (0.05 * format_score),
                    0.0,
                )
                ranked_candidates.append(
                    (rank_score, normalized, score, f"variant:{variant_index}")
                )

                if ANPR_PRINT_ALL_OCR_CANDIDATES:
                    _anpr_log(
                        "OCR_CANDIDATE "
                        f"engine={OCR_ENGINE} "
                        f"variant={variant_index} "
                        f"text={normalized} "
                        f"confidence={score:.3f} "
                        f"format={format_score:.2f} "
                        f"plausible={str(plausible).lower()}"
                    )

            # Motorcycle plates are frequently square/two-line. Read the two
            # rows separately and concatenate them in physical top->bottom order.
            for pair_index, (top_crop, bottom_crop) in enumerate(
                _build_two_line_row_pairs(plate_crop)
            ):
                try:
                    top_text, top_conf = _run_selected_ocr(ocr, top_crop)
                    bottom_text, bottom_conf = _run_selected_ocr(ocr, bottom_crop)
                except Exception as exc:
                    logger.debug(
                        "Two-line OCR failed pair=%s: %s",
                        pair_index,
                        exc,
                        exc_info=True,
                    )
                    continue

                top = normalize_plate_text(top_text)
                bottom = normalize_plate_text(bottom_text)
                if not top or not bottom:
                    continue

                combined = normalize_plate_text(top + bottom)
                if not combined:
                    continue

                candidate_count += 1
                top_conf = _safe_confidence(top_conf, 0.0)
                bottom_conf = _safe_confidence(bottom_conf, 0.0)
                combined_conf = min(top_conf, bottom_conf)
                plausible = _plate_text_is_plausible(combined)
                format_score = _indian_plate_format_score(combined)
                rank_score = _safe_confidence(
                    combined_conf
                    + (0.08 if plausible else -0.20)
                    + (0.08 * format_score),
                    0.0,
                )
                ranked_candidates.append(
                    (rank_score, combined, combined_conf, f"two-line:{pair_index}")
                )

                if ANPR_PRINT_ALL_OCR_CANDIDATES:
                    _anpr_log(
                        "OCR_TWO_LINE_CANDIDATE "
                        f"pair={pair_index} "
                        f"text={combined} "
                        f"confidence={combined_conf:.3f} "
                        f"format={format_score:.2f}"
                    )

        if not ranked_candidates:
            _anpr_warning(f"OCR_NO_VALID_TEXT candidates={candidate_count}")
            return None, 0.0

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_text, best_raw_score, best_source = ranked_candidates[0]

        _anpr_log(
            "OCR_BEST "
            f"engine={OCR_ENGINE} "
            f"model={OCR_MODEL_NAME} "
            f"source={best_source} "
            f"text={best_text} "
            f"confidence={best_raw_score:.3f} "
            f"threshold={PLATE_OCR_CONF:.3f}"
        )

        if best_raw_score < PLATE_OCR_CONF:
            _anpr_warning(
                "OCR_LOW_CONFIDENCE "
                f"text={best_text} "
                f"confidence={best_raw_score:.3f} "
                "returning_as_weak_evidence=true"
            )

        _anpr_log(
            "PLATE_NUMBER "
            f"text={best_text} "
            f"confidence={best_raw_score:.3f}"
        )
        return best_text, best_raw_score

    except Exception:
        logger.exception("ANPR OCR recognition failed | engine=%s", OCR_ENGINE)
        _anpr_error(
            "OCR_RECOGNITION_FAILED "
            f"engine={OCR_ENGINE} "
            f"model={OCR_MODEL_NAME}"
        )
        return None, 0.0



def _aggregation_key(
    camera_id: Optional[str],
    local_track_id: Optional[str],
) -> Optional[
    tuple[str, str]
]:

    if (
        camera_id is None
        or local_track_id is None
    ):

        return None

    camera = str(
        camera_id
    ).strip()

    track = str(
        local_track_id
    ).strip()

    if not camera or not track:

        return None

    return (
        camera,
        track,
    )


def _cleanup_aggregator_states(
    now: float,
) -> None:

    expired = [
        key
        for key, state
        in _plate_track_states.items()
        if (
            now
            - state.last_seen
            > ANPR_AGGREGATOR_TTL_SECONDS
        )
    ]

    for key in expired:

        _plate_track_states.pop(
            key,
            None,
        )

    overflow = (
        len(
            _plate_track_states
        )
        - ANPR_MAX_TRACK_STATES
    )

    if overflow <= 0:

        return

    oldest = sorted(
        _plate_track_states.items(),
        key=lambda item:
            item[1].last_seen,
    )[
        :overflow
    ]

    for key, _ in oldest:

        _plate_track_states.pop(
            key,
            None,
        )


# ============================================================
# TEMPORAL WEIGHTED CONSENSUS
# ============================================================

def _weighted_plate_consensus(
    observations: list[
        dict[str, Any]
    ],
    now: float,
) -> dict[str, Any]:

    if not observations:

        return {
            "plate":
                None,

            "confidence":
                0.0,

            "support_count":
                0,

            "observation_count":
                0,

            "consensus":
                0.0,

            "margin":
                0.0,

            "status":
                "NO_EVIDENCE",
        }

    scores: dict[
        str,
        float,
    ] = {}

    counts: Counter[
        str
    ] = Counter()

    best: dict[
        str,
        dict[str, float],
    ] = {}

    for observation in observations:

        plate = normalize_plate_text(
            observation.get(
                "plate"
            )
        )

        if not plate:

            continue

        ocr = _safe_confidence(
            observation.get(
                "ocr_confidence",
                0.0,
            ),
            0.0,
        )

        detection = _safe_confidence(
            observation.get(
                "detection_confidence",
                0.0,
            ),
            0.0,
        )

        quality = _safe_confidence(
            observation.get(
                "quality_score",
                1.0,
            ),
            1.0,
        )

        timestamp = _safe_float(
            observation.get(
                "timestamp",
                now,
            ),
            now,
        )

        age = max(
            0.0,
            now
            - timestamp,
        )

        temporal_weight = max(
            0.05,
            math.exp(
                -age
                / ANPR_AGGREGATOR_DECAY_SECONDS
            ),
        )

        weight = (
            (
                0.60
                * ocr
            )
            + (
                0.25
                * detection
            )
            + (
                0.15
                * quality
            )
        ) * temporal_weight

        if not math.isfinite(
            weight
        ):

            continue

        scores[
            plate
        ] = (
            scores.get(
                plate,
                0.0,
            )
            + weight
        )

        counts[
            plate
        ] += 1

        previous_best = best.get(
            plate
        )

        if (
            previous_best is None
            or ocr
            > previous_best[
                "ocr_confidence"
            ]
        ):

            best[
                plate
            ] = {
                "ocr_confidence":
                    ocr,

                "detection_confidence":
                    detection,

                "quality_score":
                    quality,
            }

    if not scores:

        return {
            "plate":
                None,

            "confidence":
                0.0,

            "support_count":
                0,

            "observation_count":
                len(
                    observations
                ),

            "consensus":
                0.0,

            "margin":
                0.0,

            "status":
                "NO_VALID_EVIDENCE",
        }

    ranked = sorted(
        scores.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    winner, winner_score = (
        ranked[0]
    )

    runner_score = (
        ranked[1][1]
        if len(
            ranked
        ) > 1
        else 0.0
    )

    total_score = sum(
        scores.values()
    )

    consensus = (
        winner_score
        / total_score
        if total_score > 0.0
        else 0.0
    )

    support = counts[
        winner
    ]

    winner_quality = best[
        winner
    ]

    format_score = _indian_plate_format_score(winner)
    partial_plate = _partial_character_consensus(observations)

    confidence = _safe_confidence(
        (0.50 * winner_quality["ocr_confidence"])
        + (0.18 * winner_quality["detection_confidence"])
        + (0.10 * winner_quality["quality_score"])
        + (0.12 * consensus)
        + (0.10 * format_score),
        0.0,
    )

    margin = _safe_confidence(
        (
            winner_score
            - runner_score
        )
        / winner_score
        if winner_score > 0.0
        else 0.0,
        0.0,
    )

    if (
        len(
            observations
        )
        >= ANPR_AGGREGATOR_MIN_OBSERVATIONS
        and support
        >= ANPR_AGGREGATOR_MIN_SUPPORT
        and confidence
        >= ANPR_AGGREGATOR_MIN_CONFIDENCE
        and margin
        >= ANPR_AGGREGATOR_MIN_MARGIN
    ):

        status = "STABLE"

    elif (
        confidence
        >= PLATE_PROVISIONAL_THRESHOLD
    ):

        status = "PROVISIONAL"

    else:

        status = "LOW_CONFIDENCE"

    return {

        "plate":
            winner,

        "confidence":
            confidence,

        "support_count":
            int(
                support
            ),

        "observation_count":
            len(
                observations
            ),

        "consensus":
            _safe_confidence(
                consensus,
                0.0,
            ),

        "margin":
            margin,

        "status":
            status,

        "partial_plate":
            partial_plate,

        "format_score":
            format_score,

        "best_ocr_confidence":
            winner_quality[
                "ocr_confidence"
            ],

        "best_detection_confidence":
            winner_quality[
                "detection_confidence"
            ],

        "best_quality_score":
            winner_quality[
                "quality_score"
            ],
    }


# ============================================================
# UPDATE TEMPORAL EVIDENCE
# ============================================================

def update_plate_evidence(
    *,
    camera_id: Optional[str],
    local_track_id: Optional[str],
    plate: Optional[str],
    ocr_confidence: float,
    detection_confidence: float,
    quality_score: float = 1.0,
    frame_timestamp: Optional[datetime] = None,
) -> dict[str, Any]:

    key = _aggregation_key(
        camera_id,
        local_track_id,
    )

    normalized = normalize_plate_text(
        plate
    )

    if (
        key is None
        or not ANPR_ENABLE_TEMPORAL_AGGREGATION
    ):

        if normalized:

            confidence = (
                _safe_confidence(
                    ocr_confidence,
                    0.0,
                )
            )

            status = (
                "STABLE"
                if confidence
                >= PLATE_STRONG_THRESHOLD
                else "PROVISIONAL"
            )

            return {
                "plate":
                    normalized,

                "confidence":
                    confidence,

                "support_count":
                    1,

                "observation_count":
                    1,

                "consensus":
                    1.0,

                "margin":
                    1.0,

                "status":
                    status,
            }

        return {
            "plate":
                None,

            "confidence":
                0.0,

            "support_count":
                0,

            "observation_count":
                0,

            "consensus":
                0.0,

            "margin":
                0.0,

            "status":
                "NO_EVIDENCE",
        }

    now = _utc_timestamp(
        frame_timestamp
    )

    with _aggregator_lock:

        state = (
            _plate_track_states.get(
                key
            )
        )

        if state is None:

            state = (
                _PlateEvidenceAggregator()
            )

            _plate_track_states[
                key
            ] = state

        state.last_seen = now

        if normalized:

            state.observations.append(
                {
                    "plate":
                        normalized,

                    "ocr_confidence":
                        _safe_confidence(
                            ocr_confidence,
                            0.0,
                        ),

                    "detection_confidence":
                        _safe_confidence(
                            detection_confidence,
                            0.0,
                        ),

                    "quality_score":
                        _safe_confidence(
                            quality_score,
                            1.0,
                        ),

                    "timestamp":
                        now,
                }
            )

        _cleanup_aggregator_states(
            now
        )

        return _weighted_plate_consensus(
            list(
                state.observations
            ),
            now,
        )


# ============================================================
# READ CURRENT CONSENSUS
# ============================================================

def get_plate_consensus(
    camera_id: Optional[str],
    local_track_id: Optional[str],
) -> dict[str, Any]:

    key = _aggregation_key(
        camera_id,
        local_track_id,
    )

    if key is None:

        return {
            "plate":
                None,

            "confidence":
                0.0,

            "support_count":
                0,

            "observation_count":
                0,

            "consensus":
                0.0,

            "margin":
                0.0,

            "status":
                "NO_TRACK",
        }

    now = (
        datetime.now(
            timezone.utc
        ).timestamp()
    )

    with _aggregator_lock:

        state = (
            _plate_track_states.get(
                key
            )
        )

        if state is None:

            return {
                "plate":
                    None,

                "confidence":
                    0.0,

                "support_count":
                    0,

                "observation_count":
                    0,

                "consensus":
                    0.0,

                "margin":
                    0.0,

                "status":
                    "NO_EVIDENCE",
            }

        if (
            now
            - state.last_seen
            > ANPR_AGGREGATOR_TTL_SECONDS
        ):

            _plate_track_states.pop(
                key,
                None,
            )

            return {
                "plate":
                    None,

                "confidence":
                    0.0,

                "support_count":
                    0,

                "observation_count":
                    0,

                "consensus":
                    0.0,

                "margin":
                    0.0,

                "status":
                    "EXPIRED",
            }

        return _weighted_plate_consensus(
            list(
                state.observations
            ),
            now,
        )


# ============================================================
# CLEAR TEMPORAL STATE
# ============================================================

def clear_plate_track_state(
    camera_id: str,
    local_track_id: str,
) -> None:

    key = _aggregation_key(
        camera_id,
        local_track_id,
    )

    if key is None:

        return

    with _aggregator_lock:

        _plate_track_states.pop(
            key,
            None,
        )

    buffer_key = (str(camera_id), str(local_track_id))
    with _plate_crop_buffer_lock:
        _plate_crop_buffers.pop(buffer_key, None)
        _plate_sahi_call_counts.pop(buffer_key, None)


def clear_all_plate_track_states() -> None:

    with _aggregator_lock:

        _plate_track_states.clear()

    with _plate_crop_buffer_lock:
        _plate_crop_buffers.clear()
        _plate_sahi_call_counts.clear()


def clear_camera_plate_states(camera_id: str) -> None:
    """Remove temporal OCR/crop state for one camera after a scene cut."""
    normalized = str(camera_id or "").strip()
    if not normalized:
        return
    with _aggregator_lock:
        for key in list(_plate_track_states):
            if isinstance(key, tuple) and key and str(key[0]) == normalized:
                _plate_track_states.pop(key, None)
            elif str(key).startswith(f"{normalized}:"):
                _plate_track_states.pop(key, None)
    with _plate_crop_buffer_lock:
        for key in list(_plate_crop_buffers):
            if isinstance(key, tuple) and key and str(key[0]) == normalized:
                _plate_crop_buffers.pop(key, None)
                _plate_sahi_call_counts.pop(key, None)


# ============================================================
# PERSISTENCE THROTTLING
# ============================================================

def _should_persist_plate(
    camera_id: str,
    local_track_id: Optional[str],
    normalized_plate: str,
    confidence: float,
) -> bool:

    if not ANPR_PERSISTENCE_ENABLED:

        return False

    camera_key = str(
        camera_id
    ).strip()

    track_key = (
        str(
            local_track_id
        ).strip()
        if local_track_id is not None
        else "NO_TRACK"
    )

    plate_key = str(
        normalized_plate
    ).strip()

    if (
        not camera_key
        or not plate_key
    ):

        return False

    now = (
        datetime.now(
            timezone.utc
        ).timestamp()
    )

    key = (
        camera_key,
        track_key,
        plate_key,
    )

    with _persistence_lock:

        previous = (
            _last_persisted_plate.get(
                key
            )
        )

        if previous is None:

            _last_persisted_plate[
                key
            ] = (
                now,
                float(
                    confidence
                ),
            )

            return True

        previous_timestamp, previous_confidence = (
            previous
        )

        if (
            now
            - previous_timestamp
            >= ANPR_PERSIST_MIN_INTERVAL_SECONDS
        ):

            _last_persisted_plate[
                key
            ] = (
                now,
                max(
                    previous_confidence,
                    float(
                        confidence
                    ),
                ),
            )

            return True

        if (
            float(
                confidence
            )
            >= previous_confidence
            + ANPR_PERSIST_CONFIDENCE_DELTA
        ):

            _last_persisted_plate[
                key
            ] = (
                now,
                float(
                    confidence
                ),
            )

            return True

        return False


# ============================================================
# PERSISTENCE
# ============================================================

try:

    from services.intelligence_persistence import (
        save_plate_observation,
    )

except Exception:

    save_plate_observation = None


def persist_plate_observation(
    *,
    camera_id: str,
    plate_text: str,
    confidence: float,
    global_vehicle_id: Optional[str] = None,
    local_track_id: Optional[str] = None,
    frame_timestamp: Optional[datetime] = None,
    detection_confidence: Optional[float] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> Any:

    normalized = normalize_plate_text(
        plate_text
    )

    if not normalized:

        return None

    if save_plate_observation is None:

        logger.warning(
            "ANPR persistence unavailable"
        )

        return None

    confidence = _safe_confidence(
        confidence,
        0.0,
    )

    if not _should_persist_plate(
        camera_id,
        local_track_id,
        normalized,
        confidence,
    ):

        return None

    evidence = dict(
        metadata
        or {}
    )

    evidence.update(
        {
            "normalized_plate":
                normalized,

            "plate_detection_confidence":
                (
                    _safe_confidence(
                        detection_confidence,
                        0.0,
                    )
                    if detection_confidence
                    is not None
                    else None
                ),

            "detector":
                ANPR_DETECTOR_NAME,

            "recognizer":
                ANPR_RECOGNIZER_NAME,

            "anpr_model_version":
                (
                    model_version
                    or ANPR_MODEL_VERSION
                ),

            "persistence_policy":
                "throttled_temporal",
        }
    )

    try:

        return save_plate_observation(
            camera_id=str(
                camera_id
            ),

            plate_text=str(
                normalized
            ),

            confidence=confidence,

            global_vehicle_id=(
                str(
                    global_vehicle_id
                )
                if global_vehicle_id
                is not None
                else None
            ),

            local_track_id=(
                str(
                    local_track_id
                )
                if local_track_id
                is not None
                else None
            ),

            frame_timestamp=(
                frame_timestamp
                or datetime.now(
                    timezone.utc
                )
            ),

            model_name=(
                model_name
                or (
                    ANPR_DETECTOR_NAME
                    + "+"
                    + ANPR_RECOGNIZER_NAME
                )
            ),

            model_version=(
                model_version
                or ANPR_MODEL_VERSION
            ),

            metadata=evidence,
        )

    except TypeError:

        # Compatibility with older persistence signatures.
        try:

            return save_plate_observation(
                camera_id=str(
                    camera_id
                ),

                plate_text=str(
                    normalized
                ),

                confidence=confidence,

                global_vehicle_id=(
                    global_vehicle_id
                ),

                local_track_id=(
                    local_track_id
                ),

                frame_timestamp=(
                    frame_timestamp
                    or datetime.now(
                        timezone.utc
                    )
                ),
            )

        except Exception:

            logger.exception(
                "Failed to persist ANPR observation"
            )

            return None

    except Exception:

        logger.exception(
            "Failed to persist ANPR observation"
        )

        return None


def clear_plate_persistence_state() -> None:

    with _persistence_lock:

        _last_persisted_plate.clear()


# ============================================================
# MAIN ANPR API
# ============================================================

def detect_and_read_plate(
    vehicle_crop: Optional[np.ndarray],
    *,
    camera_id: Optional[str] = None,
    local_track_id: Optional[str] = None,
    global_vehicle_id: Optional[str] = None,
    frame_timestamp: Optional[datetime] = None,
    persist: Optional[bool] = None,
    metadata: Optional[dict[str, Any]] = None,
    vehicle_class: Optional[str] = None,
    vehicle_confidence: Optional[float] = None,
    source_frame: Optional[np.ndarray] = None,
    vehicle_bbox: Any = None,
) -> dict[str, Any]:

    empty_result = {

        "plate":
            None,

        "plate_confidence":
            0.0,

        "plate_detection_confidence":
            0.0,

        "raw_plate":
            None,

        "raw_plate_confidence":
            0.0,

        "plate_status":
            "NO_EVIDENCE",

        "plate_evidence_class":
            "NONE",

        "plate_support_count":
            0,

        "plate_observation_count":
            0,

        "plate_consensus":
            0.0,

        "plate_margin":
            0.0,

        "plate_quality":
            0.0,

        "plate_format_score":
            0.0,

        "partial_plate":
            None,

        "anpr_status":
            "UNREADABLE",

        "vehicle_class":
            None,

        "vehicle_confidence":
            _safe_confidence(vehicle_confidence, 0.0) if vehicle_confidence is not None else None,

        "expanded_vehicle_bbox":
            None,

        "persistence":
            {
                "saved":
                    False,

                "observation_id":
                    None,
            },
    }

    metadata = dict(metadata or {})
    resolved_vehicle_class = normalize_vehicle_class(
        vehicle_class
        or metadata.get("vehicle_class")
        or metadata.get("class_name")
        or metadata.get("vehicle_type")
    )
    if vehicle_confidence is None:
        raw_vehicle_confidence = metadata.get("vehicle_confidence")
        if raw_vehicle_confidence is not None:
            vehicle_confidence = _safe_confidence(raw_vehicle_confidence, 0.0)

    empty_result["vehicle_class"] = resolved_vehicle_class
    empty_result["vehicle_confidence"] = (
        _safe_confidence(vehicle_confidence, 0.0)
        if vehicle_confidence is not None
        else None
    )

    if not is_anpr_supported_vehicle(resolved_vehicle_class):
        empty_result["plate_status"] = "SKIPPED_UNSUPPORTED_VEHICLE"
        empty_result["anpr_status"] = "UNREADABLE"
        return empty_result

    working_crop = vehicle_crop
    expanded_bbox: Optional[list[int]] = None
    expanded_crop, expanded_bbox = _expanded_vehicle_crop_from_frame(
        source_frame,
        vehicle_bbox,
        resolved_vehicle_class,
    )
    if expanded_crop is not None:
        working_crop = expanded_crop
        empty_result["expanded_vehicle_bbox"] = expanded_bbox

    if not _validate_vehicle_crop(working_crop):
        return empty_result

    track_key = (str(camera_id or "unknown"), str(local_track_id or "unknown"))
    with _plate_crop_buffer_lock:
        call_count = _plate_sahi_call_counts.get(track_key, 0) + 1
        _plate_sahi_call_counts[track_key] = call_count
    _plate_detector_started=time.perf_counter()
    try:
        candidates = detect_plate_crops(
            working_crop,
            use_sahi=(call_count == 1 or call_count % ANPR_SAHI_FRAME_INTERVAL == 0),
        )
        if MODEL_METRICS is not None:
            MODEL_METRICS.record_call("anpr_detector",latency_ms=(time.perf_counter()-_plate_detector_started)*1000.0)
    except Exception:
        if MODEL_METRICS is not None:
            MODEL_METRICS.record_call("anpr_detector",latency_ms=(time.perf_counter()-_plate_detector_started)*1000.0,error=True)
        raise

    now_ts = _utc_timestamp(frame_timestamp)
    with _plate_crop_buffer_lock:
        buffer = _plate_crop_buffers.setdefault(track_key, deque(maxlen=ANPR_CROP_BUFFER_SIZE))
        while buffer and now_ts - float(buffer[0]["timestamp"]) > ANPR_CROP_BUFFER_TTL_SECONDS:
            buffer.popleft()
        for crop, confidence, bbox in candidates:
            crop_h, crop_w = crop.shape[:2]
            buffer.append({
                "crop": np.ascontiguousarray(crop.copy()),
                "detection_confidence": _safe_confidence(confidence, 0.0),
                "bbox": list(bbox[:4]),
                "quality": estimate_plate_quality(crop),
                "plate_width": int(crop_w),
                "plate_height": int(crop_h),
                "timestamp": now_ts,
            })
        selected_candidates = sorted(
            list(buffer),
            key=lambda item: float(item["quality"]) * 0.65 + float(item["detection_confidence"]) * 0.35,
            reverse=True,
        )[:ANPR_OCR_TOP_K_CROPS]

    if not selected_candidates:
        _anpr_log(
            f"NO_PLATE_DETECTED camera={camera_id or 'unknown'} "
            f"track={local_track_id or 'unknown'}"
        )
        return empty_result

    best_candidate = None
    best_score = -1.0

    for buffered in selected_candidates:
        plate_crop = buffered["crop"]
        detection_confidence = buffered["detection_confidence"]
        plate_bbox = buffered["bbox"]

        quality = float(buffered["quality"])

        _ocr_started=time.perf_counter()
        try:
            text, ocr_confidence = read_plate(plate_crop)
            if MODEL_METRICS is not None:
                MODEL_METRICS.record_call("ocr",latency_ms=(time.perf_counter()-_ocr_started)*1000.0)
        except Exception:
            if MODEL_METRICS is not None:
                MODEL_METRICS.record_call("ocr",latency_ms=(time.perf_counter()-_ocr_started)*1000.0,error=True)
            raise

        if text:

            _anpr_log(
                f"OCR_RESULT "
                f"text_redacted=true "
                f"confidence={ocr_confidence:.3f} "
                f"detection_confidence={detection_confidence:.3f} "
                f"quality={quality:.3f}"
            )

        else:

            _anpr_warning(
                f"OCR_FAILED "
                f"detection_confidence={detection_confidence:.3f} "
                f"quality={quality:.3f}"
            )

            continue

        # OCR text from an extremely poor crop is retained only as weak
        # evidence. This prevents tiny/blurred plates from becoming stable.
        if quality < 0.20:
            continue

        detection_confidence = (
            _safe_confidence(
                detection_confidence,
                0.0,
            )
        )

        ocr_confidence = (
            _safe_confidence(
                ocr_confidence,
                0.0,
            )
        )

        combined_score = (
            (
                0.65
                * ocr_confidence
            )
            + (
                0.25
                * detection_confidence
            )
            + (
                0.10
                * quality
            )
        )

        if (
            math.isfinite(
                combined_score
            )
            and combined_score
            > best_score
        ):

            best_score = (
                combined_score
            )

            best_candidate = {

                "plate":
                    text,

                "ocr_confidence":
                    ocr_confidence,

                "detection_confidence":
                    detection_confidence,

                "quality":
                    quality,

                "bbox":
                    plate_bbox,

                "plate_width":
                    int(buffered.get("plate_width", plate_crop.shape[1])),

                "plate_height":
                    int(buffered.get("plate_height", plate_crop.shape[0])),

                # Keep the actual best plate crop in-memory only so the
                # secondary visual verifier receives plate pixels, never the
                # complete CCTV frame. It is not returned/persisted as JSON.
                "crop":
                    plate_crop,
            }

    # A crop may contribute OCR evidence only once. Without this removal, the
    # same sharp frame could be replayed on later calls and falsely appear as
    # multi-frame support.
    selected_ids = {id(item) for item in selected_candidates}
    with _plate_crop_buffer_lock:
        current_buffer = _plate_crop_buffers.get(track_key)
        if current_buffer is not None:
            _plate_crop_buffers[track_key] = deque(
                (item for item in current_buffer if id(item) not in selected_ids),
                maxlen=ANPR_CROP_BUFFER_SIZE,
            )

    if best_candidate is None:

        return empty_result

    raw_plate = (
        best_candidate[
            "plate"
        ]
    )

    raw_ocr_confidence = (
        best_candidate[
            "ocr_confidence"
        ]
    )

    detection_confidence = (
        best_candidate[
            "detection_confidence"
        ]
    )

    quality = (
        best_candidate[
            "quality"
        ]
    )

    consensus = (
        update_plate_evidence(
            camera_id=camera_id,

            local_track_id=local_track_id,

            plate=raw_plate,

            ocr_confidence=(
                raw_ocr_confidence
            ),

            detection_confidence=(
                detection_confidence
            ),

            quality_score=quality,

            frame_timestamp=(
                frame_timestamp
            ),
        )
    )

    _anpr_log(
        f"PLATE_EVIDENCE "
        f"plate_present={bool(raw_plate)} "
        f"consensus_present={bool(consensus.get('plate'))} "
        f"confidence={_safe_confidence(consensus.get('confidence', 0.0), 0.0):.3f} "
        f"support={int(consensus.get('support_count', 0))} "
        f"observations={int(consensus.get('observation_count', 0))} "
        f"status={str(consensus.get('status', 'UNKNOWN')).upper()}"
    )

    consensus_plate = (
        consensus.get(
            "plate"
        )
    )

    consensus_confidence = (
        _safe_confidence(
            consensus.get(
                "confidence",
                0.0,
            ),
            0.0,
        )
    )

    status = str(
        consensus.get(
            "status",
            "PROVISIONAL",
        )
    ).upper()

    # --------------------------------------------------------
    # SOURCE-AWARE OCR / TEMPORAL / GEMMA-4 FUSION
    # --------------------------------------------------------
    # This is intentionally inside the canonical ANPR service so the fused
    # plate is available BEFORE observation validation and cross-camera
    # correlation. Ollama/Gemma remains advisory and asynchronous.
    fusion_candidates: list[dict[str, Any]] = []
    vote_candidates: list[OCRCandidate] = []

    if raw_plate:
        fusion_candidates.append({"text": raw_plate, "confidence": raw_ocr_confidence, "source": "ppocr"})
        vote_candidates.append(OCRCandidate(raw_plate, raw_ocr_confidence, "ppocr"))

    if consensus_plate:
        fusion_candidates.append({"text": consensus_plate, "confidence": consensus_confidence, "source": "temporal"})
        vote_candidates.append(OCRCandidate(consensus_plate, consensus_confidence, "temporal"))

    voted_plate, voted_confidence = (
        character_vote(vote_candidates)
        if ANPR_CHARACTER_VOTING_ENABLED and vote_candidates
        else ("", 0.0)
    )
    if voted_plate:
        fusion_candidates.append({"text": voted_plate, "confidence": voted_confidence, "source": "candidate"})

    verifier_key = f"{camera_id or 'unknown'}:{local_track_id or 'unknown'}"
    ollama_candidate = ollama_plate_verifier.get_result(verifier_key)
    if isinstance(ollama_candidate, dict) and ollama_candidate.get("text"):
        fusion_candidates.append({
            "text": ollama_candidate.get("text"),
            "confidence": _safe_confidence(ollama_candidate.get("confidence", 0.0), 0.0),
            "source": "ollama",
        })

    primary_confidence = max(raw_ocr_confidence, consensus_confidence)
    if (status != "STABLE" or primary_confidence < PLATE_STABLE_THRESHOLD) and ollama_plate_verifier.should_verify(primary_confidence, uncertain=(status != "STABLE")):
        try:
            ollama_plate_verifier.submit(
                verifier_key,
                best_candidate.get("crop"),
                current_candidates=fusion_candidates[:8],
            )
        except Exception:
            logger.debug("Gemma plate verification submit failed | camera=%s track=%s", camera_id, local_track_id, exc_info=True)

    fused = fuse_plate_candidates(fusion_candidates)

    # Strong deterministic temporal evidence remains authoritative. A Gemma
    # result cannot downgrade or replace a stable deterministic plate unless
    # source-aware fusion also has deterministic support.
    if consensus_plate and status == "STABLE":
        final_plate = consensus_plate
    elif fused.get("plate") and fused.get("decision") in {"CONFIRMED", "PROBABLE"}:
        final_plate = str(fused["plate"])
        status = "STABLE" if fused.get("decision") == "CONFIRMED" else "PROVISIONAL"
    elif consensus_plate and status == "PROVISIONAL":
        final_plate = consensus_plate
    else:
        final_plate = raw_plate

    support_count = int(consensus.get("support_count", 0))
    observation_count = int(consensus.get("observation_count", 0))
    temporal_consistency = _safe_confidence(consensus.get("consensus", 0.0), 0.0)
    partial_plate = consensus.get("partial_plate")
    format_score = _indian_plate_format_score(final_plate)
    plate_width = int(best_candidate.get("plate_width", 0) or 0)
    plate_height = int(best_candidate.get("plate_height", 0) or 0)
    best_ocr_component = max(
        raw_ocr_confidence,
        _safe_confidence(consensus.get("best_ocr_confidence", 0.0), 0.0),
        _safe_confidence(voted_confidence, 0.0),
    )

    final_confidence = _end_to_end_anpr_confidence(
        ocr_confidence=best_ocr_component,
        detection_confidence=detection_confidence,
        quality_score=quality,
        temporal_consistency=temporal_consistency,
        format_score=format_score,
        vehicle_confidence=vehicle_confidence,
        plate_width=plate_width,
        plate_height=plate_height,
    )

    readable_plate = (
        plate_width >= ANPR_MIN_READABLE_PLATE_WIDTH
        and plate_height >= ANPR_MIN_READABLE_PLATE_HEIGHT
    )
    confirmed = (
        bool(final_plate)
        and status == "STABLE"
        and support_count >= max(3, ANPR_AGGREGATOR_MIN_SUPPORT)
        and observation_count >= max(3, ANPR_AGGREGATOR_MIN_OBSERVATIONS)
        and temporal_consistency >= ANPR_CONFIRMED_MIN_TEMPORAL
        and final_confidence >= PLATE_STABLE_THRESHOLD
        and format_score >= ANPR_CONFIRMED_MIN_FORMAT_SCORE
        and readable_plate
    )

    if confirmed:
        anpr_status = "CONFIRMED"
        status = "STABLE"
    elif partial_plate and "?" in str(partial_plate):
        anpr_status = "PARTIAL"
        if status == "STABLE":
            status = "PROVISIONAL"
    elif final_plate and final_confidence >= PLATE_PROVISIONAL_THRESHOLD:
        anpr_status = "PROVISIONAL"
        if status == "STABLE":
            status = "PROVISIONAL"
    else:
        anpr_status = "UNREADABLE"
        if status == "STABLE":
            status = "LOW_CONFIDENCE"

    _anpr_log(
        f"PLATE_DECISION "
        f"plate_present={bool(final_plate)} "
        f"confidence={final_confidence:.3f} "
        f"status={status} "
        f"anpr_status={anpr_status} "
        f"raw_present={bool(raw_plate)} "
        f"support={support_count} "
        f"format={format_score:.2f} "
        f"plate_size={plate_width}x{plate_height}"
    )

    result = {

        "plate":
            final_plate,

        "plate_confidence":
            final_confidence,

        "plate_detection_confidence":
            detection_confidence,

        "raw_plate":
            raw_plate,

        "raw_plate_confidence":
            raw_ocr_confidence,

        "plate_status":
            status,

        "plate_evidence_class": (
            "STABLE"
            if status == "STABLE"
            else "PROVISIONAL"
            if status == "PROVISIONAL"
            else "WEAK"
        ),

        "plate_support_count":
            int(
                consensus.get(
                    "support_count",
                    0,
                )
            ),

        "plate_observation_count":
            int(
                consensus.get(
                    "observation_count",
                    0,
                )
            ),

        "plate_consensus":
            _safe_confidence(
                consensus.get(
                    "consensus",
                    0.0,
                ),
                0.0,
            ),

        "plate_margin":
            _safe_confidence(
                consensus.get(
                    "margin",
                    0.0,
                ),
                0.0,
            ),

        "plate_quality":
            quality,

        "plate_format_score":
            format_score,

        "partial_plate":
            partial_plate,

        "anpr_status":
            anpr_status,

        "vehicle_class":
            resolved_vehicle_class,

        "vehicle_confidence":
            _safe_confidence(vehicle_confidence, 0.0) if vehicle_confidence is not None else None,

        "plate_width":
            plate_width,

        "plate_height":
            plate_height,

        "readable_plate":
            readable_plate,

        "expanded_vehicle_bbox":
            expanded_bbox,

        "plate_bbox":
            best_candidate.get(
                "bbox"
            ),

        "detector":
            ANPR_DETECTOR_NAME,

        "recognizer":
            ANPR_RECOGNIZER_NAME,

        "model_version":
            ANPR_MODEL_VERSION,

        "character_voted_plate": voted_plate or None,

        "character_vote_confidence": voted_confidence,

        "plate_fusion": fused,

        "ollama_plate_verification": ({
            key: value for key, value in ollama_candidate.items() if key != "received_at"
        } if isinstance(ollama_candidate, dict) else None),

        "ocr_threshold":
            PLATE_OCR_CONF,

        "detection_threshold":
            PLATE_DETECTION_CONF,

        "persistence":
            {
                "saved":
                    False,

                "observation_id":
                    None,
            },
    }

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    should_persist = (
        ANPR_PERSISTENCE_ENABLED
        if persist is None
        else bool(
            persist
        )
    )

    # Persist only sufficiently strong temporal evidence by default.
    # Raw/provisional OCR remains available to correlation but should not
    # become durable identity evidence unless explicitly requested.
    persist_allowed = anpr_status == "CONFIRMED" and status == "STABLE"

    if (
        should_persist
        and persist_allowed
        and camera_id
        and final_plate
    ):

        persistence_metadata = dict(
            metadata
            or {}
        )

        persistence_metadata.update(
            {
                "raw_plate":
                    raw_plate,

                "raw_plate_confidence":
                    raw_ocr_confidence,

                "plate_detection_confidence":
                    detection_confidence,

                "plate_quality":
                    quality,

                "plate_status":
                    status,

                "anpr_status":
                    anpr_status,

                "partial_plate":
                    partial_plate,

                "plate_format_score":
                    format_score,

                "plate_width":
                    plate_width,

                "plate_height":
                    plate_height,

                "vehicle_class":
                    resolved_vehicle_class,

                "plate_support_count":
                    consensus.get(
                        "support_count",
                        0,
                    ),

                "plate_observation_count":
                    consensus.get(
                        "observation_count",
                        0,
                    ),

                "plate_consensus":
                    consensus.get(
                        "consensus",
                        0.0,
                    ),

                "plate_margin":
                    consensus.get(
                        "margin",
                        0.0,
                    ),

                "plate_bbox":
                    best_candidate.get(
                        "bbox"
                    ),

                "anpr_pipeline": (
                    "YOLO-Plate+SAHI+Awiros-ANPR-OCR+TemporalConsensus"
                    if OCR_ENGINE == "awiros_paddle"
                    else "YOLO-Plate+SAHI+PP-OCRv5-ONNX+TemporalConsensus"
                ),
            }
        )

        record = (
            persist_plate_observation(
                camera_id=(
                    camera_id
                ),

                plate_text=(
                    final_plate
                ),

                confidence=(
                    final_confidence
                ),

                global_vehicle_id=(
                    global_vehicle_id
                ),

                local_track_id=(
                    local_track_id
                ),

                frame_timestamp=(
                    frame_timestamp
                ),

                detection_confidence=(
                    detection_confidence
                ),

                model_name=(
                    ANPR_DETECTOR_NAME
                    + "+"
                    + ANPR_RECOGNIZER_NAME
                ),

                model_version=(
                    ANPR_MODEL_VERSION
                ),

                metadata=(
                    persistence_metadata
                ),
            )
        )

        if record is not None:

            _anpr_log(
                f"ANPR_PERSISTED "
                f"plate_present={bool(final_plate)} "
                f"confidence={final_confidence:.3f} "
                f"camera={camera_id} "
                f"track={local_track_id or 'unknown'} "
                f"global_vehicle={global_vehicle_id or 'unknown'} "
                f"observation_id={getattr(record, 'id', None)}"
            )

            result[
                "persistence"
            ] = {

                "saved":
                    True,

                "observation_id":
                    getattr(
                        record,
                        "id",
                        None,
                    ),
            }

    return result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def process_vehicle_anpr(
    vehicle_crop: np.ndarray,
    **kwargs,
) -> dict[str, Any]:

    return detect_and_read_plate(
        vehicle_crop,
        **kwargs,
    )


def run_anpr(
    vehicle_crop: np.ndarray,
    **kwargs,
) -> dict[str, Any]:

    return detect_and_read_plate(
        vehicle_crop,
        **kwargs,
    )


def detectANPR(
    vehicle_crop: np.ndarray,
    **kwargs,
) -> dict[str, Any]:

    return detect_and_read_plate(
        vehicle_crop,
        **kwargs,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_plate_model_classes() -> dict:

    model = get_plate_model()

    names = getattr(
        model,
        "names",
        {},
    )

    if isinstance(
        names,
        dict,
    ):

        return dict(
            names
        )

    if isinstance(
        names,
        list,
    ):

        return {
            index: name
            for index, name
            in enumerate(
                names
            )
        }

    return {}


# ============================================================
# HEALTH CHECK
# ============================================================

def anpr_health() -> dict[str, Any]:

    with _aggregator_lock:

        active_track_aggregators = (
            len(
                _plate_track_states
            )
        )

    with _plate_crop_buffer_lock:
        active_crop_buffers = len(_plate_crop_buffers)

    return {

        "component":
            "vehicleANPR",

        "version":
            ANPR_MODEL_VERSION,

        "plate_model":
            PLATE_MODEL_PATH.name,

        "plate_model_exists":
            PLATE_MODEL_PATH.exists(),

        "plate_model_loaded":
            _plate_model is not None,

        "sahi_enabled": ANPR_SAHI_ENABLED,
        "sahi_required": ANPR_SAHI_REQUIRED,
        "sahi_loaded": _sahi_plate_model is not None,
        "sahi_slice_size": [ANPR_SAHI_SLICE_WIDTH, ANPR_SAHI_SLICE_HEIGHT],
        "sahi_overlap": [ANPR_SAHI_OVERLAP_WIDTH, ANPR_SAHI_OVERLAP_HEIGHT],
        "sahi_frame_interval": ANPR_SAHI_FRAME_INTERVAL,
        "detect_image_size": ANPR_DETECT_IMAGE_SIZE,
        "supported_vehicle_classes": sorted(ANPR_SUPPORTED_VEHICLES),
        "two_line_plates": ANPR_TWO_LINE_PLATES,
        "perspective_correction": ANPR_PERSPECTIVE_CORRECTION,
        "partial_results": ANPR_PARTIAL_RESULTS,
        "character_voting": ANPR_CHARACTER_VOTING_ENABLED,
        "vehicle_crop_expansion": ANPR_VEHICLE_CROP_EXPANSION_ENABLED,
        "min_plate_candidate_size": [MIN_PLATE_WIDTH, MIN_PLATE_HEIGHT],
        "min_readable_plate_size": [ANPR_MIN_READABLE_PLATE_WIDTH, ANPR_MIN_READABLE_PLATE_HEIGHT],
        "target_plate_width": ANPR_TARGET_PLATE_WIDTH,
        "confirmed_min_temporal": ANPR_CONFIRMED_MIN_TEMPORAL,
        "confirmed_min_format_score": ANPR_CONFIRMED_MIN_FORMAT_SCORE,
        "active_crop_buffers": active_crop_buffers,
        "ocr_top_k_crops": ANPR_OCR_TOP_K_CROPS,

        "ocr_model":
            OCR_MODEL_NAME,

        "ocr_engine": OCR_ENGINE,

        "ocr_onnx_model":
            OCR_ONNX_MODEL_PATH.name,

        "ocr_dictionary":
            OCR_DICT_PATH.name,

        "ocr_engine_version":
            _ocr_engine_version,

        "ocr_loaded":
            _ocr_engine is not None,

        "ocr_device": getattr(_ocr_engine, "actual_device", OCR_DEVICE),

        "awiros": (
            _ocr_engine.health()
            if OCR_ENGINE == "awiros_paddle" and _ocr_engine is not None
            else {
                "weights_present": AWIROS_MODEL_PATH.exists(),
                "dictionary_present": AWIROS_DICT_PATH.exists(),
                "paddleocr_present": (
                    AWIROS_PADDLEOCR_DIR / "ppocr" / "__init__.py"
                ).is_file(),
            }
        ),

        "console_logs_enabled":
            ANPR_CONSOLE_LOGS,

        "persistence_enabled":
            ANPR_PERSISTENCE_ENABLED,

        "persistence_available":
            save_plate_observation
            is not None,

        "temporal_aggregation_enabled":
            ANPR_ENABLE_TEMPORAL_AGGREGATION,

        "active_track_aggregators":
            active_track_aggregators,

        "aggregator_max_observations":
            ANPR_AGGREGATOR_MAX_OBSERVATIONS,

        "aggregator_min_observations":
            ANPR_AGGREGATOR_MIN_OBSERVATIONS,

        "aggregator_min_support":
            ANPR_AGGREGATOR_MIN_SUPPORT,

        "plate_stable_threshold":
            PLATE_STABLE_THRESHOLD,

        "plate_strong_threshold":
            PLATE_STRONG_THRESHOLD,
    }


# ============================================================
# RESET / CLEANUP
# ============================================================

def cleanup_anpr_state() -> int:

    now = (
        datetime.now(
            timezone.utc
        ).timestamp()
    )

    with _aggregator_lock:

        before = len(
            _plate_track_states
        )

        _cleanup_aggregator_states(
            now
        )

        after = len(
            _plate_track_states
        )

        removed = before - after

    with _plate_crop_buffer_lock:
        expired = [
            key for key, values in _plate_crop_buffers.items()
            if not values or now - float(values[-1]["timestamp"]) > ANPR_CROP_BUFFER_TTL_SECONDS
        ]
        for key in expired:
            _plate_crop_buffers.pop(key, None)
            _plate_sahi_call_counts.pop(key, None)

    return removed + len(expired)


# ============================================================
# EXPLICIT APPLICATION STARTUP HOOK
# ============================================================

def initialize_anpr_ocr() -> bool:
    """
    Initialize the configured ANPR OCR engine once during application startup.

    FastAPI main/lifespan can call this explicitly. The module-level
    prewarm below also supports existing applications that do not
    yet call a startup hook.
    """

    engine = get_ocr_engine()

    if engine is None:
        _anpr_error(
            "OCR_STARTUP_FAILED "
            f"model={OCR_MODEL_NAME} "
            f"device={OCR_DEVICE}"
        )
        return False

    actual_providers = (
        list(engine.get_providers())
        if hasattr(engine, "get_providers")
        else []
    )
    actual_device = getattr(
        engine,
        "actual_device",
        "cuda" if "CUDAExecutionProvider" in actual_providers else "cpu",
    )

    _anpr_log(
        "OCR_STARTUP_READY "
        f"model={OCR_MODEL_NAME} "
        f"device={actual_device} "
        f"providers={actual_providers}"
    )

    return True


def get_ocr_runtime_status() -> dict[str, Any]:
    """Backward-compatible diagnostics for the selected OCR runtime."""

    if OCR_ENGINE == "awiros_paddle":
        details = (
            _ocr_engine.health()
            if _ocr_engine is not None
            else {
                "engine": OCR_ENGINE,
                "model": "Awiros-ANPR-OCR",
                "loaded": False,
                "requested_device": OCR_DEVICE,
                "actual_device": "uninitialized",
                "weights_present": AWIROS_MODEL_PATH.exists(),
                "dictionary_present": AWIROS_DICT_PATH.exists(),
                "paddleocr_present": (
                    AWIROS_PADDLEOCR_DIR / "ppocr" / "__init__.py"
                ).is_file(),
            }
        )
        return {
            **details,
            "ocr_loaded": _ocr_engine is not None,
            "ocr_initialization_attempted": _ocr_initialization_attempted,
            "ocr_initialization_failed": _ocr_initialization_failed,
            "ocr_engine_version": _ocr_engine_version,
        }

    return {
        "paddle_import_ok": False,
        "paddle_version": "disabled_for_anpr_ocr",
        "paddle_cuda": False,
        "paddle_device": "disabled_for_anpr_ocr",
        "ocr_model": OCR_MODEL_NAME,
        "ocr_onnx_model": str(OCR_ONNX_MODEL_PATH),
        "ocr_dictionary": str(OCR_DICT_PATH),
        "ocr_device": (
            "cuda"
            if _ocr_engine is not None
            and "CUDAExecutionProvider" in _ocr_engine.get_providers()
            else "cpu"
        ),
        "ocr_requested_device": OCR_DEVICE,
        "ocr_cuda_device_id": OCR_CUDA_DEVICE_ID,
        "ocr_available_providers": (
            list(ort.get_available_providers())
            if ort is not None
            else []
        ),
        "ocr_providers": (
            list(_ocr_engine.get_providers())
            if _ocr_engine is not None
            else []
        ),
        "ocr_runtime": "onnxruntime",
        "ocr_runtime_version": _OCR_RUNTIME_VERSION,
        "ocr_loaded": _ocr_engine is not None,
        "ocr_initialization_attempted": _ocr_initialization_attempted,
        "ocr_initialization_failed": _ocr_initialization_failed,
        "ocr_engine_version": _ocr_engine_version,
    }


def get_paddle_runtime_status() -> dict[str, Any]:
    """Backward-compatible alias for older callers."""
    return get_ocr_runtime_status()


# ============================================================
# MODULE-LEVEL OCR PREWARM
# ============================================================

if ANPR_OCR_PREWARM:
    try:
        initialize_anpr_ocr()
    except Exception:
        logger.exception(
            "ANPR OCR module prewarm failed"
        )


# ============================================================
# STARTUP VALIDATION
# ============================================================

def validate_anpr_configuration() -> dict[str, Any]:

    errors = []

    if (
        PLATE_DETECTION_CONF
        <= 0.0
        or PLATE_DETECTION_CONF
        > 1.0
    ):

        errors.append(
            "Invalid plate detection confidence"
        )

    if (
        PLATE_IOU
        <= 0.0
        or PLATE_IOU
        > 1.0
    ):

        errors.append(
            "Invalid plate IoU"
        )

    if (
        PLATE_OCR_CONF
        <= 0.0
        or PLATE_OCR_CONF
        > 1.0
    ):

        errors.append(
            "Invalid OCR confidence"
        )

    if PLATE_PROVISIONAL_THRESHOLD > PLATE_STABLE_THRESHOLD:
        errors.append(
            "Provisional plate threshold exceeds stable threshold"
        )

    if PLATE_STABLE_THRESHOLD > PLATE_STRONG_THRESHOLD:
        errors.append(
            "Stable plate threshold exceeds strong threshold"
        )

    if not ANPR_SUPPORTED_VEHICLES:
        errors.append("ANPR_VEHICLE_CLASSES must contain at least one vehicle class")

    if ANPR_DETECT_IMAGE_SIZE < 320:
        errors.append("ANPR_DETECT_IMAGE_SIZE is too small")

    if ANPR_MIN_READABLE_PLATE_WIDTH < MIN_PLATE_WIDTH:
        errors.append("Readable plate width cannot be smaller than candidate width")

    if ANPR_MIN_READABLE_PLATE_HEIGHT < MIN_PLATE_HEIGHT:
        errors.append("Readable plate height cannot be smaller than candidate height")

    if (
        MIN_PLATE_LENGTH
        > MAX_PLATE_LENGTH
    ):

        errors.append(
            "Invalid plate length range"
        )

    if (
        MIN_PLATE_WIDTH
        > MAX_PLATE_WIDTH
    ):

        errors.append(
            "Invalid plate width range"
        )

    if (
        MIN_PLATE_HEIGHT
        > MAX_PLATE_HEIGHT
    ):

        errors.append(
            "Invalid plate height range"
        )

    if not PLATE_MODEL_PATH.exists():

        errors.append(
            "Plate model file does not exist"
        )

    if OCR_ENGINE == "awiros_paddle":
        if ANPR_OCR_REMOTE_ENABLED:
            if not ANPR_OCR_REMOTE_URL:
                errors.append("ANPR_OCR_REMOTE_URL is required for remote Awiros OCR")
            if len(ANPR_OCR_INTERNAL_TOKEN) < 32:
                errors.append("ANPR_OCR_INTERNAL_TOKEN must be at least 32 characters")
        else:
            if not AWIROS_MODEL_PATH.exists():
                errors.append("Awiros model.safetensors file does not exist")
            if not AWIROS_DICT_PATH.exists():
                errors.append("Awiros en_dict.txt file does not exist")
            if not (AWIROS_PADDLEOCR_DIR / "ppocr" / "__init__.py").is_file():
                errors.append("Reviewed PaddleOCR source tree does not exist")
    else:
        if not OCR_ONNX_MODEL_PATH.exists():
            errors.append("PP-OCRv5 ONNX model file does not exist")
        if not OCR_DICT_PATH.exists():
            errors.append("PP-OCRv5 English dictionary file does not exist")

    return {

        "valid":
            len(errors) == 0,

        "errors":
            errors,

        "plate_model":
            str(
                PLATE_MODEL_PATH
            ),

        "ocr_model":
            OCR_MODEL_NAME,

        "ocr_engine": OCR_ENGINE,

        "awiros_model": str(AWIROS_MODEL_PATH),

        "awiros_dictionary": str(AWIROS_DICT_PATH),

        "awiros_paddleocr_dir": str(AWIROS_PADDLEOCR_DIR),

        "ocr_remote_enabled": ANPR_OCR_REMOTE_ENABLED,

        "ocr_remote_url": ANPR_OCR_REMOTE_URL if ANPR_OCR_REMOTE_ENABLED else None,

        "ocr_onnx_model":
            str(OCR_ONNX_MODEL_PATH),

        "ocr_dictionary":
            str(OCR_DICT_PATH),

        "ocr_engine_version":
            _ocr_engine_version,

        "version":
            ANPR_MODEL_VERSION,
    }


# ============================================================
# MODULE STARTUP
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    validation = (
        validate_anpr_configuration()
    )

    logger.info(
        "INTEL-I vehicleANPR "
        "configuration: %s",
        validation,
    )
