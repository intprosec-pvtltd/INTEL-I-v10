from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def _env_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    raw = os.getenv(name, str(default)).strip()

    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid numeric configuration for %s; using default.",
            name,
        )
        value = float(default)

    if not math.isfinite(value):
        value = float(default)

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


def _env_int(
    name: str,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    raw = os.getenv(name, str(default)).strip()

    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer configuration for %s; using default.",
            name,
        )
        value = int(default)

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


# ============================================================
# FACE MODEL CONFIGURATION
# ============================================================

DETECTOR = os.getenv(
    "FACE_DETECTOR_MODEL_PATH",
    "",
).strip()

RECOGNIZER = os.getenv(
    "FACE_RECOGNIZER_MODEL_PATH",
    "",
).strip()

VERSION = (
    os.getenv(
        "FACE_RECOGNITION_MODEL_VERSION",
        "1.0",
    ).strip()
    or "1.0"
)

THRESH = _env_float(
    "PERSON_FACE_MATCH_THRESHOLD",
    0.45,
    minimum=-1.0,
    maximum=1.0,
)

MIN_SIZE = _env_int(
    "PERSON_MIN_FACE_SIZE",
    50,
    minimum=16,
    maximum=4096,
)

MIN_DET_SCORE = _env_float(
    "PERSON_FACE_MIN_DET_SCORE",
    0.70,
    minimum=0.1,
    maximum=0.99,
)

MIN_BLUR_SCORE = _env_float(
    "PERSON_FACE_MIN_BLUR_SCORE",
    _env_float(
        # Backward-compatible alias for deployments created before the
        # canonical variable name was introduced.
        "PERSON_FACE_MIN_BLUR",
        35.0,
        minimum=0.0,
    ),
    minimum=0.0,
)

# Reference photographs are manually enrolled images. They may be taken in
# low light or have a modest head turn, so their detection gate is intentionally
# separate from the stricter live-camera gate above. This does not lower the
# watchlist match threshold.
REFERENCE_MIN_SIZE = _env_int(
    "PERSON_REFERENCE_MIN_FACE_SIZE",
    40,
    minimum=16,
    maximum=4096,
)

REFERENCE_MIN_DET_SCORE = _env_float(
    "PERSON_REFERENCE_MIN_DET_SCORE",
    0.45,
    minimum=0.1,
    maximum=0.99,
)

REFERENCE_MIN_BLUR_SCORE = _env_float(
    "PERSON_REFERENCE_MIN_BLUR",
    12.0,
    minimum=0.0,
)

REFERENCE_LOW_LIGHT_RETRY_ENABLED = os.getenv(
    "PERSON_REFERENCE_LOW_LIGHT_RETRY_ENABLED",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}

LIVE_LOW_LIGHT_RETRY_ENABLED = os.getenv(
    "PERSON_FACE_LOW_LIGHT_RETRY_ENABLED",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}

# Final live-camera recovery gate. It is used only when normal YuNet and the
# standard low-light retry both find no face. The resulting embedding must
# still pass PERSON_FACE_MATCH_THRESHOLD before a watchlist alert is possible.
LIVE_FALLBACK_MIN_SIZE = _env_int(
    "PERSON_FACE_FALLBACK_MIN_SIZE",
    32,
    minimum=16,
    maximum=4096,
)

LIVE_FALLBACK_MIN_DET_SCORE = _env_float(
    "PERSON_FACE_FALLBACK_MIN_DET_SCORE",
    0.25,
    minimum=0.1,
    maximum=0.99,
)

LIVE_FALLBACK_MIN_BLUR_SCORE = _env_float(
    "PERSON_FACE_FALLBACK_MIN_BLUR",
    5.0,
    minimum=0.0,
)

FACE_NMS_THRESHOLD = _env_float(
    "PERSON_FACE_NMS_THRESHOLD",
    0.30,
    minimum=0.05,
    maximum=0.95,
)

FACE_TOP_K = _env_int(
    "PERSON_FACE_TOP_K",
    5000,
    minimum=1,
    maximum=10000,
)

MAX_FRAME_PIXELS = _env_int(
    "PERSON_FACE_MAX_FRAME_PIXELS",
    20_000_000,
    minimum=100_000,
    maximum=100_000_000,
)


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

_det = None
_rec = None


# Model construction must not happen simultaneously.
_model_lock = threading.RLock()

# YuNet owns mutable OpenCV DNN state.
_detector_lock = threading.RLock()

# SFace also owns native OpenCV DNN state.
_recognizer_lock = threading.RLock()


# ============================================================
# EMBEDDING ENCRYPTION
# ============================================================

def _key() -> bytes:
    raw = os.getenv(
        "PERSON_EMBEDDING_KEY",
        "",
    ).strip()

    if not raw:
        raise RuntimeError(
            "PERSON_EMBEDDING_KEY is required"
        )

    try:
        key = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "PERSON_EMBEDDING_KEY must be an ASCII Fernet key"
        ) from exc

    try:
        Fernet(key)
    except Exception as exc:
        raise RuntimeError(
            "PERSON_EMBEDDING_KEY is not a valid Fernet key"
        ) from exc

    return key


def encrypt_embedding(vector: Any) -> bytes:
    array = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            "Cannot encrypt an empty face embedding"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Cannot encrypt a non-finite face embedding"
        )

    return Fernet(_key()).encrypt(
        array.tobytes()
    )


def decrypt_embedding(
    blob: bytes,
    dim: int,
) -> np.ndarray:
    if not blob:
        raise RuntimeError(
            "Encrypted person embedding is empty"
        )

    try:
        decrypted = Fernet(_key()).decrypt(blob)
    except InvalidToken as exc:
        raise RuntimeError(
            "Unable to decrypt person embedding"
        ) from exc

    vector = np.frombuffer(
        decrypted,
        dtype=np.float32,
    ).copy()

    expected_dim = int(dim)

    if expected_dim <= 0:
        raise RuntimeError(
            "Invalid person embedding dimension"
        )

    if vector.size != expected_dim:
        raise RuntimeError(
            "Person embedding dimension mismatch"
        )

    if not np.all(np.isfinite(vector)):
        raise RuntimeError(
            "Person embedding contains invalid values"
        )

    norm = float(np.linalg.norm(vector))

    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError(
            "Invalid person embedding"
        )

    return (
        vector / norm
    ).astype(
        np.float32,
        copy=False,
    )


def encrypt_bytes(data: bytes) -> bytes:
    if not data:
        raise ValueError(
            "Cannot encrypt empty binary data"
        )

    return Fernet(_key()).encrypt(data)


def decrypt_bytes(blob: bytes) -> bytes:
    if not blob:
        raise RuntimeError(
            "Encrypted binary data is empty"
        )

    try:
        return Fernet(_key()).decrypt(blob)
    except InvalidToken as exc:
        raise RuntimeError(
            "Unable to decrypt protected binary data"
        ) from exc


# ============================================================
# MODEL PATH VALIDATION
# ============================================================

def _resolve_model_path(
    value: str,
    label: str,
) -> Path:
    if not value:
        raise RuntimeError(
            f"{label} model path is not configured"
        )

    path = Path(value).expanduser()

    if not path.is_file():
        raise RuntimeError(
            f"{label} model not found: {path}"
        )

    return path


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def _models():
    """
    Lazily initialize YuNet and SFace.

    Model construction is protected because OpenCV DNN model
    initialization must not race between camera workers.
    """

    global _det, _rec

    if _det is not None and _rec is not None:
        return _det, _rec

    with _model_lock:
        if _det is not None and _rec is not None:
            return _det, _rec

        try:
            detector_path = _resolve_model_path(
                DETECTOR,
                "Face detector",
            )

            recognizer_path = _resolve_model_path(
                RECOGNIZER,
                "Face recognizer",
            )
        except RuntimeError:
            logger.exception(
                "Face-recognition model configuration is invalid"
            )
            raise

        try:
            detector = cv2.FaceDetectorYN.create(
                str(detector_path),
                "",
                (320, 320),
                MIN_DET_SCORE,
                FACE_NMS_THRESHOLD,
                FACE_TOP_K,
            )
        except cv2.error as exc:
            logger.exception(
                "Unable to initialize YuNet face detector | path=%s",
                detector_path,
            )
            raise RuntimeError(
                "Unable to initialize YuNet face detector"
            ) from exc

        try:
            recognizer = cv2.FaceRecognizerSF.create(
                str(recognizer_path),
                "",
            )
        except cv2.error as exc:
            logger.exception(
                "Unable to initialize SFace face recognizer | path=%s",
                recognizer_path,
            )
            raise RuntimeError(
                "Unable to initialize SFace face recognizer"
            ) from exc

        if detector is None:
            raise RuntimeError(
                "OpenCV returned an empty YuNet detector"
            )

        if recognizer is None:
            raise RuntimeError(
                "OpenCV returned an empty SFace recognizer"
            )

        # Publish only after BOTH models initialized successfully.
        _det = detector
        _rec = recognizer

        logger.info(
            "Face recognition models initialized | "
            "version=%s detector=%s recognizer=%s",
            VERSION,
            detector_path.name,
            recognizer_path.name,
        )

        return _det, _rec


# ============================================================
# FRAME VALIDATION
# ============================================================

def _prepare_frame(
    frame: Any,
) -> Optional[np.ndarray]:
    """
    Convert incoming frames to safe OpenCV BGR uint8 contiguous
    arrays.

    Returns None for invalid frames.
    """

    if frame is None:
        return None

    if not isinstance(frame, np.ndarray):
        return None

    if frame.size == 0:
        return None

    if frame.ndim == 2:
        try:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )
        except cv2.error:
            return None

    elif frame.ndim == 3:
        channels = frame.shape[2]

        if channels == 4:
            try:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGRA2BGR,
                )
            except cv2.error:
                return None

        elif channels != 3:
            return None

    else:
        return None

    height, width = frame.shape[:2]

    if height < 32 or width < 32:
        return None

    if height * width > MAX_FRAME_PIXELS:
        logger.warning(
            "Face recognition frame rejected because it exceeds "
            "the configured pixel limit | width=%d height=%d",
            width,
            height,
        )
        return None

    if frame.dtype != np.uint8:
        try:
            if np.issubdtype(
                frame.dtype,
                np.floating,
            ):
                frame = np.nan_to_num(
                    frame,
                    nan=0.0,
                    posinf=255.0,
                    neginf=0.0,
                )

            frame = np.clip(
                frame,
                0,
                255,
            ).astype(
                np.uint8,
                copy=False,
            )

        except (
            TypeError,
            ValueError,
            FloatingPointError,
        ):
            return None

    # Important for OpenCV DNN native inference.
    frame = np.ascontiguousarray(frame)

    return frame


# ============================================================
# FACE DETECTION
# ============================================================

def _detect_faces(
    detector: Any,
    frame: np.ndarray,
    minimum_score: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Thread-safe YuNet detection.

    FaceDetectorYN contains mutable native DNN state because
    setInputSize() changes detector state. The same detector must
    therefore not be used concurrently by multiple camera workers.
    """

    height, width = frame.shape[:2]
    score_threshold = (
        MIN_DET_SCORE
        if minimum_score is None
        else float(minimum_score)
    )

    try:
        with _detector_lock:
            detector.setInputSize(
                (int(width), int(height))
            )
            detector.setScoreThreshold(score_threshold)
            try:
                _, faces = detector.detect(frame)
            finally:
                # Do not allow an enrollment request to weaken live workers.
                detector.setScoreThreshold(MIN_DET_SCORE)

            # Detach the result from OpenCV-owned native memory before
            # releasing the detector lock.
            if faces is not None:
                faces = np.asarray(
                    faces,
                    dtype=np.float32,
                ).copy()

    except cv2.error as exc:
        logger.error(
            "YuNet inference failed | "
            "width=%d height=%d dtype=%s contiguous=%s error=%s",
            width,
            height,
            frame.dtype,
            bool(frame.flags.c_contiguous),
            str(exc),
        )

        raise RuntimeError(
            "YuNet face detector inference failed "
            f"for {width}x{height} frame"
        ) from exc

    if faces is None:
        return None

    if faces.ndim != 2:
        logger.warning(
            "YuNet returned unexpected face result shape: %s",
            getattr(faces, "shape", None),
        )
        return None

    return faces


# ============================================================
# FACE QUALITY
# ============================================================

def _face_blur_score(
    face_roi: np.ndarray,
) -> float:
    if face_roi.size == 0:
        return 0.0

    try:
        if face_roi.ndim == 3:
            gray = cv2.cvtColor(
                face_roi,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = face_roi

        score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

    except cv2.error:
        return 0.0

    if not math.isfinite(score):
        return 0.0

    return score


def detect_face_evidence_candidates(
    frame: Any,
    *,
    minimum_face_size: int = 45,
    minimum_detection_score: float = 0.75,
    minimum_blur_score: float = 30.0,
    max_faces: int = 4,
) -> list[dict[str, Any]]:
    """Detect face-quality candidates without extracting biometric embeddings.

    This path is intentionally separate from ``extract_embeddings``.  It is
    used only to select the clearest incident-evidence frame, so running SFace
    recognition would add work without improving evidence selection.
    """
    from inference.triton_client import triton_enabled, face_operation
    if triton_enabled():
        return face_operation("face_evidence", frame, minimum_face_size=minimum_face_size,
                              minimum_detection_score=minimum_detection_score,
                              minimum_blur_score=minimum_blur_score, max_faces=max_faces)

    prepared = _prepare_frame(frame)
    if prepared is None:
        return []

    required_size = max(16, int(minimum_face_size))
    required_score = max(0.1, min(0.99, float(minimum_detection_score)))
    required_blur = max(0.0, float(minimum_blur_score))
    limit = max(1, min(16, int(max_faces)))
    height, width = prepared.shape[:2]

    detector, _ = _models()
    faces = _detect_faces(detector, prepared, required_score)
    if faces is None:
        return []

    candidates: list[dict[str, Any]] = []
    for face in faces:
        if face.size < 15 or not np.all(np.isfinite(face)):
            continue

        x, y, face_width, face_height = map(float, face[:4])
        if face_width < required_size or face_height < required_size:
            continue

        detection_score = float(face[-1])
        if detection_score < required_score:
            continue

        x1 = max(0, min(width, int(math.floor(x))))
        y1 = max(0, min(height, int(math.floor(y))))
        x2 = max(0, min(width, int(math.ceil(x + face_width))))
        y2 = max(0, min(height, int(math.ceil(y + face_height))))
        if x2 <= x1 or y2 <= y1:
            continue

        face_roi = prepared[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        blur_score = _face_blur_score(face_roi)
        if blur_score < required_blur:
            continue

        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
        except cv2.error:
            brightness = 0.0

        # YuNet exposes five landmarks after x/y/w/h.  Measure simple facial
        # symmetry to prefer frontal evidence without performing recognition.
        frontal_score = 0.5
        try:
            right_eye = np.asarray(face[4:6], dtype=np.float32)
            left_eye = np.asarray(face[6:8], dtype=np.float32)
            nose = np.asarray(face[8:10], dtype=np.float32)
            right_mouth = np.asarray(face[10:12], dtype=np.float32)
            left_mouth = np.asarray(face[12:14], dtype=np.float32)
            eye_distance = max(1.0, float(np.linalg.norm(left_eye - right_eye)))
            eye_midpoint = (left_eye + right_eye) * 0.5
            mouth_midpoint = (left_mouth + right_mouth) * 0.5
            eye_level = 1.0 - min(
                1.0,
                abs(float(left_eye[1] - right_eye[1])) / (eye_distance * 0.30),
            )
            nose_center = 1.0 - min(
                1.0,
                abs(float(nose[0] - eye_midpoint[0])) / (eye_distance * 0.55),
            )
            mouth_center = 1.0 - min(
                1.0,
                abs(float(mouth_midpoint[0] - eye_midpoint[0])) / (eye_distance * 0.65),
            )
            frontal_score = max(
                0.0,
                min(1.0, eye_level * 0.30 + nose_center * 0.45 + mouth_center * 0.25),
            )
        except (TypeError, ValueError, FloatingPointError, IndexError):
            frontal_score = 0.5

        candidates.append(
            {
                "box": [x1, y1, x2, y2],
                "detection_score": detection_score,
                "blur_score": blur_score,
                "brightness": brightness,
                "frontal_score": frontal_score,
                "face_width": int(x2 - x1),
                "face_height": int(y2 - y1),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item.get("detection_score", 0.0)),
            float(item.get("blur_score", 0.0)),
            int(item.get("face_width", 0)) * int(item.get("face_height", 0)),
        ),
        reverse=True,
    )
    return candidates[:limit]


def _reference_variants(frame: np.ndarray) -> list[np.ndarray]:
    """Return identity-preserving exposure variants for enrollment detection."""
    variants = [frame]
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8),
        ).apply(lightness)
        enhanced = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
        variants.append(np.ascontiguousarray(enhanced))
    except cv2.error:
        pass
    return variants


# ============================================================
# SFACE EMBEDDING
# ============================================================

def _extract_sface_embedding(
    recognizer: Any,
    frame: np.ndarray,
    face: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Thread-safe SFace alignment and feature extraction.
    """

    try:
        face_for_opencv = np.asarray(
            face,
            dtype=np.float32,
        ).reshape(-1)

        with _recognizer_lock:
            aligned = recognizer.alignCrop(
                frame,
                face_for_opencv,
            )

            if aligned is None or aligned.size == 0:
                return None

            feature = recognizer.feature(
                aligned
            )

            # Copy before releasing native model state.
            if feature is not None:
                feature = np.asarray(
                    feature,
                    dtype=np.float32,
                ).reshape(-1).copy()

    except cv2.error as exc:
        logger.debug(
            "SFace inference rejected face | error=%s",
            str(exc),
        )
        return None

    except (
        TypeError,
        ValueError,
    ):
        return None

    if feature is None or feature.size == 0:
        return None

    if not np.all(np.isfinite(feature)):
        return None

    norm = float(
        np.linalg.norm(feature)
    )

    if not math.isfinite(norm) or norm <= 1e-12:
        return None

    return (
        feature / norm
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# PUBLIC FACE EXTRACTION
# ============================================================

def _embeddings_from_candidates(
    recognizer: Any,
    frame: np.ndarray,
    faces: Optional[np.ndarray],
    *,
    minimum_face_size: int,
    minimum_detection_score: float,
    minimum_blur_score: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate YuNet candidates and create SFace embeddings.

    Returning rejection counters is operationally useful: a production
    camera operator can distinguish "no face" from "face too small/blurred"
    without logging biometric vectors or image data.
    """

    counters = {
        "detected": 0,
        "invalid": 0,
        "too_small": 0,
        "low_confidence": 0,
        "blurred": 0,
        "embedding_failed": 0,
    }
    results: list[dict[str, Any]] = []

    if faces is None:
        return results, counters

    height, width = frame.shape[:2]

    for raw_face in faces:
        counters["detected"] += 1
        face = np.asarray(raw_face, dtype=np.float32).reshape(-1)

        # YuNet output: x, y, w, h, five landmark pairs, confidence.
        # SFace alignment requires the landmark-bearing output, not merely a
        # four-value bounding box.
        if face.size < 15 or not np.all(np.isfinite(face)):
            counters["invalid"] += 1
            continue

        x, y, face_width, face_height = map(float, face[:4])
        if face_width <= 0.0 or face_height <= 0.0:
            counters["invalid"] += 1
            continue

        if (
            face_width < minimum_face_size
            or face_height < minimum_face_size
        ):
            counters["too_small"] += 1
            continue

        detection_score = float(face[-1])
        if (
            not math.isfinite(detection_score)
            or detection_score < minimum_detection_score
        ):
            counters["low_confidence"] += 1
            continue

        x1 = max(0, min(width, int(math.floor(x))))
        y1 = max(0, min(height, int(math.floor(y))))
        x2 = max(0, min(width, int(math.ceil(x + face_width))))
        y2 = max(0, min(height, int(math.ceil(y + face_height))))

        if x2 <= x1 or y2 <= y1:
            counters["invalid"] += 1
            continue

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            counters["invalid"] += 1
            continue

        blur_score = _face_blur_score(face_roi)
        if blur_score < minimum_blur_score:
            counters["blurred"] += 1
            continue

        embedding = _extract_sface_embedding(
            recognizer,
            frame,
            face,
        )
        if embedding is None:
            counters["embedding_failed"] += 1
            continue

        results.append(
            {
                "embedding": embedding,
                "box": [x1, y1, x2, y2],
                "detection_score": detection_score,
                "blur_score": blur_score,
                "model": "SFace",
                "model_version": VERSION,
            }
        )

    return results, counters


def extract_embeddings(
    frame: Any,
    *,
    profile: str = "live",
    minimum_face_size: Optional[int] = None,
    minimum_detection_score: Optional[float] = None,
    minimum_blur_score: Optional[float] = None,
    allow_fallback: bool = True,
) -> list[dict[str, Any]]:
    """Detect faces and return normalized SFace embeddings.

    ``profile="live"`` uses CCTV gates and may perform a bounded relaxed
    retry. ``profile="enrollment"`` uses the separate reference-image gates
    and never weakens them below their configured values. The biometric match
    threshold is never relaxed by either profile.
    """

    from inference.triton_client import triton_enabled, face_operation
    if triton_enabled():
        return face_operation("face_embeddings", frame, profile=profile,
                              minimum_face_size=minimum_face_size,
                              minimum_detection_score=minimum_detection_score,
                              minimum_blur_score=minimum_blur_score,
                              allow_fallback=allow_fallback)

    prepared = _prepare_frame(frame)
    if prepared is None:
        return []

    normalized_profile = str(profile or "live").strip().lower()
    if normalized_profile not in {"live", "enrollment"}:
        raise ValueError("Face extraction profile must be 'live' or 'enrollment'")

    if normalized_profile == "enrollment":
        default_size = REFERENCE_MIN_SIZE
        default_detection_score = REFERENCE_MIN_DET_SCORE
        default_blur_score = REFERENCE_MIN_BLUR_SCORE
    else:
        default_size = MIN_SIZE
        default_detection_score = MIN_DET_SCORE
        default_blur_score = MIN_BLUR_SCORE

    required_size = (
        default_size
        if minimum_face_size is None
        else max(16, int(minimum_face_size))
    )
    required_detection_score = (
        default_detection_score
        if minimum_detection_score is None
        else max(0.1, min(0.99, float(minimum_detection_score)))
    )
    required_blur_score = (
        default_blur_score
        if minimum_blur_score is None
        else max(0.0, float(minimum_blur_score))
    )

    detector, recognizer = _models()
    height, width = prepared.shape[:2]

    variants = _reference_variants(prepared)
    retry_low_light = (
        REFERENCE_LOW_LIGHT_RETRY_ENABLED
        if normalized_profile == "enrollment"
        else LIVE_LOW_LIGHT_RETRY_ENABLED
    )
    if not retry_low_light:
        variants = variants[:1]

    attempts: list[tuple[str, np.ndarray, int, float, float]] = []
    for index, candidate in enumerate(variants):
        attempts.append(
            (
                "original" if index == 0 else "low-light",
                candidate,
                required_size,
                required_detection_score,
                required_blur_score,
            )
        )

    # A live retry is allowed after *all candidates were rejected*, not only
    # when YuNet returned no boxes. This fixes the V6 regression where a small
    # or blurred detection prevented the recovery path from running.
    if normalized_profile == "live" and allow_fallback:
        fallback_size = min(required_size, LIVE_FALLBACK_MIN_SIZE)
        fallback_detection = min(
            required_detection_score,
            LIVE_FALLBACK_MIN_DET_SCORE,
        )
        fallback_blur = min(
            required_blur_score,
            LIVE_FALLBACK_MIN_BLUR_SCORE,
        )
        for index, candidate in enumerate(variants):
            attempts.append(
                (
                    "fallback-original" if index == 0 else "fallback-low-light",
                    candidate,
                    fallback_size,
                    fallback_detection,
                    fallback_blur,
                )
            )

    last_counters: dict[str, int] = {}
    for attempt_name, candidate, size_gate, score_gate, blur_gate in attempts:
        faces = _detect_faces(detector, candidate, score_gate)
        results, counters = _embeddings_from_candidates(
            recognizer,
            candidate,
            faces,
            minimum_face_size=size_gate,
            minimum_detection_score=score_gate,
            minimum_blur_score=blur_gate,
        )
        last_counters = counters

        if results:
            if attempt_name.startswith("fallback"):
                logger.info(
                    "Recovered live face candidates after quality rejection "
                    "| attempt=%s width=%d height=%d accepted=%d",
                    attempt_name,
                    width,
                    height,
                    len(results),
                )
            return results

    logger.debug(
        "No usable face embeddings | profile=%s width=%d height=%d "
        "rejections=%s",
        normalized_profile,
        width,
        height,
        last_counters,
    )
    return []


# ============================================================
# SINGLE FACE EXTRACTION
# ============================================================

def extract_single_embedding(
    frame: Any,
) -> np.ndarray:
    """Extract one enrollment-quality reference embedding.

    Live analytics should call :func:`extract_embeddings`, which keeps the
    stricter live-camera quality gates.
    """
    prepared = _prepare_frame(frame)

    if prepared is None:
        raise ValueError("The reference image is invalid or too small.")

    embeddings = extract_embeddings(
        prepared,
        profile="enrollment",
        allow_fallback=False,
    )

    if len(embeddings) == 1:
        return embeddings[0]["embedding"]

    if len(embeddings) > 1:
        raise ValueError(
            "Multiple faces detected. Please upload an image "
            "containing exactly one face."
        )

    raise ValueError(
        "No usable face detected in the reference image. Please upload "
        "one well-lit image containing a single visible face."
    )


# ============================================================
# BACKWARD-COMPATIBLE PUBLIC ALIASES
# ============================================================

def generate_face_embedding(
    frame: Any,
) -> np.ndarray:
    return extract_single_embedding(frame)


def extract_face_embedding(
    frame: Any,
) -> np.ndarray:
    return extract_single_embedding(frame)


def get_face_embedding(
    frame: Any,
) -> np.ndarray:
    return extract_single_embedding(frame)


def create_face_embedding(
    frame: Any,
) -> np.ndarray:
    return extract_single_embedding(frame)


def encode_face(
    frame: Any,
) -> np.ndarray:
    return extract_single_embedding(frame)


# ============================================================
# EMBEDDING MATCHING
# ============================================================

def _normalize_embedding(
    value: Any,
) -> Optional[np.ndarray]:
    try:
        vector = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if vector.size == 0:
        return None

    if not np.all(np.isfinite(vector)):
        return None

    norm = float(
        np.linalg.norm(vector)
    )

    if not math.isfinite(norm) or norm <= 1e-12:
        return None

    return (
        vector / norm
    ).astype(
        np.float32,
        copy=False,
    )


def match_embeddings(
    query: Any,
    refs: Iterable,
    threshold: Optional[float] = None,
):
    """
    Compare a normalized query embedding against reference embeddings
    using cosine similarity.

    Returns:
        (best_entry, score)

    If the best score does not satisfy the threshold:
        (None, best_score)
    """

    query_vector = _normalize_embedding(
        query
    )

    if query_vector is None:
        return None, -1.0

    if threshold is None:
        cut = THRESH
    else:
        try:
            cut = float(threshold)
        except (TypeError, ValueError):
            raise ValueError(
                "Face match threshold must be numeric"
            )

        if not math.isfinite(cut):
            raise ValueError(
                "Face match threshold must be finite"
            )

        cut = max(-1.0, min(1.0, cut))

    best = None
    best_score = -1.0

    for item in refs:
        try:
            entry, embedding = item
        except (
            TypeError,
            ValueError,
        ):
            continue

        reference = _normalize_embedding(
            embedding
        )

        if reference is None:
            continue

        if reference.size != query_vector.size:
            continue

        score = float(
            np.dot(
                query_vector,
                reference,
            )
        )

        if not math.isfinite(score):
            continue

        # Numerical rounding can produce tiny excursions.
        score = max(
            -1.0,
            min(1.0, score),
        )

        if score > best_score:
            best = entry
            best_score = score

    if best is not None and best_score >= cut:
        return best, best_score

    return None, best_score


# ============================================================
# MODEL HEALTH / READINESS SUPPORT
# ============================================================

def model_status() -> dict[str, Any]:
    """
    Lightweight configuration/load status.

    This does NOT claim biometric accuracy has been validated.
    """

    detector_path = (
        Path(DETECTOR).expanduser()
        if DETECTOR
        else None
    )

    recognizer_path = (
        Path(RECOGNIZER).expanduser()
        if RECOGNIZER
        else None
    )

    return {
        "version": VERSION,
        "detector_configured": bool(DETECTOR),
        "recognizer_configured": bool(RECOGNIZER),
        "detector_present": bool(
            detector_path
            and detector_path.is_file()
        ),
        "recognizer_present": bool(
            recognizer_path
            and recognizer_path.is_file()
        ),
        "detector_loaded": _det is not None,
        "recognizer_loaded": _rec is not None,
        "minimum_detection_score": MIN_DET_SCORE,
        "minimum_face_size": MIN_SIZE,
        "minimum_blur_score": MIN_BLUR_SCORE,
        "reference_minimum_detection_score": REFERENCE_MIN_DET_SCORE,
        "reference_minimum_face_size": REFERENCE_MIN_SIZE,
        "reference_minimum_blur_score": REFERENCE_MIN_BLUR_SCORE,
        "reference_low_light_retry_enabled": REFERENCE_LOW_LIGHT_RETRY_ENABLED,
        "live_low_light_retry_enabled": LIVE_LOW_LIGHT_RETRY_ENABLED,
        "live_fallback_minimum_detection_score": LIVE_FALLBACK_MIN_DET_SCORE,
        "live_fallback_minimum_face_size": LIVE_FALLBACK_MIN_SIZE,
        "live_fallback_minimum_blur_score": LIVE_FALLBACK_MIN_BLUR_SCORE,
        "match_threshold": THRESH,
    }
