from __future__ import annotations

import json
import logging
import math
import os
import re
import struct
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from db.database import SessionLocal
from db.model import CameraConnection

from db.intelligence_model import (
    Vehicle,
    VehicleObservation,
    VehicleEmbedding,
    PlateObservation,
    CorrelationDecision,
    ActivityEvent,
)


# ============================================================
# LOGGING
# ============================================================

def _env_positive_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int = 10_000_000,
) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_GLOBAL_VEHICLE_ID_LENGTH = 100
MAX_CAMERA_ID_LENGTH = 100
MAX_TRACK_ID_LENGTH = 100
MAX_PLATE_LENGTH = 50
MAX_MODEL_NAME_LENGTH = 150
MAX_MODEL_VERSION_LENGTH = 100
MAX_REASON_LENGTH = 4000
MAX_EVENT_ID_LENGTH = 120
MAX_RULE_ID_LENGTH = 100
MAX_EVENT_TYPE_LENGTH = 100
MAX_SEVERITY_LENGTH = 30
MAX_STATUS_LENGTH = 30

MAX_METADATA_KEYS = 100
MAX_METADATA_DEPTH = 6
MAX_METADATA_STRING_LENGTH = 2000
MAX_METADATA_LIST_LENGTH = 100

MAX_EMBEDDING_DIMENSION = _env_positive_int(
    "MAX_REID_EMBEDDING_DIMENSION",
    4096,
    minimum=16,
    maximum=65536,
)

MIN_EMBEDDING_DIMENSION = _env_positive_int(
    "MIN_REID_EMBEDDING_DIMENSION",
    16,
    minimum=1,
    maximum=65536,
)

MAX_EMBEDDING_BYTES = (
    MAX_EMBEDDING_DIMENSION * 4
)

MAX_QUERY_LIMIT = _env_positive_int(
    "INTELLIGENCE_MAX_QUERY_LIMIT",
    1000,
    minimum=1,
    maximum=100000,
)


# ============================================================
# TIME
# ============================================================

def utc_now() -> datetime:
    """
    Return naive UTC datetime.

    The current SQLAlchemy DateTime columns are timezone-naive,
    therefore timezone information is removed before persistence.
    """

    return datetime.now(
        timezone.utc
    ).replace(
        tzinfo=None
    )


def normalize_datetime(
    value: Optional[datetime],
) -> datetime:
    """
    Normalize an incoming datetime to naive UTC.

    Naive timestamps are treated as UTC because the intelligence
    database currently uses timezone-naive DateTime columns.
    """

    if value is None:
        return utc_now()

    if not isinstance(
        value,
        datetime,
    ):
        raise ValueError(
            "timestamp must be a datetime"
        )

    if value.tzinfo is None:
        return value

    return value.astimezone(
        timezone.utc
    ).replace(
        tzinfo=None
    )


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    allow_none: bool = True,
) -> Optional[float]:

    if value is None:

        if allow_none:
            return None

        raise ValueError(
            f"{name} is required"
        )

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:

        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(
        number
    ):

        raise ValueError(
            f"{name} must be finite"
        )

    if (
        minimum is not None
        and number < minimum
    ):

        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    if (
        maximum is not None
        and number > maximum
    ):

        raise ValueError(
            f"{name} must be <= {maximum}"
        )

    return number


def clamp01(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return default

    if not math.isfinite(
        number
    ):

        return default

    return max(
        0.0,
        min(
            1.0,
            number,
        ),
    )


# ============================================================
# STRING VALIDATION
# ============================================================

def _safe_string(
    value: Any,
    *,
    name: str,
    max_length: int,
    required: bool = False,
) -> Optional[str]:

    if value is None:

        if required:
            raise ValueError(
                f"{name} is required"
            )

        return None

    try:
        text = str(
            value
        ).strip()

    except Exception as exc:

        raise ValueError(
            f"{name} is invalid"
        ) from exc

    if not text:

        if required:
            raise ValueError(
                f"{name} cannot be empty"
            )

        return None

    if len(text) > max_length:

        raise ValueError(
            f"{name} exceeds maximum length "
            f"{max_length}"
        )

    # Reject control characters.
    if any(
        ord(character) < 32
        and character not in (
            "\t",
            "\n",
            "\r",
        )
        for character in text
    ):

        raise ValueError(
            f"{name} contains invalid control characters"
        )

    return text


def _safe_optional_id(
    value: Any,
    *,
    name: str,
    max_length: int,
) -> Optional[str]:

    return _safe_string(
        value,
        name=name,
        max_length=max_length,
        required=False,
    )


# ============================================================
# PLATE NORMALIZATION
# ============================================================

_PLATE_PATTERN = re.compile(
    r"^[A-Z0-9]+$"
)


def normalize_plate(
    plate_text: Optional[str],
) -> Optional[str]:
    """
    Conservative ANPR normalization.

    Formatting separators are removed, but OCR character
    substitutions such as O<->0 or I<->1 are NOT performed.

    This is intentional because aggressive correction can turn
    one vehicle's observation into another vehicle's identity.
    """

    if plate_text is None:

        return None

    try:

        value = str(
            plate_text
        ).strip().upper()

    except Exception:

        return None

    if not value:

        return None

    value = re.sub(
        r"[\s\-_.:/\\|]+",
        "",
        value,
    )

    value = re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )

    if not value:

        return None

    if len(value) > MAX_PLATE_LENGTH:

        return None

    if not _PLATE_PATTERN.fullmatch(
        value
    ):

        return None

    return value


# ============================================================
# JSON / METADATA SAFETY
# ============================================================

def _sanitize_json_value(
    value: Any,
    *,
    depth: int = 0,
) -> Any:

    if depth > MAX_METADATA_DEPTH:

        return "[MAX_DEPTH]"

    if value is None:

        return None

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        int,
    ):

        return value

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):

            return None

        return value

    if isinstance(
        value,
        str,
    ):

        return value[
            :MAX_METADATA_STRING_LENGTH
        ]

    if isinstance(
        value,
        datetime,
    ):

        return normalize_datetime(
            value
        ).isoformat()

    if isinstance(
        value,
        dict,
    ):

        result = {}

        for index, (
            key,
            item,
        ) in enumerate(
            value.items()
        ):

            if (
                index
                >= MAX_METADATA_KEYS
            ):

                break

            safe_key = str(
                key
            )[
                :200
            ]

            result[
                safe_key
            ] = _sanitize_json_value(
                item,
                depth=depth + 1,
            )

        return result

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        result = []

        for item in list(
            value
        )[
            :MAX_METADATA_LIST_LENGTH
        ]:

            result.append(
                _sanitize_json_value(
                    item,
                    depth=depth + 1,
                )
            )

        return result

    # NumPy scalar support without
    # requiring NumPy as a persistence dependency.
    if hasattr(
        value,
        "item",
    ):

        try:

            return _sanitize_json_value(
                value.item(),
                depth=depth + 1,
            )

        except Exception:

            pass

    # Final safe representation.
    return str(
        value
    )[
        :MAX_METADATA_STRING_LENGTH
    ]


def sanitize_metadata(
    metadata: Any,
) -> Optional[dict[str, Any]]:

    if metadata is None:

        return None

    if not isinstance(
        metadata,
        dict,
    ):

        raise ValueError(
            "metadata must be a dictionary"
        )

    return _sanitize_json_value(
        metadata
    )


# ============================================================
# DATABASE HELPERS
# ============================================================

def _rollback_and_raise(
    db,
    exc: Exception,
):
    try:
        db.rollback()
    except Exception:
        logger.exception(
            "Database rollback failed"
        )

    raise exc


def _safe_commit(
    db,
) -> None:

    try:

        db.commit()

    except Exception as exc:

        try:
            db.rollback()
        except Exception:
            logger.exception(
                "Database rollback failed"
            )

        raise exc


# ============================================================
# EMBEDDING CONVERSION
# ============================================================

def embedding_to_bytes(
    embedding: Any,
) -> tuple[bytes, int]:
    """
    Convert a Re-ID embedding into float32 bytes.

    Supports:
      - numpy arrays
      - torch tensors
      - Python lists
      - tuples

    The stored representation is explicitly float32.
    """

    if embedding is None:

        raise ValueError(
            "embedding cannot be None"
        )

    # Torch tensor.
    if hasattr(
        embedding,
        "detach",
    ):

        try:

            embedding = (
                embedding
                .detach()
                .cpu()
                .numpy()
            )

        except Exception as exc:

            raise ValueError(
                "unable to convert tensor embedding"
            ) from exc

    # NumPy / array-like.
    if hasattr(
        embedding,
        "astype",
    ) and hasattr(
        embedding,
        "tobytes",
    ):

        try:

            array = embedding.astype(
                "float32",
                copy=False,
            )

            dimension = int(
                array.size
            )

            values = array.reshape(
                -1
            )

            if dimension <= 0:

                raise ValueError(
                    "embedding is empty"
                )

            if (
                dimension
                < MIN_EMBEDDING_DIMENSION
                or dimension
                > MAX_EMBEDDING_DIMENSION
            ):

                raise ValueError(
                    "embedding dimension is outside "
                    "allowed range"
                )

            # Reject NaN/Inf.
            try:

                import numpy as np

                if not np.isfinite(
                    values
                ).all():

                    raise ValueError(
                        "embedding contains NaN or Inf"
                    )

            except ImportError:

                pass

            data = values.tobytes()

            if len(
                data
            ) > MAX_EMBEDDING_BYTES:

                raise ValueError(
                    "embedding exceeds storage limit"
                )

            return (
                data,
                dimension,
            )

        except ValueError:
            raise

        except Exception as exc:

            raise ValueError(
                "invalid array embedding"
            ) from exc

    # Python iterable fallback.
    try:

        values = list(
            embedding
        )

    except Exception as exc:

        raise ValueError(
            "embedding must be array-like"
        ) from exc

    if not values:

        raise ValueError(
            "embedding is empty"
        )

    dimension = len(
        values
    )

    if (
        dimension
        < MIN_EMBEDDING_DIMENSION
        or dimension
        > MAX_EMBEDDING_DIMENSION
    ):

        raise ValueError(
            "embedding dimension is outside "
            "allowed range"
        )

    floats = []

    for value in values:

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:

            raise ValueError(
                "embedding contains a non-numeric value"
            ) from exc

        if not math.isfinite(
            number
        ):

            raise ValueError(
                "embedding contains NaN or Inf"
            )

        floats.append(
            number
        )

    try:

        data = struct.pack(
            f"{dimension}f",
            *floats,
        )

    except Exception as exc:

        raise ValueError(
            "unable to serialize embedding"
        ) from exc

    if len(
        data
    ) > MAX_EMBEDDING_BYTES:

        raise ValueError(
            "embedding exceeds storage limit"
        )

    return (
        data,
        dimension,
    )


# ============================================================
# GLOBAL VEHICLE
# ============================================================

def create_global_vehicle(
    global_vehicle_id: str,
    vehicle_type: Optional[str] = None,
    color: Optional[str] = None,
    make: Optional[str] = None,
    model: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
):
    """
    Create or update a global vehicle.

    Idempotent by global_vehicle_id.
    """

    global_id = _safe_string(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
        required=True,
    )

    vehicle_type = _safe_optional_id(
        vehicle_type,
        name="vehicle_type",
        max_length=50,
    )

    color = _safe_optional_id(
        color,
        name="color",
        max_length=50,
    )

    make = _safe_optional_id(
        make,
        name="make",
        max_length=100,
    )

    model = _safe_optional_id(
        model,
        name="model",
        max_length=100,
    )

    metadata = sanitize_metadata(
        metadata
    )

    db = SessionLocal()

    try:

        vehicle = (
            db.query(Vehicle)
            .filter(
                Vehicle.global_vehicle_id
                == global_id
            )
            .first()
        )

        now = utc_now()

        if vehicle:

            changed = False

            if (
                vehicle_type
                and not vehicle.vehicle_type
            ):

                vehicle.vehicle_type = (
                    vehicle_type
                )

                changed = True

            if (
                color
                and not vehicle.color
            ):

                vehicle.color = color
                changed = True

            if (
                make
                and not vehicle.make
            ):

                vehicle.make = make
                changed = True

            if (
                model
                and not vehicle.model
            ):

                vehicle.model = model
                changed = True

            if metadata:

                current = (
                    vehicle.metadata_json
                    or {}
                )

                if isinstance(
                    current,
                    dict,
                ):

                    merged = dict(
                        current
                    )

                    merged.update(
                        metadata
                    )

                    vehicle.metadata_json = (
                        sanitize_metadata(
                            merged
                        )
                    )

                    changed = True

            if changed:

                vehicle.updated_at = now

                _safe_commit(
                    db
                )

                db.refresh(
                    vehicle
                )

            return vehicle

        vehicle = Vehicle(
            global_vehicle_id=global_id,
            vehicle_type=vehicle_type,
            color=color,
            make=make,
            model=model,
            first_seen_at=now,
            last_seen_at=now,
            observation_count=0,
            status="ACTIVE",
            metadata_json=metadata,
            created_at=now,
            updated_at=now,
        )

        db.add(
            vehicle
        )

        try:

            db.flush()

        except IntegrityError:

            db.rollback()

            # Another worker may have created
            # the same global identity.
            vehicle = (
                db.query(Vehicle)
                .filter(
                    Vehicle.global_vehicle_id
                    == global_id
                )
                .first()
            )

            if vehicle is None:

                raise

            return vehicle

        _safe_commit(
            db
        )

        db.refresh(
            vehicle
        )

        return vehicle

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to create/update global vehicle: %s",
            global_id,
        )

        raise

    finally:

        db.close()


# ============================================================
# GET GLOBAL VEHICLE
# ============================================================

def get_global_vehicle(
    global_vehicle_id: str,
):
    if not global_vehicle_id:
        return None

    global_id = _safe_string(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
        required=True,
    )

    db = SessionLocal()

    try:

        return (
            db.query(Vehicle)
            .filter(
                Vehicle.global_vehicle_id
                == global_id
            )
            .first()
        )

    finally:

        db.close()


# ============================================================
# VEHICLE OBSERVATION
# ============================================================

def save_vehicle_observation(
    camera_id: str,
    global_vehicle_id: Optional[str] = None,
    local_track_id: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    confidence: Optional[float] = None,
    frame_timestamp: Optional[
        datetime
    ] = None,
    bbox: Optional[Any] = None,
    crop_storage_key: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
    vehicle_id: Optional[int] = None,
):
    """
    Persist one vehicle observation.

    The observation is the durable evidence unit. Global identity
    is attached when correlation has assigned one.
    """

    camera = _safe_string(
        camera_id,
        name="camera_id",
        max_length=MAX_CAMERA_ID_LENGTH,
        required=True,
    )

    global_id = _safe_optional_id(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
    )

    track_id = _safe_optional_id(
        local_track_id,
        name="local_track_id",
        max_length=MAX_TRACK_ID_LENGTH,
    )

    vehicle_type = _safe_optional_id(
        vehicle_type,
        name="vehicle_type",
        max_length=50,
    )

    confidence = _finite_float(
        confidence,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )

    frame_time = normalize_datetime(
        frame_timestamp
    )

    crop_key = _safe_optional_id(
        crop_storage_key,
        name="crop_storage_key",
        max_length=500,
    )

    model_name = _safe_optional_id(
        model_name,
        name="model_name",
        max_length=MAX_MODEL_NAME_LENGTH,
    )

    model_version = _safe_optional_id(
        model_version,
        name="model_version",
        max_length=MAX_MODEL_VERSION_LENGTH,
    )

    metadata = sanitize_metadata(
        metadata
    )

    if bbox is not None:

        if not isinstance(
            bbox,
            (
                list,
                tuple,
                dict,
            ),
        ):

            raise ValueError(
                "bbox must be a JSON-compatible object"
            )

    db = SessionLocal()

    try:

        vehicle = None

        if vehicle_id is not None:

            try:

                vehicle_pk = int(
                    vehicle_id
                )

            except (
                TypeError,
                ValueError,
            ):

                raise ValueError(
                    "vehicle_id must be an integer"
                )

            vehicle = (
                db.query(Vehicle)
                .filter(
                    Vehicle.id
                    == vehicle_pk
                )
                .first()
            )

        elif global_id:

            vehicle = (
                db.query(Vehicle)
                .filter(
                    Vehicle.global_vehicle_id
                    == global_id
                )
                .first()
            )

            if vehicle is None:

                now = utc_now()

                vehicle = Vehicle(
                    global_vehicle_id=global_id,
                    vehicle_type=vehicle_type,
                    first_seen_at=frame_time,
                    last_seen_at=frame_time,
                    observation_count=0,
                    status="ACTIVE",
                    created_at=now,
                    updated_at=now,
                )

                db.add(
                    vehicle
                )

                try:

                    db.flush()

                except IntegrityError:

                    db.rollback()

                    vehicle = (
                        db.query(Vehicle)
                        .filter(
                            Vehicle.global_vehicle_id
                            == global_id
                        )
                        .first()
                    )

                    if vehicle is None:
                        raise

        observation = VehicleObservation(
            vehicle_id=(
                vehicle.id
                if vehicle
                else None
            ),
            global_vehicle_id=global_id,
            camera_id=camera,
            local_track_id=track_id,
            vehicle_type=vehicle_type,
            confidence=confidence,
            frame_timestamp=frame_time,
            inference_timestamp=utc_now(),
            bbox=sanitize_metadata(
                bbox
                if isinstance(
                    bbox,
                    dict,
                )
                else (
                    {
                        "xyxy": list(
                            bbox
                        )
                    }
                    if bbox is not None
                    else None
                )
            ),
            crop_storage_key=crop_key,
            model_name=model_name,
            model_version=model_version,
            metadata_json=metadata,
            created_at=utc_now(),
        )

        db.add(
            observation
        )

        if vehicle:

            vehicle.last_seen_at = (
                frame_time
            )

            vehicle.observation_count = (
                int(
                    vehicle.observation_count
                    or 0
                )
                + 1
            )

            if (
                vehicle_type
                and not vehicle.vehicle_type
            ):

                vehicle.vehicle_type = (
                    vehicle_type
                )

            vehicle.updated_at = utc_now()

        _safe_commit(
            db
        )

        db.refresh(
            observation
        )

        return observation

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to persist vehicle observation "
            "camera=%s track=%s global=%s",
            camera,
            track_id,
            global_id,
        )

        raise

    finally:

        db.close()


# ============================================================
# VEHICLE EMBEDDING
# ============================================================

def save_vehicle_embedding(
    global_vehicle_id: str,
    embedding: Any,
    model_name: str,
    model_version: str,
    observation_id: Optional[int] = None,
    quality_score: Optional[float] = None,
):
    """
    Persist a validated Re-ID embedding.

    This is durable evidence. Real-time vector matching should
    continue to use the existing in-memory/vector path.
    """

    global_id = _safe_string(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
        required=True,
    )

    model_name = _safe_string(
        model_name,
        name="model_name",
        max_length=MAX_MODEL_NAME_LENGTH,
        required=True,
    )

    model_version = (
        _safe_string(
            model_version,
            name="model_version",
            max_length=MAX_MODEL_VERSION_LENGTH,
            required=False,
        )
        or "unknown"
    )

    quality_score = _finite_float(
        quality_score,
        name="quality_score",
        minimum=0.0,
        maximum=1.0,
    )

    if observation_id is not None:

        try:

            observation_id = int(
                observation_id
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "observation_id must be an integer"
            )

    embedding_bytes, dimension = (
        embedding_to_bytes(
            embedding
        )
    )

    db = SessionLocal()

    try:

        vehicle = (
            db.query(Vehicle)
            .filter(
                Vehicle.global_vehicle_id
                == global_id
            )
            .first()
        )

        if vehicle is None:

            now = utc_now()

            vehicle = Vehicle(
                global_vehicle_id=global_id,
                vehicle_type=None,
                first_seen_at=now,
                last_seen_at=now,
                observation_count=0,
                status="ACTIVE",
                created_at=now,
                updated_at=now,
            )

            db.add(
                vehicle
            )

            try:

                db.flush()

            except IntegrityError:

                db.rollback()

                vehicle = (
                    db.query(Vehicle)
                    .filter(
                        Vehicle.global_vehicle_id
                        == global_id
                    )
                    .first()
                )

                if vehicle is None:
                    raise

        record = VehicleEmbedding(
            vehicle_id=vehicle.id,
            observation_id=observation_id,
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            embedding=embedding_bytes,
            quality_score=quality_score,
            created_at=utc_now(),
        )

        db.add(
            record
        )

        _safe_commit(
            db
        )

        db.refresh(
            record
        )

        return record

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to persist vehicle embedding "
            "global=%s",
            global_id,
        )

        raise

    finally:

        db.close()


# ============================================================
# ANPR / PLATE OBSERVATION
# ============================================================

def save_plate_observation(
    camera_id: str,
    plate_text: str,
    confidence: Optional[float] = None,
    global_vehicle_id: Optional[str] = None,
    local_track_id: Optional[str] = None,
    frame_timestamp: Optional[
        datetime
    ] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
):
    """
    Persist an ANPR observation.

    Both raw OCR text and normalized plate are retained.
    Normalization is conservative and does not perform
    O/0 or I/1 identity substitutions.
    """

    camera = _safe_string(
        camera_id,
        name="camera_id",
        max_length=MAX_CAMERA_ID_LENGTH,
        required=True,
    )

    raw_plate = _safe_string(
        plate_text,
        name="plate_text",
        max_length=MAX_PLATE_LENGTH,
        required=True,
    )

    normalized = normalize_plate(
        raw_plate
    )

    if not normalized:

        raise ValueError(
            "plate_text is invalid after normalization"
        )

    confidence = _finite_float(
        confidence,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )

    global_id = _safe_optional_id(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
    )

    track_id = _safe_optional_id(
        local_track_id,
        name="local_track_id",
        max_length=MAX_TRACK_ID_LENGTH,
    )

    frame_time = normalize_datetime(
        frame_timestamp
    )

    model_name = _safe_optional_id(
        model_name,
        name="model_name",
        max_length=MAX_MODEL_NAME_LENGTH,
    )

    model_version = _safe_optional_id(
        model_version,
        name="model_version",
        max_length=MAX_MODEL_VERSION_LENGTH,
    )

    metadata = sanitize_metadata(
        metadata
    )

    db = SessionLocal()

    try:

        vehicle = None

        if global_id:

            vehicle = (
                db.query(Vehicle)
                .filter(
                    Vehicle.global_vehicle_id
                    == global_id
                )
                .first()
            )

        record = PlateObservation(
            vehicle_id=(
                vehicle.id
                if vehicle
                else None
            ),
            camera_id=camera,
            local_track_id=track_id,
            plate_text=raw_plate,
            normalized_plate=normalized,
            confidence=confidence,
            frame_timestamp=frame_time,
            model_name=model_name,
            model_version=model_version,
            metadata_json=metadata,
            created_at=utc_now(),
        )

        db.add(
            record
        )

        _safe_commit(
            db
        )

        db.refresh(
            record
        )

        return record

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to persist plate observation "
            "camera=%s plate=%s",
            camera,
            normalized,
        )

        raise

    finally:

        db.close()


# ============================================================
# CORRELATION DECISION
# ============================================================

def save_correlation_decision(
    source_camera_id: str,
    source_track_id: Optional[str] = None,
    source_global_vehicle_id: Optional[str] = None,
    candidate_global_vehicle_id: Optional[str] = None,
    reid_score: Optional[float] = None,
    plate_score: Optional[float] = None,
    vehicle_type_score: Optional[float] = None,
    track_quality_score: Optional[float] = None,
    temporal_score: Optional[float] = None,
    gis_score: Optional[float] = None,
    direction_score: Optional[float] = None,
    final_score: Optional[float] = None,
    decision: str = "UNKNOWN",
    reason: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    reid_observation_quality: Optional[float] = None,
    crop_quality: Optional[float] = None,
    embedding_quality: Optional[float] = None,
    candidate_margin: Optional[float] = None,
    candidate_count: Optional[int] = None,
    transition_feasible: Optional[bool] = None,
    evidence_metadata: Optional[dict[str, Any]] = None,
):
    """
    Persist the complete correlation evidence breakdown.

    Scores are stored separately so investigators can understand
    why a correlation was accepted or rejected.
    """

    source_camera = _safe_string(
        source_camera_id,
        name="source_camera_id",
        max_length=MAX_CAMERA_ID_LENGTH,
        required=True,
    )

    source_track = _safe_optional_id(
        source_track_id,
        name="source_track_id",
        max_length=MAX_TRACK_ID_LENGTH,
    )

    source_global = _safe_optional_id(
        source_global_vehicle_id,
        name="source_global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
    )

    candidate_global = _safe_optional_id(
        candidate_global_vehicle_id,
        name="candidate_global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
    )

    scores = {
        "reid_score":
            reid_score,

        "plate_score":
            plate_score,

        "vehicle_type_score":
            vehicle_type_score,

        "track_quality_score":
            track_quality_score,

        "temporal_score":
            temporal_score,

        "gis_score":
            gis_score,

        "direction_score":
            direction_score,

        "final_score":
            final_score,
    }

    normalized_scores = {}

    for name, value in scores.items():

        normalized_scores[
            name
        ] = _finite_float(
            value,
            name=name,
            minimum=0.0,
            maximum=1.0,
        )

    decision = _safe_string(
        decision,
        name="decision",
        max_length=30,
        required=True,
    ).upper()

    allowed_decisions = {
        "CONFIRMED",
        "PROBABLE",
        "POSSIBLE",
        "UNKNOWN",
        "REJECTED",
        "ACCEPTED",
        "REVIEW",
        "MERGED",
        "SPLIT",
    }

    if decision not in allowed_decisions:

        logger.warning(
            "Unknown correlation decision '%s'; "
            "persisting normalized value",
            decision,
        )

    reason = _safe_string(
        reason,
        name="reason",
        max_length=MAX_REASON_LENGTH,
    )

    reid_observation_quality = _finite_float(
        reid_observation_quality,
        name="reid_observation_quality",
        minimum=0.0,
        maximum=1.0,
    )

    crop_quality = _finite_float(
        crop_quality,
        name="crop_quality",
        minimum=0.0,
        maximum=1.0,
    )

    embedding_quality = _finite_float(
        embedding_quality,
        name="embedding_quality",
        minimum=0.0,
        maximum=1.0,
    )

    candidate_margin = _finite_float(
        candidate_margin,
        name="candidate_margin",
        minimum=0.0,
        maximum=1.0,
    )

    if candidate_count is not None:
        try:
            candidate_count = int(candidate_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "candidate_count must be an integer"
            ) from exc

        candidate_count = max(0, min(candidate_count, MAX_QUERY_LIMIT))

    evidence_metadata = sanitize_metadata(
        evidence_metadata
    )

    model_name = _safe_optional_id(
        model_name,
        name="model_name",
        max_length=MAX_MODEL_NAME_LENGTH,
    )

    model_version = _safe_optional_id(
        model_version,
        name="model_version",
        max_length=MAX_MODEL_VERSION_LENGTH,
    )

    db = SessionLocal()

    try:

        correlation_kwargs = {
            "source_camera_id": source_camera,
            "source_track_id": source_track,
            "source_global_vehicle_id": source_global,
            "candidate_global_vehicle_id": candidate_global,
            "reid_score": normalized_scores["reid_score"],
            "plate_score": normalized_scores["plate_score"],
            "vehicle_type_score": normalized_scores["vehicle_type_score"],
            "track_quality_score": normalized_scores["track_quality_score"],
            "temporal_score": normalized_scores["temporal_score"],
            "gis_score": normalized_scores["gis_score"],
            "direction_score": normalized_scores["direction_score"],
            "final_score": normalized_scores["final_score"],
            "decision": decision,
            "reason": reason,
            "model_name": model_name,
            "model_version": model_version,
            "created_at": utc_now(),
        }

        optional_correlation_fields = {
            "reid_observation_quality": reid_observation_quality,
            "crop_quality": crop_quality,
            "embedding_quality": embedding_quality,
            "candidate_margin": candidate_margin,
            "candidate_count": candidate_count,
            "transition_feasible": transition_feasible,
            "evidence_metadata": evidence_metadata,
        }

        # Only pass fields that exist on the deployed ORM model.
        # This allows the persistence layer to be upgraded before
        # the database migration is deployed.
        correlation_columns = getattr(
            CorrelationDecision,
            "__table__",
            None,
        )
        correlation_column_names = set()

        if correlation_columns is not None:
            correlation_column_names = {
                column.name
                for column in correlation_columns.columns
            }

        for field_name, field_value in optional_correlation_fields.items():
            if (
                field_value is not None
                and field_name in correlation_column_names
            ):
                correlation_kwargs[field_name] = field_value

        result = CorrelationDecision(
            **correlation_kwargs
        )

        

        db.add(
            result
        )

        _safe_commit(
            db
        )

        db.refresh(
            result
        )

        return result

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to persist correlation decision "
            "camera=%s track=%s candidate=%s",
            source_camera,
            source_track,
            candidate_global,
        )

        raise

    finally:

        db.close()


# ============================================================
# ACTIVITY EVENT
# ============================================================

def save_activity_event(
    event_id,
    camera_id,
    rule_id,
    event_type,
    severity,
    started_at,
    ended_at=None,
    track_id=None,
    global_person_id=None,
    global_vehicle_id=None,
    confidence=None,
    status="CONFIRMED",
    evidence=None,
    model_name=None,
    model_version=None,
    metadata=None,
):
    """
    Persist an activity event.

    event_id provides idempotency so retries do not create
    duplicate events.
    """

    event_id = _safe_string(
        event_id,
        name="event_id",
        max_length=MAX_EVENT_ID_LENGTH,
        required=True,
    )

    camera_id = _safe_string(
        camera_id,
        name="camera_id",
        max_length=MAX_CAMERA_ID_LENGTH,
        required=True,
    )

    rule_id = (
        _safe_string(
            rule_id,
            name="rule_id",
            max_length=MAX_RULE_ID_LENGTH,
        )
        or "UNKNOWN"
    )

    event_type = (
        _safe_string(
            event_type,
            name="event_type",
            max_length=MAX_EVENT_TYPE_LENGTH,
        )
        or "UNKNOWN"
    )

    severity = (
        _safe_string(
            severity,
            name="severity",
            max_length=MAX_SEVERITY_LENGTH,
        )
        or "LOW"
    ).upper()

    status = (
        _safe_string(
            status,
            name="status",
            max_length=MAX_STATUS_LENGTH,
        )
        or "CONFIRMED"
    ).upper()

    track_id = _safe_optional_id(
        track_id,
        name="track_id",
        max_length=MAX_TRACK_ID_LENGTH,
    )

    global_person_id = _safe_optional_id(
        global_person_id,
        name="global_person_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
    )

    global_vehicle_id = _safe_optional_id(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
    )

    confidence = _finite_float(
        confidence,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )

    started_at = normalize_datetime(
        started_at
    )

    if ended_at is not None:

        ended_at = normalize_datetime(
            ended_at
        )

        if ended_at < started_at:

            raise ValueError(
                "ended_at cannot be earlier than started_at"
            )

    evidence = (
        sanitize_metadata(
            evidence
        )
        if evidence is not None
        else None
    )

    metadata = sanitize_metadata(
        metadata
    )

    model_name = _safe_optional_id(
        model_name,
        name="model_name",
        max_length=MAX_MODEL_NAME_LENGTH,
    )

    model_version = _safe_optional_id(
        model_version,
        name="model_version",
        max_length=MAX_MODEL_VERSION_LENGTH,
    )

    db = SessionLocal()

    try:

        existing = (
            db.query(ActivityEvent)
            .filter(
                ActivityEvent.event_id
                == event_id
            )
            .first()
        )

        if existing:

            # Idempotent retry. Do not create
            # a second event.
            return existing

        event = ActivityEvent(
            event_id=event_id,
            camera_id=camera_id,
            track_id=track_id,
            global_person_id=global_person_id,
            global_vehicle_id=global_vehicle_id,
            rule_id=rule_id,
            event_type=event_type,
            severity=severity,
            confidence=confidence,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            evidence=evidence,
            model_name=model_name,
            model_version=model_version,
            metadata_json=metadata,
            created_at=utc_now(),
        )

        db.add(
            event
        )

        try:

            _safe_commit(
                db
            )

        except IntegrityError:

            db.rollback()

            # Another worker may have inserted
            # the same event concurrently.
            existing = (
                db.query(ActivityEvent)
                .filter(
                    ActivityEvent.event_id
                    == event_id
                )
                .first()
            )

            if existing:

                return existing

            raise

        db.refresh(
            event
        )

        return event

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to persist activity event %s",
            event_id,
        )

        raise

    finally:

        db.close()


# ============================================================
# MODEL EXECUTION
# ============================================================

def save_model_execution(
    model_name: str,
    model_version: str,
    model_hash: Optional[str] = None,
    camera_id: Optional[str] = None,
    track_id: Optional[str] = None,
    inference_timestamp: Optional[
        datetime
    ] = None,
    frame_timestamp: Optional[
        datetime
    ] = None,
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
    device: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
):
    """
    Persist model/inference execution telemetry.

    This complements the observation tables and allows later
    audit of model/version/latency.
    """

    # Import locally so deployments that do not currently
    # use ModelExecution don't fail at module import time.
    from db.intelligence_model import (
        ModelExecution,
    )

    model_name = _safe_string(
        model_name,
        name="model_name",
        max_length=MAX_MODEL_NAME_LENGTH,
        required=True,
    )

    model_version = _safe_string(
        model_version,
        name="model_version",
        max_length=MAX_MODEL_VERSION_LENGTH,
        required=True,
    )

    model_hash = _safe_optional_id(
        model_hash,
        name="model_hash",
        max_length=128,
    )

    camera_id = _safe_optional_id(
        camera_id,
        name="camera_id",
        max_length=MAX_CAMERA_ID_LENGTH,
    )

    track_id = _safe_optional_id(
        track_id,
        name="track_id",
        max_length=MAX_TRACK_ID_LENGTH,
    )

    inference_timestamp = normalize_datetime(
        inference_timestamp
    )

    frame_timestamp = (
        normalize_datetime(
            frame_timestamp
        )
        if frame_timestamp is not None
        else None
    )

    confidence = _finite_float(
        confidence,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )

    latency_ms = _finite_float(
        latency_ms,
        name="latency_ms",
        minimum=0.0,
        maximum=3600000.0,
    )

    device = _safe_optional_id(
        device,
        name="device",
        max_length=50,
    )

    metadata = sanitize_metadata(
        metadata
    )

    db = SessionLocal()

    try:

        record = ModelExecution(
            model_name=model_name,
            model_version=model_version,
            model_hash=model_hash,
            camera_id=camera_id,
            track_id=track_id,
            inference_timestamp=inference_timestamp,
            frame_timestamp=frame_timestamp,
            confidence=confidence,
            latency_ms=latency_ms,
            device=device,
            metadata_json=metadata,
            created_at=utc_now(),
        )

        db.add(
            record
        )

        _safe_commit(
            db
        )

        db.refresh(
            record
        )

        return record

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to persist model execution "
            "model=%s version=%s",
            model_name,
            model_version,
        )

        raise

    finally:

        db.close()


# ============================================================
# GET ACTIVITY EVENT
# ============================================================

def get_activity_event(
    event_id: str,
):
    if not event_id:
        return None

    event_id = _safe_string(
        event_id,
        name="event_id",
        max_length=MAX_EVENT_ID_LENGTH,
        required=True,
    )

    db = SessionLocal()

    try:

        return (
            db.query(
                ActivityEvent
            )
            .filter(
                ActivityEvent.event_id
                == event_id
            )
            .first()
        )

    finally:

        db.close()


# ============================================================
# GET VEHICLE OBSERVATIONS
# ============================================================

def get_vehicle_observations(
    global_vehicle_id: str,
    limit: int = 100,
):
    if not global_vehicle_id:
        return []

    global_id = _safe_string(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
        required=True,
    )

    limit = max(
        1,
        min(
            int(limit),
            MAX_QUERY_LIMIT,
        ),
    )

    db = SessionLocal()

    try:

        return (
            db.query(
                VehicleObservation
            )
            .filter(
                VehicleObservation.global_vehicle_id
                == global_id
            )
            .order_by(
                VehicleObservation.frame_timestamp.desc()
            )
            .limit(
                limit
            )
            .all()
        )

    finally:

        db.close()


# ============================================================
# GET PLATE OBSERVATIONS
# ============================================================

def get_plate_observations(
    normalized_plate: str,
    limit: int = 100,
):
    normalized = normalize_plate(
        normalized_plate
    )

    if not normalized:
        return []

    limit = max(
        1,
        min(
            int(limit),
            MAX_QUERY_LIMIT,
        ),
    )

    db = SessionLocal()

    try:

        return (
            db.query(
                PlateObservation
            )
            .filter(
                PlateObservation.normalized_plate
                == normalized
            )
            .order_by(
                PlateObservation.frame_timestamp.desc()
            )
            .limit(
                limit
            )
            .all()
        )

    finally:

        db.close()


# ============================================================
# GET CORRELATION HISTORY
# ============================================================

def get_correlation_history(
    global_vehicle_id: Optional[str] = None,
    decision: Optional[str] = None,
    limit: int = 100,
):
    limit = max(
        1,
        min(
            int(limit),
            MAX_QUERY_LIMIT,
        ),
    )

    db = SessionLocal()

    try:

        query = db.query(
            CorrelationDecision
        )

        if global_vehicle_id:

            global_id = _safe_string(
                global_vehicle_id,
                name="global_vehicle_id",
                max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
                required=True,
            )

            query = query.filter(
                (
                    CorrelationDecision
                    .candidate_global_vehicle_id
                    == global_id
                )
                |
                (
                    CorrelationDecision
                    .source_global_vehicle_id
                    == global_id
                )
            )

        if decision:

            normalized_decision = (
                _safe_string(
                    decision,
                    name="decision",
                    max_length=30,
                    required=True,
                )
                .upper()
            )

            query = query.filter(
                CorrelationDecision.decision
                == normalized_decision
            )

        return (
            query
            .order_by(
                CorrelationDecision.created_at.desc()
            )
            .limit(
                limit
            )
            .all()
        )

    finally:

        db.close()


# ============================================================
# GIS / CAMERA CONNECTION
# ============================================================

def _validate_nonnegative(
    value: Any,
    name: str,
) -> Optional[float]:

    return _finite_float(
        value,
        name=name,
        minimum=0.0,
        maximum=1e9,
    )


def save_camera_connection(
    source_camera_id: str,
    destination_camera_id: str,
    distance_km: Optional[float] = None,
    expected_min_seconds: Optional[float] = None,
    expected_max_seconds: Optional[float] = None,
    expected_seconds: Optional[float] = None,
    road_name: Optional[str] = None,
    direction: Optional[float] = None,
    direction_tolerance: Optional[float] = None,
    road_distance_km: Optional[float] = None,
    road_duration_seconds: Optional[float] = None,
    route_geometry: Optional[str] = None,
    routing_provider: Optional[str] = None,
    confidence: Optional[float] = None,
    active: bool = True,
):
    """
    Persist/update camera topology.

    Camera topology is separate from individual vehicle evidence.
    """

    source = _safe_string(
        source_camera_id,
        name="source_camera_id",
        max_length=255,
        required=True,
    )

    destination = _safe_string(
        destination_camera_id,
        name="destination_camera_id",
        max_length=255,
        required=True,
    )

    if source == destination:

        raise ValueError(
            "source_camera_id and destination_camera_id "
            "cannot be the same"
        )

    distance_km = _validate_nonnegative(
        distance_km,
        "distance_km",
    )

    expected_min_seconds = (
        _validate_nonnegative(
            expected_min_seconds,
            "expected_min_seconds",
        )
    )

    expected_max_seconds = (
        _validate_nonnegative(
            expected_max_seconds,
            "expected_max_seconds",
        )
    )

    expected_seconds = (
        _validate_nonnegative(
            expected_seconds,
            "expected_seconds",
        )
    )

    road_distance_km = (
        _validate_nonnegative(
            road_distance_km,
            "road_distance_km",
        )
    )

    road_duration_seconds = (
        _validate_nonnegative(
            road_duration_seconds,
            "road_duration_seconds",
        )
    )

    if (
        expected_min_seconds is not None
        and expected_max_seconds is not None
        and expected_min_seconds
        > expected_max_seconds
    ):

        raise ValueError(
            "expected_min_seconds cannot exceed "
            "expected_max_seconds"
        )

    if (
        expected_seconds is not None
        and expected_min_seconds is not None
        and expected_seconds
        < expected_min_seconds
    ):

        raise ValueError(
            "expected_seconds cannot be below "
            "expected_min_seconds"
        )

    if (
        expected_seconds is not None
        and expected_max_seconds is not None
        and expected_seconds
        > expected_max_seconds
    ):

        raise ValueError(
            "expected_seconds cannot exceed "
            "expected_max_seconds"
        )

    direction = _finite_float(
        direction,
        name="direction",
        minimum=-360.0,
        maximum=360.0,
    )

    direction_tolerance = _finite_float(
        direction_tolerance,
        name="direction_tolerance",
        minimum=0.0,
        maximum=360.0,
    )

    confidence = _finite_float(
        confidence,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )

    road_name = _safe_optional_id(
        road_name,
        name="road_name",
        max_length=255,
    )

    route_geometry = _safe_optional_id(
        route_geometry,
        name="route_geometry",
        max_length=100000,
    )

    routing_provider = _safe_optional_id(
        routing_provider,
        name="routing_provider",
        max_length=100,
    )

    db = SessionLocal()

    try:

        connection = (
            db.query(
                CameraConnection
            )
            .filter(
                CameraConnection.source_camera_id
                == source,
                CameraConnection.destination_camera_id
                == destination,
            )
            .first()
        )

        now = utc_now()

        if connection:

            connection.distance_km = (
                distance_km
            )

            connection.expected_min_seconds = (
                expected_min_seconds
            )

            connection.expected_max_seconds = (
                expected_max_seconds
            )

            connection.expected_seconds = (
                expected_seconds
            )

            connection.road_name = (
                road_name
            )

            connection.direction = (
                direction
            )

            connection.direction_tolerance = (
                direction_tolerance
            )

            connection.road_distance_km = (
                road_distance_km
            )

            connection.road_duration_seconds = (
                road_duration_seconds
            )

            connection.route_geometry = (
                route_geometry
            )

            connection.routing_provider = (
                routing_provider
            )

            connection.confidence = (
                confidence
            )

            connection.active = bool(
                active
            )

            connection.updated_at = now

        else:

            connection = CameraConnection(
                source_camera_id=source,
                destination_camera_id=destination,
                distance_km=distance_km,
                expected_min_seconds=(
                    expected_min_seconds
                ),
                expected_max_seconds=(
                    expected_max_seconds
                ),
                expected_seconds=(
                    expected_seconds
                ),
                road_name=road_name,
                direction=direction,
                direction_tolerance=(
                    direction_tolerance
                ),
                road_distance_km=(
                    road_distance_km
                ),
                road_duration_seconds=(
                    road_duration_seconds
                ),
                route_geometry=route_geometry,
                routing_provider=routing_provider,
                confidence=confidence,
                active=bool(
                    active
                ),
                created_at=now,
                updated_at=now,
            )

            db.add(
                connection
            )

        try:

            _safe_commit(
                db
            )

        except IntegrityError:

            db.rollback()

            # Unique source/destination constraint can
            # race between multiple GIS workers.
            connection = (
                db.query(
                    CameraConnection
                )
                .filter(
                    CameraConnection.source_camera_id
                    == source,
                    CameraConnection.destination_camera_id
                    == destination,
                )
                .first()
            )

            if connection is None:

                raise

            connection.distance_km = (
                distance_km
            )

            connection.expected_min_seconds = (
                expected_min_seconds
            )

            connection.expected_max_seconds = (
                expected_max_seconds
            )

            connection.expected_seconds = (
                expected_seconds
            )

            connection.road_name = (
                road_name
            )

            connection.direction = (
                direction
            )

            connection.direction_tolerance = (
                direction_tolerance
            )

            connection.road_distance_km = (
                road_distance_km
            )

            connection.road_duration_seconds = (
                road_duration_seconds
            )

            connection.route_geometry = (
                route_geometry
            )

            connection.routing_provider = (
                routing_provider
            )

            connection.confidence = (
                confidence
            )

            connection.active = bool(
                active
            )

            connection.updated_at = utc_now()

            _safe_commit(
                db
            )

        db.refresh(
            connection
        )

        return connection

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to persist camera connection "
            "%s -> %s",
            source,
            destination,
        )

        raise

    finally:

        db.close()


# ============================================================
# GET CAMERA CONNECTION
# ============================================================

def get_camera_connection(
    source_camera_id: str,
    destination_camera_id: str,
):
    if (
        not source_camera_id
        or not destination_camera_id
    ):

        return None

    source = _safe_string(
        source_camera_id,
        name="source_camera_id",
        max_length=255,
        required=True,
    )

    destination = _safe_string(
        destination_camera_id,
        name="destination_camera_id",
        max_length=255,
        required=True,
    )

    db = SessionLocal()

    try:

        return (
            db.query(
                CameraConnection
            )
            .filter(
                CameraConnection.source_camera_id
                == source,
                CameraConnection.destination_camera_id
                == destination,
            )
            .first()
        )

    finally:

        db.close()


# ============================================================
# GET CAMERA CONNECTIONS
# ============================================================

def get_camera_connections(
    camera_id: str,
    active_only: bool = True,
):
    if not camera_id:
        return []

    camera = _safe_string(
        camera_id,
        name="camera_id",
        max_length=255,
        required=True,
    )

    db = SessionLocal()

    try:

        query = (
            db.query(
                CameraConnection
            )
            .filter(
                CameraConnection.source_camera_id
                == camera
            )
        )

        if active_only:

            query = query.filter(
                CameraConnection.active.is_(True)
            )

        return (
            query
            .order_by(
                CameraConnection.confidence.desc()
            )
            .limit(
                MAX_QUERY_LIMIT
            )
            .all()
        )

    finally:

        db.close()


# ============================================================
# GET CAMERA NEIGHBORS
# ============================================================

def get_camera_neighbors(
    camera_id: str,
    active_only: bool = True,
):
    return get_camera_connections(
        camera_id,
        active_only=active_only,
    )


# ============================================================
# DEACTIVATE CAMERA CONNECTION
# ============================================================

def deactivate_camera_connection(
    source_camera_id: str,
    destination_camera_id: str,
):
    source = _safe_string(
        source_camera_id,
        name="source_camera_id",
        max_length=255,
        required=True,
    )

    destination = _safe_string(
        destination_camera_id,
        name="destination_camera_id",
        max_length=255,
        required=True,
    )

    db = SessionLocal()

    try:

        connection = (
            db.query(
                CameraConnection
            )
            .filter(
                CameraConnection.source_camera_id
                == source,
                CameraConnection.destination_camera_id
                == destination,
            )
            .first()
        )

        if connection is None:

            return None

        connection.active = False
        connection.updated_at = utc_now()

        _safe_commit(
            db
        )

        db.refresh(
            connection
        )

        return connection

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to deactivate camera connection "
            "%s -> %s",
            source,
            destination,
        )

        raise

    finally:

        db.close()


# ============================================================
# GET VEHICLE BY PLATE
# ============================================================

def get_plate_vehicle_candidates(
    normalized_plate: str,
    limit: int = 100,
):
    """
    Return distinct vehicle IDs associated with a plate.

    This function intentionally does NOT say that the plate alone
    proves vehicle identity. It only returns evidence candidates.
    """

    normalized = normalize_plate(
        normalized_plate
    )

    if not normalized:

        return []

    limit = max(
        1,
        min(
            int(limit),
            MAX_QUERY_LIMIT,
        ),
    )

    db = SessionLocal()

    try:

        rows = (
            db.query(
                PlateObservation
            )
            .filter(
                PlateObservation.normalized_plate
                == normalized,
                PlateObservation.vehicle_id
                .is_not(None),
            )
            .order_by(
                PlateObservation.frame_timestamp.desc()
            )
            .limit(
                limit
            )
            .all()
        )

        vehicle_ids = []

        seen = set()

        for row in rows:

            if (
                row.vehicle_id is None
            ):

                continue

            vehicle_id = int(
                row.vehicle_id
            )

            if vehicle_id in seen:
                continue

            seen.add(
                vehicle_id
            )

            vehicle_ids.append(
                vehicle_id
            )

        return vehicle_ids

    finally:

        db.close()


# ============================================================
# VEHICLE EMBEDDING HISTORY
# ============================================================

def get_vehicle_embeddings(
    global_vehicle_id: str,
    limit: int = 100,
):
    if not global_vehicle_id:

        return []

    global_id = _safe_string(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
        required=True,
    )

    limit = max(
        1,
        min(
            int(limit),
            MAX_QUERY_LIMIT,
        ),
    )

    db = SessionLocal()

    try:

        vehicle = (
            db.query(
                Vehicle
            )
            .filter(
                Vehicle.global_vehicle_id
                == global_id
            )
            .first()
        )

        if vehicle is None:

            return []

        return (
            db.query(
                VehicleEmbedding
            )
            .filter(
                VehicleEmbedding.vehicle_id
                == vehicle.id
            )
            .order_by(
                VehicleEmbedding.created_at.desc()
            )
            .limit(
                limit
            )
            .all()
        )

    finally:

        db.close()


# ============================================================
# VEHICLE BY PRIMARY ID
# ============================================================

def get_vehicle_by_id(
    vehicle_id: int,
):
    try:

        vehicle_id = int(
            vehicle_id
        )

    except (
        TypeError,
        ValueError,
    ):

        raise ValueError(
            "vehicle_id must be an integer"
        )

    db = SessionLocal()

    try:

        return (
            db.query(
                Vehicle
            )
            .filter(
                Vehicle.id
                == vehicle_id
            )
            .first()
        )

    finally:

        db.close()


# ============================================================
# VEHICLE OBSERVATION COUNT
# ============================================================

def get_vehicle_observation_count(
    global_vehicle_id: str,
) -> int:

    if not global_vehicle_id:

        return 0

    global_id = _safe_string(
        global_vehicle_id,
        name="global_vehicle_id",
        max_length=MAX_GLOBAL_VEHICLE_ID_LENGTH,
        required=True,
    )

    db = SessionLocal()

    try:

        return int(
            db.query(
                VehicleObservation
            )
            .filter(
                VehicleObservation.global_vehicle_id
                == global_id
            )
            .count()
        )

    finally:

        db.close()


# ============================================================
# HEALTH CHECK
# ============================================================

def intelligence_persistence_health() -> dict[str, Any]:
    """
    Lightweight database health check.

    Does not expose connection credentials or sensitive
    configuration.
    """

    db = SessionLocal()

    try:

        db.execute(
            sql_text("SELECT 1")
        )

        return {
            "healthy": True,
            "database": "reachable",
            "sqlalchemy": "2.x-compatible",
            "timestamp": utc_now().isoformat(),
        }

    except Exception as exc:

        logger.exception(
            "Intelligence persistence health check failed"
        )

        return {
            "healthy": False,
            "database": "unreachable",
            "error_type": type(
                exc
            ).__name__,
            "timestamp": utc_now().isoformat(),
        }

    finally:

        db.close()


# ============================================================
# MODULE SELF-CHECK
# ============================================================

def validate_persistence_configuration() -> dict[str, Any]:
    """
    Validate local persistence configuration without writing data.
    """

    errors = []

    if (
        MIN_EMBEDDING_DIMENSION
        <= 0
    ):

        errors.append(
            "MIN_EMBEDDING_DIMENSION must be positive"
        )

    if (
        MAX_EMBEDDING_DIMENSION
        < MIN_EMBEDDING_DIMENSION
    ):

        errors.append(
            "MAX_EMBEDDING_DIMENSION is invalid"
        )

    if (
        MAX_METADATA_DEPTH
        <= 0
    ):

        errors.append(
            "MAX_METADATA_DEPTH must be positive"
        )

    if (
        MAX_QUERY_LIMIT
        <= 0
    ):

        errors.append(
            "MAX_QUERY_LIMIT must be positive"
        )

    if MAX_EMBEDDING_BYTES != MAX_EMBEDDING_DIMENSION * 4:
        errors.append(
            "MAX_EMBEDDING_BYTES must equal "
            "MAX_EMBEDDING_DIMENSION * 4"
        )

    return {
        "valid":
            not errors,

        "errors":
            errors,

        "max_embedding_bytes":
            MAX_EMBEDDING_BYTES,

        "max_embedding_dimension":
            MAX_EMBEDDING_DIMENSION,

        "min_embedding_dimension":
            MIN_EMBEDDING_DIMENSION,

        "max_query_limit":
            MAX_QUERY_LIMIT,

        "max_metadata_depth":
            MAX_METADATA_DEPTH,
    }

# ============================================================
# GENERIC TRANSACTION HELPER
# ============================================================

def run_persistence_transaction(
    operation,
):
    """
    Execute a caller-supplied database operation in one managed
    SQLAlchemy session.

    `operation` receives the Session and must return its result.

    The helper is intentionally small: it centralizes commit/rollback
    behavior for future persistence operations without replacing the
    existing public save_* functions.
    """

    if not callable(operation):
        raise ValueError(
            "operation must be callable"
        )

    db = SessionLocal()

    try:
        result = operation(db)
        db.commit()
        return result

    except Exception:
        try:
            db.rollback()
        except Exception:
            logger.exception(
                "Database rollback failed in transaction helper"
            )
        raise

    finally:
        db.close()

