from __future__ import annotations

import copy
import logging
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from vehicleReid import cosine_similarity


logger = logging.getLogger(__name__)


# ============================================================
# OPTIONAL PERSISTENCE
# ============================================================

try:
    from services.intelligence_persistence import (
        save_correlation_decision,
        save_activity_event,
    )
except ImportError:
    save_correlation_decision = None
    save_activity_event = None


# ============================================================
# OPTIONAL GIS
# ============================================================

try:
    from cameraConfig import get_camera_gis
except ImportError:
    get_camera_gis = None


try:
    from services.gis_engine import evaluate_vehicle_transition
except ImportError:
    try:
        from sservices.gis_engine import evaluate_vehicle_transition
    except ImportError:
        evaluate_vehicle_transition = None


# ============================================================
# OPTIONAL POSTGIS CANDIDATE GATE
# ============================================================

try:
    from db.database import SessionLocal
except ImportError:
    SessionLocal = None

try:
    from sqlalchemy import text as sql_text
except ImportError:
    sql_text = None

try:
    from services.postgisSpatial import (
        SpatialError,
        SpatialUnavailableError,
        get_postgis_spatial_service,
    )
except ImportError:
    SpatialError = Exception
    SpatialUnavailableError = Exception
    get_postgis_spatial_service = None


# ============================================================
# CONFIGURATION
# ============================================================

REID_FRAME_THRESHOLD = float(
    os.getenv(
        "REID_FRAME_THRESHOLD",
        "0.78",
    )
)

REID_SUPPORT_THRESHOLD = float(
    os.getenv(
        "REID_SUPPORT_THRESHOLD",
        "0.80",
    )
)

REID_STRONG_THRESHOLD = float(
    os.getenv(
        "REID_STRONG_THRESHOLD",
        "0.88",
    )
)

REID_STRONG_CONSENSUS = float(
    os.getenv(
        "REID_STRONG_CONSENSUS",
        "0.84",
    )
)

MIN_SUPPORTING_EMBEDDINGS = max(
    1,
    int(
        os.getenv(
            "MIN_SUPPORTING_EMBEDDINGS",
            "3",
        )
    ),
)

MIN_STRONG_SUPPORTING_EMBEDDINGS = max(
    MIN_SUPPORTING_EMBEDDINGS,
    int(
        os.getenv(
            "MIN_STRONG_SUPPORTING_EMBEDDINGS",
            "4",
        )
    ),
)

MAX_SIGHTING_GAP_SECONDS = max(
    1.0,
    float(
        os.getenv(
            "MAX_SIGHTING_GAP_SECONDS",
            str(15 * 60),
        )
    ),
)

MIN_TRAVEL_TIME_SECONDS = max(
    0.0,
    float(
        os.getenv(
            "MIN_TRAVEL_TIME_SECONDS",
            "0.0",
        )
    ),
)

MAX_EMBEDDINGS_PER_GLOBAL_VEHICLE = max(
    3,
    int(
        os.getenv(
            "MAX_EMBEDDINGS_PER_GLOBAL_VEHICLE",
            "20",
        )
    ),
)

MAX_LOCAL_EMBEDDINGS = max(
    3,
    int(
        os.getenv(
            "MAX_LOCAL_EMBEDDINGS",
            "10",
        )
    ),
)

MAX_GLOBAL_VEHICLES = max(
    100,
    int(
        os.getenv(
            "MAX_GLOBAL_VEHICLES",
            "10000",
        )
    ),
)

LOCAL_TRACK_TIMEOUT_SECONDS = max(
    1.0,
    float(
        os.getenv(
            "LOCAL_TRACK_TIMEOUT_SECONDS",
            "30",
        )
    ),
)

MAX_SIGHTINGS_PER_GLOBAL_VEHICLE = max(
    10,
    int(
        os.getenv(
            "MAX_SIGHTINGS_PER_GLOBAL_VEHICLE",
            "100",
        )
    ),
)


# ============================================================
# CORRELATION WEIGHTS
# ============================================================

REID_WEIGHT = float(
    os.getenv(
        "CORRELATION_REID_WEIGHT",
        "0.50",
    )
)

PLATE_WEIGHT = float(
    os.getenv(
        "CORRELATION_PLATE_WEIGHT",
        "0.15",
    )
)

GIS_WEIGHT = float(
    os.getenv(
        "CORRELATION_GIS_WEIGHT",
        "0.10",
    )
)

TEMPORAL_WEIGHT = float(
    os.getenv(
        "CORRELATION_TEMPORAL_WEIGHT",
        "0.10",
    )
)

TYPE_WEIGHT = float(
    os.getenv(
        "CORRELATION_TYPE_WEIGHT",
        "0.05",
    )
)

TRACK_WEIGHT = float(
    os.getenv(
        "CORRELATION_TRACK_WEIGHT",
        "0.10",
    )
)

MIN_TRACK_QUALITY_FOR_CORRELATION = float(
    os.getenv(
        "MIN_TRACK_QUALITY_FOR_CORRELATION",
        "0.35",
    )
)

HARD_REJECT_LOW_TRACK_QUALITY = float(
    os.getenv(
        "HARD_REJECT_LOW_TRACK_QUALITY",
        "0.15",
    )
)

CORRELATION_MATCH_THRESHOLD = float(
    os.getenv(
        "CORRELATION_MATCH_THRESHOLD",
        "0.72",
    )
)

CORRELATION_STRONG_THRESHOLD = float(
    os.getenv(
        "CORRELATION_STRONG_THRESHOLD",
        "0.84",
    )
)

CORRELATION_MARGIN_THRESHOLD = float(
    os.getenv(
        "CORRELATION_MARGIN_THRESHOLD",
        "0.08",
    )
)


# ============================================================
# SAME-CAMERA TRACK RECOVERY
# ============================================================

SAME_CAMERA_MIN_GAP_SECONDS = max(
    0.0,
    float(
        os.getenv(
            "SAME_CAMERA_MIN_GAP_SECONDS",
            "0.50",
        )
    ),
)

SAME_CAMERA_MAX_GAP_SECONDS = max(
    SAME_CAMERA_MIN_GAP_SECONDS,
    float(
        os.getenv(
            "SAME_CAMERA_MAX_GAP_SECONDS",
            "20.0",
        )
    ),
)

SAME_CAMERA_REID_THRESHOLD = float(
    os.getenv(
        "SAME_CAMERA_REID_THRESHOLD",
        "0.86",
    )
)

SAME_CAMERA_STRONG_THRESHOLD = float(
    os.getenv(
        "SAME_CAMERA_STRONG_THRESHOLD",
        "0.92",
    )
)

SAME_CAMERA_MIN_SUPPORT = max(
    1,
    int(
        os.getenv(
            "SAME_CAMERA_MIN_SUPPORT",
            "3",
        )
    ),
)

SAME_CAMERA_MIN_CONSISTENCY = float(
    os.getenv(
        "SAME_CAMERA_MIN_CONSISTENCY",
        "0.50",
    )
)

SAME_CAMERA_STRONG_CONSISTENCY = float(
    os.getenv(
        "SAME_CAMERA_STRONG_CONSISTENCY",
        "0.60",
    )
)

MAX_RECENT_LOCAL_TRACKS = max(
    100,
    int(
        os.getenv(
            "MAX_RECENT_LOCAL_TRACKS",
            "5000",
        )
    ),
)

RECENT_LOCAL_TRACK_TTL_SECONDS = max(
    1.0,
    float(
        os.getenv(
            "RECENT_LOCAL_TRACK_TTL_SECONDS",
            "30.0",
        )
    ),
)


# ============================================================
# ANPR / PLATE EVIDENCE
# ============================================================

PLATE_MIN_CONFIDENCE = float(
    os.getenv(
        "PLATE_MIN_CONFIDENCE",
        "0.80",
    )
)

PLATE_STRONG_CONFIDENCE = float(
    os.getenv(
        "PLATE_STRONG_CONFIDENCE",
        "0.92",
    )
)

PLATE_MIN_SUPPORT = max(
    1,
    int(
        os.getenv(
            "PLATE_MIN_SUPPORT",
            "2",
        )
    ),
)

MAX_PLATE_OBSERVATIONS = max(
    5,
    int(
        os.getenv(
            "MAX_PLATE_OBSERVATIONS",
            "30",
        )
    ),
)

PLATE_MIN_LENGTH = max(
    1,
    int(
        os.getenv(
            "PLATE_MIN_LENGTH",
            "4",
        )
    ),
)

PLATE_MAX_LENGTH = max(
    PLATE_MIN_LENGTH,
    int(
        os.getenv(
            "PLATE_MAX_LENGTH",
            "12",
        )
    ),
)

PLATE_ALLOWED_PATTERN = re.compile(
    r"^[A-Z0-9]+$"
)


# ============================================================
# JOURNEY / GIS LIMITS
# ============================================================

MAX_JOURNEY_EVENTS_PER_VEHICLE = max(
    20,
    int(
        os.getenv(
            "MAX_JOURNEY_EVENTS_PER_VEHICLE",
            "200",
        )
    ),
)

MAX_GIS_JOURNEY_POINTS = max(
    20,
    int(
        os.getenv(
            "MAX_GIS_JOURNEY_POINTS",
            "200",
        )
    ),
)

MAX_GIS_CAMERAS_PER_VEHICLE = max(
    10,
    int(
        os.getenv(
            "MAX_GIS_CAMERAS_PER_VEHICLE",
            "100",
        )
    ),
)

MAX_GIS_CAMERA_ID_LENGTH = 128
MAX_GIS_ZONE_LENGTH = 128

EARTH_RADIUS_METERS = 6371008.8

GIS_CORRELATION_WEIGHT = max(
    0.0,
    min(
        1.0,
        GIS_WEIGHT,
    ),
)

GIS_MIN_ACCEPTABLE_SCORE = float(
    os.getenv(
        "GIS_MIN_ACCEPTABLE_SCORE",
        "0.20",
    )
)

POSTGIS_VEHICLE_GATE_ENABLED = (
    os.getenv(
        "POSTGIS_VEHICLE_GATE_ENABLED",
        "true",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

POSTGIS_VEHICLE_GATE_FAIL_OPEN = (
    os.getenv(
        "POSTGIS_VEHICLE_GATE_FAIL_OPEN",
        "true",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH = max(
    1.0,
    float(
        os.getenv(
            "POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH",
            os.getenv(
                "GIS_MAX_REASONABLE_SPEED_KMH",
                "180",
            ),
        )
    ),
)

POSTGIS_VEHICLE_GATE_MAX_TRANSITION_SECONDS = max(
    1.0,
    float(
        os.getenv(
            "POSTGIS_VEHICLE_GATE_MAX_TRANSITION_SECONDS",
            os.getenv(
                "GIS_MAX_TRANSITION_SECONDS",
                "3600",
            ),
        )
    ),
)

POSTGIS_VEHICLE_GATE_MAX_DISTANCE_KM = max(
    0.001,
    float(
        os.getenv(
            "POSTGIS_VEHICLE_GATE_MAX_DISTANCE_KM",
            os.getenv(
                "GIS_MAX_CAMERA_DISTANCE_KM",
                "100",
            ),
        )
    ),
)


# ============================================================
# JOURNEY SIMILARITY LABELS
# ============================================================

JOURNEY_WEAK_SIMILARITY_THRESHOLD = 0.78
JOURNEY_STRONG_SIMILARITY_THRESHOLD = 0.88


# ============================================================
# INTERNAL STATE
# ============================================================

_correlation_lock = threading.RLock()

_global_vehicles: dict[
    str,
    dict[str, Any],
] = {}

_local_to_global: dict[
    str,
    str,
] = {}

_local_track_evidence: dict[
    str,
    dict[str, Any],
] = {}


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def _clamp01(
    value: Any,
) -> float:

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return 0.0

    if not np.isfinite(number):
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            number,
        ),
    )


def _safe_float(
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

    if not np.isfinite(number):
        return default

    return number


def _safe_timestamp(
    value: Any,
) -> float:

    timestamp = _safe_float(
        value,
        time.time(),
    )

    if timestamp <= 0.0:
        return time.time()

    return timestamp


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def _normalize_embedding(
    embedding: Any,
) -> Optional[list[float]]:

    if embedding is None:
        return None

    try:

        array = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if array.size == 0:
        return None

    if not np.all(
        np.isfinite(array)
    ):
        return None

    norm = float(
        np.linalg.norm(array)
    )

    if (
        not np.isfinite(norm)
        or norm <= 1e-12
    ):
        return None

    normalized = (
        array / norm
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.all(
        np.isfinite(normalized)
    ):
        return None

    return normalized.tolist()



def _embedding_quality(
    embedding: Any,
) -> bool:

    normalized = _normalize_embedding(
        embedding
    )

    if normalized is None:
        return False

    # Existing INTEL-I Re-ID model is 512-dimensional.
    # Keep the validator tolerant for future model upgrades.
    return len(normalized) >= 32


# ============================================================
# VEHICLE TYPE COMPATIBILITY
# ============================================================

def _normalize_vehicle_type(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    value = str(
        value
    ).strip().lower()

    if not value:
        return None

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value or None


def _vehicle_type_compatible(
    current_type: Any,
    stored_type: Any,
) -> bool:

    current = _normalize_vehicle_type(
        current_type
    )

    stored = _normalize_vehicle_type(
        stored_type
    )

    if (
        current is None
        or stored is None
    ):
        return True

    if current == stored:
        return True

    aliases = {
        "car": {
            "car",
            "sedan",
            "hatchback",
            "suv",
            "vehicle",
        },
        "suv": {
            "suv",
            "car",
            "vehicle",
        },
        "motorcycle": {
            "motorcycle",
            "motorbike",
            "bike",
            "vehicle",
        },
        "motorbike": {
            "motorcycle",
            "motorbike",
            "bike",
            "vehicle",
        },
        "truck": {
            "truck",
            "lorry",
            "vehicle",
        },
        "bus": {
            "bus",
            "vehicle",
        },
        "vehicle": {
            "vehicle",
            "car",
            "suv",
            "sedan",
            "hatchback",
            "motorcycle",
            "motorbike",
            "bike",
            "truck",
            "lorry",
            "bus",
        },
    }

    current_group = aliases.get(
        current,
        {current},
    )

    return stored in current_group


# ============================================================
# PLATE NORMALIZATION
# ============================================================

def _normalize_plate_text(
    plate: Any,
) -> Optional[str]:

    if plate is None:
        return None

    if not isinstance(
        plate,
        str,
    ):
        return None

    normalized = (
        plate
        .strip()
        .upper()
    )

    if not normalized:
        return None

    # Remove only common OCR formatting.
    normalized = re.sub(
        r"[\s\-_.:/\\|]+",
        "",
        normalized,
    )

    normalized = re.sub(
        r"[^A-Z0-9]",
        "",
        normalized,
    )

    if (
        len(normalized)
        < PLATE_MIN_LENGTH
    ):
        return None

    if (
        len(normalized)
        > PLATE_MAX_LENGTH
    ):
        return None

    if not PLATE_ALLOWED_PATTERN.fullmatch(
        normalized
    ):
        return None

    return normalized


# ============================================================
# VEHICLE PLATE EXTRACTION
# ============================================================

def _get_vehicle_plate(
    vehicle: dict[str, Any],
) -> tuple[
    Optional[str],
    float,
]:

    if not isinstance(
        vehicle,
        dict,
    ):
        return None, 0.0

    status = str(
        vehicle.get(
            "plate_status",
            "",
        )
    ).strip().upper()

    # --------------------------------------------------------
    # Prefer stable temporal ANPR evidence.
    # --------------------------------------------------------

    stable_plate = _normalize_plate_text(
        vehicle.get(
            "stable_plate"
        )
    )

    if (
        status == "STABLE"
        and stable_plate is not None
    ):

        confidence = _clamp01(
            vehicle.get(
                "stable_plate_confidence",
                vehicle.get(
                    "plate_confidence",
                    0.0,
                ),
            )
        )

        return (
            stable_plate,
            confidence,
        )

    # --------------------------------------------------------
    # Upstream temporal consensus may be exposed directly.
    # --------------------------------------------------------

    consensus = vehicle.get(
        "plate_consensus"
    )

    if isinstance(
        consensus,
        dict,
    ):

        consensus_status = str(
            consensus.get(
                "status",
                "",
            )
        ).upper()

        consensus_plate = (
            _normalize_plate_text(
                consensus.get(
                    "plate"
                )
            )
        )

        if (
            consensus_status == "STABLE"
            and consensus_plate is not None
        ):

            confidence = _clamp01(
                consensus.get(
                    "confidence",
                    vehicle.get(
                        "plate_confidence",
                        0.0,
                    ),
                )
            )

            return (
                consensus_plate,
                confidence,
            )

    # --------------------------------------------------------
    # Raw plate evidence.
    # --------------------------------------------------------

    plate = vehicle.get(
        "plate"
    )

    if plate is None:
        plate = vehicle.get(
            "raw_plate"
        )

    if plate is None:
        plate = vehicle.get(
            "license_plate"
        )

    if plate is None:
        plate = vehicle.get(
            "plate_text"
        )

    confidence = _clamp01(
        vehicle.get(
            "plate_confidence",
            vehicle.get(
                "raw_plate_confidence",
                vehicle.get(
                    "license_plate_confidence",
                    0.0,
                ),
            ),
        )
    )

    normalized = _normalize_plate_text(
        plate
    )

    if normalized is None:
        return None, 0.0

    return (
        normalized,
        confidence,
    )


# ============================================================
# PLATE OBSERVATION AGGREGATION
# ============================================================

def _record_plate_observation(
    global_vehicle: dict[str, Any],
    vehicle: dict[str, Any],
) -> Optional[dict[str, Any]]:

    if not isinstance(
        global_vehicle,
        dict,
    ):
        return None

    if not isinstance(
        vehicle,
        dict,
    ):
        return None

    plate, confidence = (
        _get_vehicle_plate(
            vehicle
        )
    )

    if (
        plate is None
        or confidence < PLATE_MIN_CONFIDENCE
    ):
        return None

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    observations = (
        global_vehicle.setdefault(
            "plate_observations",
            [],
        )
    )

    candidates = (
        global_vehicle.setdefault(
            "plate_candidates",
            {},
        )
    )

    # --------------------------------------------------------
    # Store bounded observation history.
    # --------------------------------------------------------

    observations.append(
        {
            "plate": plate,
            "confidence": confidence,
            "timestamp": timestamp,
            "camera_id": vehicle.get(
                "camera_id"
            ),
            "track_id": vehicle.get(
                "track_id"
            ),
        }
    )

    if len(observations) > MAX_PLATE_OBSERVATIONS:
        del observations[
            :-MAX_PLATE_OBSERVATIONS
        ]

    # --------------------------------------------------------
    # Weighted candidate accumulation.
    # --------------------------------------------------------

    candidate = candidates.get(
        plate
    )

    if candidate is None:

        candidate = {
            "count": 0,
            "confidence_sum": 0.0,
            "best_confidence": 0.0,
            "last_seen": timestamp,
        }

        candidates[plate] = candidate

    candidate["count"] += 1

    candidate["confidence_sum"] += (
        confidence
    )

    candidate["best_confidence"] = max(
        candidate["best_confidence"],
        confidence,
    )

    candidate["last_seen"] = timestamp

    # --------------------------------------------------------
    # Rank candidates.
    # --------------------------------------------------------

    ranked = []

    for candidate_plate, data in candidates.items():

        count = max(
            0,
            int(
                data.get(
                    "count",
                    0,
                )
            ),
        )

        confidence_sum = _safe_float(
            data.get(
                "confidence_sum",
                0.0,
            ),
            0.0,
        )

        mean_confidence = (
            confidence_sum
            / max(
                1,
                count,
            )
        )

        best_confidence = _clamp01(
            data.get(
                "best_confidence",
                0.0,
            )
        )

        # Repeated observations matter more than one high OCR score.
        support_factor = min(
            1.0,
            count / max(
                1,
                PLATE_MIN_SUPPORT,
            ),
        )

        score = (
            0.60 * mean_confidence
            + 0.25 * best_confidence
            + 0.15 * support_factor
        )

        ranked.append(
            (
                candidate_plate,
                score,
                count,
                mean_confidence,
                best_confidence,
            )
        )

    ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    if not ranked:
        return None

    best_plate = ranked[0][0]
    best_score = ranked[0][1]
    best_count = ranked[0][2]
    best_mean = ranked[0][3]
    best_confidence = ranked[0][4]

    runner_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    total_score = sum(
        item[1]
        for item in ranked
    )

    consensus = (
        best_score / total_score
        if total_score > 0.0
        else 0.0
    )

    margin = (
        (best_score - runner_score)
        / best_score
        if best_score > 0.0
        else 0.0
    )

    stable = (
        best_count
        >= PLATE_MIN_SUPPORT
        and best_mean
        >= PLATE_MIN_CONFIDENCE
        and consensus
        >= 0.55
        and margin
        >= 0.10
    )

    # --------------------------------------------------------
    # Establish stable plate only after repeated evidence.
    # --------------------------------------------------------

    if stable:

        global_vehicle[
            "plate"
        ] = best_plate

        global_vehicle[
            "plate_confidence"
        ] = _clamp01(
            best_mean
        )

        global_vehicle[
            "plate_support_count"
        ] = best_count

        if (
            best_confidence
            >= PLATE_STRONG_CONFIDENCE
            and best_count
            >= PLATE_MIN_SUPPORT
        ):

            global_vehicle[
                "plate_match_strength"
            ] = "strong"

        else:

            global_vehicle[
                "plate_match_strength"
            ] = "confirmed"

        global_vehicle[
            "plate_status"
        ] = "STABLE"

        global_vehicle[
            "plate_consensus_margin"
        ] = _clamp01(
            margin
        )

        global_vehicle[
            "plate_conflict_candidates"
        ] = [
            item[0]
            for item in ranked[1:4]
        ]

    else:

        global_vehicle[
            "plate_status"
        ] = "PROVISIONAL"

        global_vehicle[
            "plate_consensus_margin"
        ] = _clamp01(
            margin
        )

        global_vehicle[
            "plate_conflict_candidates"
        ] = [
            item[0]
            for item in ranked[1:4]
        ]

    return {
        "plate": best_plate,
        "support_count": best_count,
        "best_confidence": _clamp01(
            best_confidence
        ),
        "mean_confidence": _clamp01(
            best_mean
        ),
        "consensus": _clamp01(
            consensus
        ),
        "margin": _clamp01(
            margin
        ),
        "stable": stable,
    }


# ============================================================
# PLATE COMPATIBILITY
# ============================================================

def _plate_compatible(
    vehicle: dict[str, Any],
    global_vehicle: dict[str, Any],
) -> bool:

    current_plate, current_confidence = (
        _get_vehicle_plate(
            vehicle
        )
    )

    if (
        current_plate is None
        or current_confidence
        < PLATE_MIN_CONFIDENCE
    ):
        return True

    established_plate = (
        _normalize_plate_text(
            global_vehicle.get(
                "plate"
            )
        )
    )

    if established_plate is None:
        return True

    return (
        current_plate
        == established_plate
    )


def _plate_evidence_score(
    vehicle: dict[str, Any],
    global_vehicle: dict[str, Any],
) -> float:

    current_plate, current_confidence = (
        _get_vehicle_plate(
            vehicle
        )
    )

    if (
        current_plate is None
        or current_confidence
        < PLATE_MIN_CONFIDENCE
    ):
        return 0.0

    established_plate = (
        _normalize_plate_text(
            global_vehicle.get(
                "plate"
            )
        )
    )

    if established_plate is None:
        return 0.0

    if current_plate != established_plate:
        return 0.0

    established_confidence = _clamp01(
        global_vehicle.get(
            "plate_confidence",
            0.0,
        )
    )

    support_count = max(
        0,
        int(
            _safe_float(
                global_vehicle.get(
                    "plate_support_count",
                    0,
                ),
                0.0,
            )
        ),
    )

    support_factor = min(
        1.0,
        support_count
        / max(
            1,
            PLATE_MIN_SUPPORT,
        ),
    )

    return _clamp01(
        (
            0.55 * current_confidence
            + 0.25 * established_confidence
            + 0.20 * support_factor
        )
    )


# ============================================================
# LOCAL TRACK EVIDENCE
# ============================================================

def _cleanup_local_track_evidence(
    now: Optional[float] = None,
) -> None:

    if now is None:
        now = time.time()

    expired = []

    for (
        local_key,
        evidence,
    ) in _local_track_evidence.items():

        last_seen = _safe_float(
            evidence.get(
                "last_seen",
                0.0,
            ),
            0.0,
        )

        if (
            now - last_seen
            > RECENT_LOCAL_TRACK_TTL_SECONDS
        ):
            expired.append(
                local_key
            )

    for local_key in expired:
        _local_track_evidence.pop(
            local_key,
            None,
        )

    if (
        len(_local_track_evidence)
        <= MAX_RECENT_LOCAL_TRACKS
    ):
        return

    sortable = sorted(
        _local_track_evidence.items(),
        key=lambda item: _safe_float(
            item[1].get(
                "last_seen",
                0.0,
            ),
            0.0,
        ),
    )

    excess = (
        len(_local_track_evidence)
        - MAX_RECENT_LOCAL_TRACKS
    )

    for (
        local_key,
        _,
    ) in sortable[:excess]:

        _local_track_evidence.pop(
            local_key,
            None,
        )


def _add_local_embedding(
    vehicle: dict[str, Any],
    embedding: list[float],
) -> Optional[dict[str, Any]]:

    camera_id = vehicle.get(
        "camera_id"
    )

    track_id = vehicle.get(
        "track_id"
    )

    if (
        camera_id is None
        or track_id is None
    ):
        return None

    local_key = (
        f"{camera_id}:{track_id}"
    )

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    evidence = _local_track_evidence.get(
        local_key
    )

    if evidence is None:

        evidence = {
            "camera_id": camera_id,
            "track_id": track_id,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "embeddings": [],
            "timestamps": [],
        }

        _local_track_evidence[
            local_key
        ] = evidence

    evidence["last_seen"] = max(
        timestamp,
        _safe_float(
            evidence.get(
                "last_seen",
                timestamp,
            ),
            timestamp,
        ),
    )

    embeddings = evidence[
        "embeddings"
    ]

    normalized = _normalize_embedding(
        embedding
    )

    if normalized is None:
        return None

    # Avoid inserting essentially identical consecutive frames.
    if embeddings:

        similarity = _safe_float(
            cosine_similarity(
                normalized,
                embeddings[-1],
            ),
            0.0,
        )

        if similarity < 0.995:
            embeddings.append(
                normalized
            )

            evidence[
                "timestamps"
            ].append(
                timestamp
            )

    else:

        embeddings.append(
            normalized
        )

        evidence[
            "timestamps"
        ].append(
            timestamp
        )

    if (
        len(embeddings)
        > MAX_LOCAL_EMBEDDINGS
    ):

        del embeddings[
            :-MAX_LOCAL_EMBEDDINGS
        ]

        if len(
            evidence["timestamps"]
        ) > MAX_LOCAL_EMBEDDINGS:

            del evidence[
                "timestamps"
            ][
                :-MAX_LOCAL_EMBEDDINGS
            ]

    _cleanup_local_track_evidence(
        timestamp
    )

    return {
        "camera_id": evidence[
            "camera_id"
        ],
        "track_id": evidence[
            "track_id"
        ],
        "first_seen": evidence[
            "first_seen"
        ],
        "last_seen": evidence[
            "last_seen"
        ],
        "embeddings": list(
            evidence[
                "embeddings"
            ]
        ),
        "timestamps": list(
            evidence[
                "timestamps"
            ]
        ),
        "supporting_frames": len(
            evidence[
                "embeddings"
            ]
        ),
    }


# ============================================================
# EMBEDDING SET COMPARISON
# ============================================================

def _compare_embedding_sets(
    current_embeddings: list,
    stored_embeddings: list,
) -> Optional[dict[str, Any]]:

    if not current_embeddings:
        return None

    if not stored_embeddings:
        return None

    current = [
        _normalize_embedding(
            embedding
        )
        for embedding in current_embeddings
    ]

    stored = [
        _normalize_embedding(
            embedding
        )
        for embedding in stored_embeddings
    ]

    current = [
        item
        for item in current
        if item is not None
    ]

    stored = [
        item
        for item in stored
        if item is not None
    ]

    if not current or not stored:
        return None

    similarities = []

    # Each current observation searches against the strongest
    # appearance evidence from the global identity.
    for current_embedding in current:

        best = -1.0

        for stored_embedding in stored:

            similarity = _safe_float(
                cosine_similarity(
                    current_embedding,
                    stored_embedding,
                ),
                0.0,
            )

            if similarity > best:
                best = similarity

        if best >= REID_FRAME_THRESHOLD:
            similarities.append(
                best
            )

    if not similarities:
        return {
            "best_similarity": 0.0,
            "mean_top_similarity": 0.0,
            "supporting_count": 0,
            "consistency": 0.0,
            "all_similarities": [],
        }

    similarities.sort(
        reverse=True
    )

    best_similarity = (
        similarities[0]
    )

    top_count = min(
        len(similarities),
        max(
            MIN_SUPPORTING_EMBEDDINGS,
            MIN_STRONG_SUPPORTING_EMBEDDINGS,
        ),
    )

    top_similarities = (
        similarities[:top_count]
    )

    mean_top_similarity = float(
        np.mean(
            top_similarities
        )
    )

    spread = float(
        np.std(
            top_similarities
        )
    )

    consistency = _clamp01(
        1.0 - (
            spread / 0.20
        )
    )

    return {
        "best_similarity":
            _clamp01(
                best_similarity
            ),

        "mean_top_similarity":
            _clamp01(
                mean_top_similarity
            ),

        "supporting_count":
            len(
                top_similarities
            ),

        "consistency":
            consistency,

        "all_similarities":
            [
                _clamp01(
                    value
                )
                for value
                in similarities
            ],
    }


# ============================================================
# RE-ID MATCH DECISION
# ============================================================

def _evaluate_match(
    comparison: Optional[dict[str, Any]],
) -> dict[str, Any]:

    if comparison is None:
        return {
            "matched": False,
            "strong_match": False,
            "best_similarity": 0.0,
            "mean_top_similarity": 0.0,
            "supporting_count": 0,
            "consistency": 0.0,
        }

    best = _clamp01(
        comparison.get(
            "best_similarity",
            0.0,
        )
    )

    mean_top = _clamp01(
        comparison.get(
            "mean_top_similarity",
            0.0,
        )
    )

    supporting = max(
        0,
        int(
            comparison.get(
                "supporting_count",
                0,
            )
        ),
    )

    consistency = _clamp01(
        comparison.get(
            "consistency",
            0.0,
        )
    )

    strong_match = (
        supporting
        >= MIN_STRONG_SUPPORTING_EMBEDDINGS
        and mean_top
        >= REID_STRONG_CONSENSUS
        and best
        >= REID_STRONG_THRESHOLD
        and consistency
        >= 0.50
    )

    normal_match = (
        supporting
        >= MIN_SUPPORTING_EMBEDDINGS
        and mean_top
        >= REID_FRAME_THRESHOLD
        and best
        >= REID_SUPPORT_THRESHOLD
        and consistency
        >= 0.40
    )

    return {
        "matched":
            bool(
                normal_match
                or strong_match
            ),

        "strong_match":
            bool(
                strong_match
            ),

        "best_similarity":
            best,

        "mean_top_similarity":
            mean_top,

        "supporting_count":
            supporting,

        "consistency":
            consistency,
    }


# ============================================================
# TIME COMPATIBILITY
# ============================================================

def _time_compatible(
    current_timestamp: Any,
    previous_timestamp: Any,
) -> bool:

    if previous_timestamp is None:
        return True

    current = _safe_timestamp(
        current_timestamp
    )

    previous = _safe_timestamp(
        previous_timestamp
    )

    gap = (
        current
        - previous
    )

    if gap < MIN_TRAVEL_TIME_SECONDS:
        return False

    if (
        gap
        > MAX_SIGHTING_GAP_SECONDS
    ):
        return False

    return True


def _temporal_score(
    current_timestamp: Any,
    previous_timestamp: Any,
) -> float:

    if previous_timestamp is None:
        return 1.0

    current = _safe_timestamp(
        current_timestamp
    )

    previous = _safe_timestamp(
        previous_timestamp
    )

    gap = (
        current
        - previous
    )

    if gap < 0.0:
        return 0.0

    if gap > MAX_SIGHTING_GAP_SECONDS:
        return 0.0

    return _clamp01(
        1.0
        - (
            gap
            / MAX_SIGHTING_GAP_SECONDS
        )
    )


# ============================================================
# POSTGIS CROSS-CAMERA CANDIDATE GATE
# ============================================================

_POSTGIS_HARD_REJECT_REASONS = {
    "destination_precedes_source",
    "transition_time_exceeded",
    "camera_distance_exceeded",
    "zero_time_nonzero_distance",
    "required_speed_exceeded",
}


def _camera_db_id(
    session: Any,
    camera_reference: Any,
) -> Optional[int]:
    """
    Resolve an INTEL-I camera reference to cameras.id.

    Resolution order:
      1. cameras.cam_id
      2. numeric cameras.id fallback

    Duplicate cam_id values are deliberately not resolved arbitrarily because
    that could cause a cross-tenant spatial correlation.
    """
    if (
        session is None
        or sql_text is None
        or camera_reference is None
    ):
        return None

    reference = str(camera_reference).strip()

    if not reference:
        return None

    try:
        rows = session.execute(
            sql_text(
                """
                SELECT id
                FROM public.cameras
                WHERE cam_id = :camera_reference
                ORDER BY id
                LIMIT 2
                """
            ),
            {
                "camera_reference": reference,
            },
        ).all()

        if len(rows) == 1:
            return int(rows[0][0])

        if len(rows) > 1:
            logger.warning(
                "PostGIS vehicle camera resolution rejected "
                "duplicate cam_id=%s",
                reference,
            )
            return None

        try:
            numeric_id = int(reference)
        except (TypeError, ValueError):
            return None

        row = session.execute(
            sql_text(
                """
                SELECT id
                FROM public.cameras
                WHERE id = :camera_id
                LIMIT 1
                """
            ),
            {
                "camera_id": numeric_id,
            },
        ).first()

        if row is None:
            return None

        return int(row[0])

    except Exception:
        logger.exception(
            "PostGIS vehicle camera resolution failed for %r",
            camera_reference,
        )
        return None


def _timestamp_to_utc_datetime(
    value: Any,
) -> datetime:
    timestamp = _safe_timestamp(
        value
    )

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    )


def _postgis_candidate_gate(
    vehicle: dict[str, Any],
    global_vehicle: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply the database-backed physical-transition gate before expensive
    cross-camera Re-ID work.

    Fail-open is deliberate for infrastructure/location unavailability:
    missing geometry must not create a false identity rejection. A
    transition is hard-rejected only when PostGIS successfully evaluates
    it and proves that the movement is physically impossible.
    """

    unavailable = {
        "available": False,
        "feasible": True,
        "hard_reject": False,
        "reason": "postgis_gate_unavailable",
        "distance_meters": None,
        "elapsed_seconds": None,
        "required_speed_kmh": None,
        "source_camera_db_id": None,
        "destination_camera_db_id": None,
    }

    if not POSTGIS_VEHICLE_GATE_ENABLED:
        result = dict(unavailable)
        result["reason"] = "postgis_gate_disabled"
        return result

    if (
        SessionLocal is None
        or sql_text is None
        or get_postgis_spatial_service is None
    ):
        result = dict(unavailable)
        result["reason"] = "postgis_dependencies_unavailable"
        return result

    if not isinstance(vehicle, dict):
        return unavailable

    if not isinstance(global_vehicle, dict):
        return unavailable

    source_camera = global_vehicle.get(
        "last_camera_id"
    )
    destination_camera = vehicle.get(
        "camera_id"
    )

    if (
        source_camera is None
        or destination_camera is None
        or str(source_camera) == str(destination_camera)
    ):
        result = dict(unavailable)
        result["reason"] = "postgis_not_cross_camera"
        return result

    source_timestamp = global_vehicle.get(
        "last_seen"
    )
    destination_timestamp = vehicle.get(
        "timestamp"
    )

    if (
        source_timestamp is None
        or destination_timestamp is None
    ):
        result = dict(unavailable)
        result["reason"] = "postgis_timestamp_unavailable"
        return result

    session = None

    try:
        session = SessionLocal()

        source_db_id = _camera_db_id(
            session,
            source_camera,
        )
        destination_db_id = _camera_db_id(
            session,
            destination_camera,
        )

        if (
            source_db_id is None
            or destination_db_id is None
        ):
            result = dict(unavailable)
            result["reason"] = "postgis_camera_not_resolved"
            result["source_camera_db_id"] = source_db_id
            result["destination_camera_db_id"] = destination_db_id
            return result

        spatial = get_postgis_spatial_service(
            session
        )

        feasibility = (
            spatial.transition_feasibility(
                source_camera_id=source_db_id,
                destination_camera_id=destination_db_id,
                source_timestamp=_timestamp_to_utc_datetime(
                    source_timestamp
                ),
                destination_timestamp=_timestamp_to_utc_datetime(
                    destination_timestamp
                ),
                max_speed_kmh=POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH,
                max_transition_seconds=(
                    POSTGIS_VEHICLE_GATE_MAX_TRANSITION_SECONDS
                ),
                max_distance_km=(
                    POSTGIS_VEHICLE_GATE_MAX_DISTANCE_KM
                ),
            )
        )

        reason = str(
            getattr(
                feasibility,
                "reason",
                "postgis_evaluated",
            )
        )

        feasible = bool(
            getattr(
                feasibility,
                "feasible",
                True,
            )
        )

        # Missing geometry is not evidence that the vehicle could not
        # travel between cameras. Preserve the existing GIS fallback.
        if reason == "camera_location_unavailable":
            return {
                "available": False,
                "feasible": True,
                "hard_reject": False,
                "reason": reason,
                "distance_meters": getattr(
                    feasibility,
                    "distance_m",
                    None,
                ),
                "elapsed_seconds": getattr(
                    feasibility,
                    "elapsed_seconds",
                    None,
                ),
                "required_speed_kmh": getattr(
                    feasibility,
                    "required_speed_kmh",
                    None,
                ),
                "source_camera_db_id": source_db_id,
                "destination_camera_db_id": destination_db_id,
            }

        hard_reject = (
            not feasible
            and reason in _POSTGIS_HARD_REJECT_REASONS
        )

        return {
            "available": True,
            "feasible": feasible,
            "hard_reject": hard_reject,
            "reason": reason,
            "distance_meters": getattr(
                feasibility,
                "distance_m",
                None,
            ),
            "elapsed_seconds": getattr(
                feasibility,
                "elapsed_seconds",
                None,
            ),
            "required_speed_kmh": getattr(
                feasibility,
                "required_speed_kmh",
                None,
            ),
            "source_camera_db_id": source_db_id,
            "destination_camera_db_id": destination_db_id,
        }

    except (
        SpatialUnavailableError,
        SpatialError,
    ) as exc:
        logger.warning(
            "PostGIS vehicle candidate gate unavailable: %s",
            exc,
        )

        if not POSTGIS_VEHICLE_GATE_FAIL_OPEN:
            return {
                **unavailable,
                "feasible": False,
                "hard_reject": True,
                "reason": "postgis_gate_required_but_unavailable",
            }

        return unavailable

    except Exception:
        logger.exception(
            "Unexpected PostGIS vehicle candidate gate failure"
        )

        if not POSTGIS_VEHICLE_GATE_FAIL_OPEN:
            return {
                **unavailable,
                "feasible": False,
                "hard_reject": True,
                "reason": "postgis_gate_required_but_failed",
            }

        return unavailable

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.debug(
                    "Failed to close PostGIS gate DB session",
                    exc_info=True,
                )


# ============================================================
# GIS HELPERS
# ============================================================

def _safe_camera_id_for_gis(
    camera_id: Any,
) -> Optional[str]:

    if not isinstance(
        camera_id,
        str,
    ):
        return None

    camera_id = camera_id.strip()

    if (
        not camera_id
        or len(camera_id)
        > MAX_GIS_CAMERA_ID_LENGTH
    ):
        return None

    if any(
        ord(ch) < 32
        for ch in camera_id
    ):
        return None

    return camera_id


def _get_valid_camera_gis(
    camera_id: Any,
) -> Optional[dict[str, Any]]:

    camera_id = (
        _safe_camera_id_for_gis(
            camera_id
        )
    )

    if (
        camera_id is None
        or get_camera_gis is None
    ):
        return None

    try:

        metadata = get_camera_gis(
            camera_id
        )

    except Exception:

        return None

    if not isinstance(
        metadata,
        dict,
    ):
        return None

    try:

        latitude = float(
            metadata.get(
                "latitude"
            )
        )

        longitude = float(
            metadata.get(
                "longitude"
            )
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None

    if (
        not np.isfinite(
            latitude
        )
        or not np.isfinite(
            longitude
        )
    ):
        return None

    if not (
        -90.0
        <= latitude
        <= 90.0
    ):
        return None

    if not (
        -180.0
        <= longitude
        <= 180.0
    ):
        return None

    result = {
        "camera_id":
            camera_id,

        "latitude":
            latitude,

        "longitude":
            longitude,
    }

    name = metadata.get(
        "camera_name"
    )

    if (
        isinstance(
            name,
            str,
        )
        and 0 < len(name)
        <= MAX_GIS_ZONE_LENGTH
    ):

        result[
            "camera_name"
        ] = name

    zone = metadata.get(
        "zone"
    )

    if (
        isinstance(
            zone,
            str,
        )
        and 0 < len(zone)
        <= MAX_GIS_ZONE_LENGTH
    ):

        result[
            "zone"
        ] = zone

    return result


def _haversine_meters(
    lat1,
    lon1,
    lat2,
    lon2,
) -> Optional[float]:

    try:

        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None

    if not all(
        np.isfinite(
            value
        )
        for value in (
            lat1,
            lon1,
            lat2,
            lon2,
        )
    ):
        return None

    if not (
        -90 <= lat1 <= 90
        and -90 <= lat2 <= 90
        and -180 <= lon1 <= 180
        and -180 <= lon2 <= 180
    ):
        return None

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dphi / 2.0
        ) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            dlambda / 2.0
        ) ** 2
    )

    a = max(
        0.0,
        min(
            1.0,
            a,
        ),
    )

    return float(
        2.0
        * EARTH_RADIUS_METERS
        * math.asin(
            math.sqrt(a)
        )
    )


def _bearing_degrees(
    lat1,
    lon1,
    lat2,
    lon2,
) -> Optional[float]:

    try:

        lat1 = math.radians(
            float(lat1)
        )

        lat2 = math.radians(
            float(lat2)
        )

        delta_lon = math.radians(
            float(lon2)
            - float(lon1)
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None

    x = (
        math.sin(
            delta_lon
        )
        * math.cos(
            lat2
        )
    )

    y = (
        math.cos(
            lat1
        )
        * math.sin(
            lat2
        )
        -
        math.sin(
            lat1
        )
        * math.cos(
            lat2
        )
        * math.cos(
            delta_lon
        )
    )

    bearing = math.degrees(
        math.atan2(
            x,
            y,
        )
    )

    return (
        bearing + 360.0
    ) % 360.0


def _angle_difference(
    a,
    b,
) -> Optional[float]:

    try:

        a = float(a) % 360.0
        b = float(b) % 360.0

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None

    difference = abs(
        a - b
    )

    return min(
        difference,
        360.0 - difference,
    )


# ============================================================
# GIS TRANSITION EVALUATION
# ============================================================

def _evaluate_gis_for_candidate(
    vehicle: dict[str, Any],
    global_vehicle: dict[str, Any],
) -> dict[str, Any]:

    unavailable = {
        "available": False,
        "gis_score": 0.0,
        "direction_score": 0.5,
        "distance_meters": None,
        "bearing_degrees": None,
        "elapsed_seconds": None,
        "implied_speed_kmh": None,
        "travel_time_score": 0.0,
        "feasible": True,
        "reason": "gis_unavailable",
    }

    if not isinstance(
        vehicle,
        dict,
    ):
        return unavailable

    if not isinstance(
        global_vehicle,
        dict,
    ):
        return unavailable

    previous_camera_id = (
        global_vehicle.get(
            "last_camera_id"
        )
    )

    current_camera_id = (
        vehicle.get(
            "camera_id"
        )
    )

    previous_timestamp = (
        global_vehicle.get(
            "last_seen"
        )
    )

    current_timestamp = (
        vehicle.get(
            "timestamp",
            time.time(),
        )
    )

    if (
        previous_camera_id is None
        or current_camera_id is None
        or previous_camera_id
        == current_camera_id
        or previous_timestamp is None
    ):
        return unavailable

    # --------------------------------------------------------
    # Prefer the existing production GIS engine.
    # --------------------------------------------------------

    if evaluate_vehicle_transition is not None:

        def metadata_loader(
            camera_id,
        ):

            return _get_valid_camera_gis(
                camera_id
            )

        source_vehicle = {
            "camera_id":
                previous_camera_id,

            "timestamp":
                previous_timestamp,

            "heading":
                global_vehicle.get(
                    "last_heading"
                ),
        }

        destination_vehicle = {
            "camera_id":
                current_camera_id,

            "timestamp":
                current_timestamp,

            "heading":
                vehicle.get(
                    "heading",
                    vehicle.get(
                        "direction"
                    ),
                ),
        }

        try:

            score = (
                evaluate_vehicle_transition(
                    source_vehicle,
                    destination_vehicle,
                    metadata_loader=metadata_loader,
                )
            )

            if score is not None:

                try:

                    result = score.as_dict()

                except AttributeError:

                    result = (
                        dict(score)
                        if isinstance(
                            score,
                            dict,
                        )
                        else None
                    )

                if isinstance(
                    result,
                    dict,
                ):

                    return {
                        "available":
                            bool(
                                result.get(
                                    "gis_available",
                                    True,
                                )
                            ),

                        "gis_score":
                            _clamp01(
                                result.get(
                                    "gis_score",
                                    0.0,
                                )
                            ),

                        "direction_score":
                            _clamp01(
                                result.get(
                                    "direction_score",
                                    0.5,
                                )
                            ),

                        "distance_meters":
                            result.get(
                                "distance_meters"
                            ),

                        "bearing_degrees":
                            result.get(
                                "bearing_degrees"
                            ),

                        "elapsed_seconds":
                            result.get(
                                "elapsed_seconds"
                            ),

                        "implied_speed_kmh":
                            result.get(
                                "implied_speed_kmh"
                            ),

                        "travel_time_score":
                            _clamp01(
                                result.get(
                                    "travel_time_score",
                                    0.0,
                                )
                            ),

                        "feasible":
                            bool(
                                result.get(
                                    "feasible",
                                    True,
                                )
                            ),

                        "reason":
                            str(
                                result.get(
                                    "reason",
                                    "gis_evaluated",
                                )
                            ),
                    }

        except Exception:

            logger.debug(
                "GIS engine evaluation failed",
                exc_info=True,
            )

    # --------------------------------------------------------
    # Fallback geographic evaluation.
    # --------------------------------------------------------

    source = _get_valid_camera_gis(
        previous_camera_id
    )

    destination = _get_valid_camera_gis(
        current_camera_id
    )

    if (
        source is None
        or destination is None
    ):
        return unavailable

    distance = _haversine_meters(
        source["latitude"],
        source["longitude"],
        destination["latitude"],
        destination["longitude"],
    )

    bearing = _bearing_degrees(
        source["latitude"],
        source["longitude"],
        destination["latitude"],
        destination["longitude"],
    )

    if distance is None:
        return unavailable

    current_time = _safe_timestamp(
        current_timestamp
    )

    previous_time = _safe_timestamp(
        previous_timestamp
    )

    elapsed = (
        current_time
        - previous_time
    )

    if elapsed < 0.0:
        return {
            "available": True,
            "gis_score": 0.0,
            "direction_score": 0.0,
            "distance_meters": distance,
            "bearing_degrees": bearing,
            "elapsed_seconds": elapsed,
            "implied_speed_kmh": None,
            "travel_time_score": 0.0,
            "feasible": False,
            "reason": "destination_timestamp_before_source",
        }

    if elapsed <= 0.0:
        return {
            "available": True,
            "gis_score": 0.0,
            "direction_score": 0.5,
            "distance_meters": distance,
            "bearing_degrees": bearing,
            "elapsed_seconds": elapsed,
            "implied_speed_kmh": None,
            "travel_time_score": 0.0,
            "feasible": False,
            "reason": "zero_travel_time",
        }

    implied_speed_kmh = (
        distance
        / elapsed
        * 3.6
    )

    # Conservative fallback upper bound.
    max_reasonable_speed = float(
        os.getenv(
            "GIS_FALLBACK_MAX_SPEED_KMH",
            "180",
        )
    )

    feasible = (
        implied_speed_kmh
        <= max_reasonable_speed
    )

    speed_score = _clamp01(
        1.0
        - (
            implied_speed_kmh
            / max_reasonable_speed
        )
    )

    # No reliable road-network travel-time model is available in
    # fallback mode, therefore don't pretend this is route duration.
    travel_time_score = speed_score

    direction_score = 0.5

    current_heading = vehicle.get(
        "heading",
        vehicle.get(
            "direction"
        ),
    )

    if (
        current_heading is not None
        and bearing is not None
    ):

        angle = _angle_difference(
            current_heading,
            bearing,
        )

        if angle is not None:

            if angle <= 30:
                direction_score = 1.0
            elif angle <= 60:
                direction_score = 0.75
            elif angle <= 100:
                direction_score = 0.50
            elif angle <= 140:
                direction_score = 0.25
            else:
                direction_score = 0.0

    gis_score = _clamp01(
        0.65 * speed_score
        + 0.35 * direction_score
    )

    return {
        "available": True,
        "gis_score": gis_score,
        "direction_score": direction_score,
        "distance_meters": distance,
        "bearing_degrees": bearing,
        "elapsed_seconds": elapsed,
        "implied_speed_kmh": implied_speed_kmh,
        "travel_time_score": travel_time_score,
        "feasible": feasible,
        "reason": (
            "gis_fallback_feasible"
            if feasible
            else "gis_fallback_impossible_speed"
        ),
    }


# ============================================================
# GLOBAL VEHICLE CREATION
# ============================================================

def _generate_global_vehicle_id() -> str:

    return (
        "GV-"
        + uuid.uuid4().hex.upper()
    )


def _create_global_vehicle(
    vehicle: dict[str, Any],
    embedding: list[float],
) -> dict[str, Any]:

    global_vehicle_id = (
        _generate_global_vehicle_id()
    )

    now = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    camera_id = vehicle.get(
        "camera_id"
    )

    track_id = vehicle.get(
        "track_id"
    )

    vehicle_type = vehicle.get(
        "vehicle_type"
    )

    local_key = (
        f"{camera_id}:{track_id}"
    )

    state: dict[str, Any] = {

        "global_vehicle_id":
            global_vehicle_id,

        "vehicle_type":
            vehicle_type,

        "first_seen":
            now,

        "last_seen":
            now,

        "last_camera_id":
            camera_id,

        "last_track_id":
            track_id,

        "last_heading":
            vehicle.get(
                "heading",
                vehicle.get(
                    "direction"
                ),
            ),

        "status":
            "ACTIVE",

        "confidence_status":
            "UNKNOWN",

        "sightings":
            [],

        "journey":
            [],

        "camera_visits":
            {},

        "embeddings":
            [],

        "correlation_count":
            0,

        "strong_match_count":
            0,

        "last_similarity":
            0.0,

        "best_similarity":
            0.0,

        "last_match_type":
            "new_global_vehicle",

        "last_correlation_score":
            0.0,

        "plate":
            None,

        "plate_confidence":
            0.0,

        "plate_support_count":
            0,

        "plate_observations":
            [],

        "plate_candidates":
            {},

        "plate_match_strength":
            "none",

        "plate_status":
            "UNKNOWN",

        "plate_consensus_margin":
            0.0,

        "plate_conflict_candidates":
            [],

        "created_at":
            now,

        "updated_at":
            now,
    }

    normalized = _normalize_embedding(
        embedding
    )

    if normalized is not None:

        state[
            "embeddings"
        ].append(
            normalized
        )

    _add_global_sighting(
        state,
        vehicle,
        similarity=1.0,
    )

    _record_journey_evidence(
        global_vehicle=state,
        vehicle=vehicle,
        similarity=1.0,
        match_type="new_global_vehicle",
        supporting_frames=1,
        consistency=1.0,
        strong_match=False,
    )

    plate_result = (
        _record_plate_observation(
            state,
            vehicle,
        )
    )

    if plate_result is not None:

        _record_plate_journey_evidence(
            state,
            vehicle,
            plate_result,
        )

    _global_vehicles[
        global_vehicle_id
    ] = state

    _local_to_global[
        local_key
    ] = global_vehicle_id

    return state


# ============================================================
# GLOBAL EMBEDDING STORAGE
# ============================================================

def _add_global_embedding(
    global_vehicle: dict[str, Any],
    embedding: Any,
) -> None:

    if not _embedding_quality(
        embedding
    ):
        return

    normalized = _normalize_embedding(
        embedding
    )

    if normalized is None:
        return

    embeddings = global_vehicle.setdefault(
        "embeddings",
        [],
    )

    if embeddings:

        similarity = _safe_float(
            cosine_similarity(
                normalized,
                embeddings[-1],
            ),
            0.0,
        )

        # Same-frame duplicate.
        if similarity >= 0.995:
            return

    embeddings.append(
        normalized
    )

    if (
        len(embeddings)
        > MAX_EMBEDDINGS_PER_GLOBAL_VEHICLE
    ):

        del embeddings[
            :-MAX_EMBEDDINGS_PER_GLOBAL_VEHICLE
        ]


# ============================================================
# GLOBAL SIGHTING
# ============================================================

def _add_global_sighting(
    global_vehicle: dict[str, Any],
    vehicle: dict[str, Any],
    similarity: float,
) -> None:

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    bbox = vehicle.get(
        "bbox"
    )

    if isinstance(
        bbox,
        (list, tuple),
    ):

        try:
            bbox = [
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
            bbox = None

    center = vehicle.get(
        "center"
    )

    if isinstance(
        center,
        (list, tuple),
    ):

        try:
            center = [
                float(center[0]),
                float(center[1]),
            ]
        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            center = None

    sighting = {

        "camera_id":
            str(
                vehicle.get(
                    "camera_id",
                    "",
                )
            )[:MAX_GIS_CAMERA_ID_LENGTH],

        "track_id":
            str(
                vehicle.get(
                    "track_id",
                    "",
                )
            )[:128],

        "timestamp":
            timestamp,

        "bbox":
            bbox,

        "center":
            center,

        "confidence":
            _clamp01(
                vehicle.get(
                    "confidence",
                    0.0,
                )
            ),

        "similarity":
            _clamp01(
                similarity
            ),

        "vehicle_type":
            _normalize_vehicle_type(
                vehicle.get(
                    "vehicle_type"
                )
            ),
    }

    sightings = (
        global_vehicle.setdefault(
            "sightings",
            [],
        )
    )

    sightings.append(
        sighting
    )

    if (
        len(sightings)
        > MAX_SIGHTINGS_PER_GLOBAL_VEHICLE
    ):

        del sightings[
            :-MAX_SIGHTINGS_PER_GLOBAL_VEHICLE
        ]


# ============================================================
# CAMERA VISITS
# ============================================================

def _update_camera_visit(
    global_vehicle: dict[str, Any],
    vehicle: dict[str, Any],
) -> None:

    camera_id = str(
        vehicle.get(
            "camera_id",
            "",
        )
    ).strip()

    if not camera_id:
        return

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    visits = (
        global_vehicle.setdefault(
            "camera_visits",
            {},
        )
    )

    visit = visits.get(
        camera_id
    )

    if visit is None:

        if (
            len(visits)
            >= MAX_GIS_CAMERAS_PER_VEHICLE
        ):

            oldest_camera = min(
                visits,
                key=lambda key:
                    _safe_float(
                        visits[key].get(
                            "last_seen",
                            0.0,
                        ),
                        0.0,
                    ),
            )

            visits.pop(
                oldest_camera,
                None,
            )

        visit = {
            "first_seen":
                timestamp,

            "last_seen":
                timestamp,

            "observation_count":
                0,

            "track_ids":
                [],
        }

        visits[
            camera_id
        ] = visit

    visit[
        "first_seen"
    ] = min(
        _safe_timestamp(
            visit.get(
                "first_seen",
                timestamp,
            )
        ),
        timestamp,
    )

    visit[
        "last_seen"
    ] = max(
        _safe_timestamp(
            visit.get(
                "last_seen",
                timestamp,
            )
        ),
        timestamp,
    )

    visit[
        "observation_count"
    ] += 1

    track_id = vehicle.get(
        "track_id"
    )

    if track_id is not None:

        track_id = str(
            track_id
        )

        if track_id not in visit[
            "track_ids"
        ]:

            visit[
                "track_ids"
            ].append(
                track_id
            )

        if len(
            visit["track_ids"]
        ) > 50:

            del visit[
                "track_ids"
            ][:-50]


# ============================================================
# JOURNEY EVENT
# ============================================================

def _build_journey_event(
    vehicle: dict[str, Any],
    similarity: float,
    match_type: str,
    supporting_frames: int = 0,
    consistency: float = 0.0,
    strong_match: bool = False,
) -> Optional[dict[str, Any]]:

    if not isinstance(
        vehicle,
        dict,
    ):
        return None

    camera_id = str(
        vehicle.get(
            "camera_id",
            "",
        )
    ).strip()

    if not camera_id:
        return None

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    event = {

        "camera_id":
            camera_id[:MAX_GIS_CAMERA_ID_LENGTH],

        "track_id":
            str(
                vehicle.get(
                    "track_id",
                    "",
                )
            )[:128],

        "timestamp":
            timestamp,

        "similarity":
            _clamp01(
                similarity
            ),

        "match_type":
            str(
                match_type
                or "unknown"
            )[:64],

        "supporting_frames":
            max(
                0,
                int(
                    supporting_frames
                ),
            ),

        "consistency":
            _clamp01(
                consistency
            ),

        "strong_match":
            bool(
                strong_match
            ),

        "vehicle_type":
            _normalize_vehicle_type(
                vehicle.get(
                    "vehicle_type"
                )
            ),
    }

    gis = _get_valid_camera_gis(
        camera_id
    )

    if gis is not None:

        event[
            "gis"
        ] = gis

    return event


def _add_journey_event(
    global_vehicle: dict[str, Any],
    event: dict[str, Any],
) -> None:

    if not isinstance(
        event,
        dict,
    ):
        return

    journey = (
        global_vehicle.setdefault(
            "journey",
            [],
        )
    )

    journey.append(
        event
    )

    if (
        len(journey)
        > MAX_JOURNEY_EVENTS_PER_VEHICLE
    ):

        del journey[
            :-MAX_JOURNEY_EVENTS_PER_VEHICLE
        ]


def _record_journey_evidence(
    global_vehicle: dict[str, Any],
    vehicle: dict[str, Any],
    similarity: float,
    match_type: str,
    supporting_frames: int = 0,
    consistency: float = 0.0,
    strong_match: bool = False,
) -> None:

    event = _build_journey_event(
        vehicle=vehicle,
        similarity=similarity,
        match_type=match_type,
        supporting_frames=supporting_frames,
        consistency=consistency,
        strong_match=strong_match,
    )

    if event is None:
        return

    _add_journey_event(
        global_vehicle,
        event,
    )

    _update_camera_visit(
        global_vehicle,
        vehicle,
    )


# ============================================================
# PLATE JOURNEY EVIDENCE
# ============================================================

def _record_plate_journey_evidence(
    global_vehicle: dict[str, Any],
    vehicle: dict[str, Any],
    plate_result: dict[str, Any],
) -> None:

    if (
        not isinstance(
            plate_result,
            dict,
        )
        or not plate_result.get(
            "stable",
            False,
        )
    ):
        return

    event = _build_journey_event(
        vehicle=vehicle,
        similarity=plate_result.get(
            "mean_confidence",
            0.0,
        ),
        match_type="plate_confirmed",
        supporting_frames=plate_result.get(
            "support_count",
            0,
        ),
        consistency=1.0,
        strong_match=(
            plate_result.get(
                "best_confidence",
                0.0,
            )
            >= PLATE_STRONG_CONFIDENCE
        ),
    )

    if event is None:
        return

    event[
        "plate"
    ] = plate_result.get(
        "plate"
    )

    event[
        "plate_confidence"
    ] = _clamp01(
        plate_result.get(
            "mean_confidence",
            0.0,
        )
    )

    _add_journey_event(
        global_vehicle,
        event,
    )


# ============================================================
# TRACK QUALITY EVIDENCE
# ============================================================

def _track_quality_score(
    vehicle: dict[str, Any],
) -> float:
    """
    Canonical tracking-quality evidence.

    Prefer an explicit tracking confidence from tracker.py.
    Fall back to detector confidence only when no tracker metric
    is available. This value is evidence, not identity.
    """
    if not isinstance(vehicle, dict):
        return 0.0

    for key in (
        "tracking_confidence",
        "track_confidence",
        "track_quality",
        "track_quality_score",
    ):
        if key in vehicle and vehicle.get(key) is not None:
            return _clamp01(vehicle.get(key))

    return _clamp01(
        vehicle.get(
            "confidence",
            0.0,
        )
    )


def _embedding_quality_score(
    vehicle: dict[str, Any],
) -> float:
    """
    Appearance observation quality. Uses upstream quality when
    available; otherwise validates that a usable embedding exists.
    """
    if not isinstance(vehicle, dict):
        return 0.0

    for key in (
        "reid_quality",
        "embedding_quality",
        "reid_confidence",
    ):
        if key in vehicle and vehicle.get(key) is not None:
            return _clamp01(vehicle.get(key))

    return 1.0 if _embedding_quality(
        vehicle.get("embedding")
    ) else 0.0


# ============================================================
# CORRELATION SCORE
# ============================================================

def _calculate_correlation_score(
    decision: dict[str, Any],
    plate_score: float,
    temporal_score: float = 1.0,
    vehicle_type_score: float = 1.0,
    track_quality_score: float = 1.0,
    embedding_quality_score: float = 1.0,
) -> float:
    """
    Canonical production correlation score.

    The score is computed exactly once from available evidence.
    Missing GIS evidence is redistributed across the evidence that
    is actually available; it is never treated as positive evidence.

    Evidence:
        Re-ID appearance
        ANPR
        GIS/topology
        temporal feasibility
        vehicle type
        track quality
        observation/Re-ID quality
    """
    if not isinstance(decision, dict):
        return 0.0

    reid_component = _clamp01(
        0.70 * decision.get(
            "mean_top_similarity",
            0.0,
        )
        + 0.20 * decision.get(
            "best_similarity",
            0.0,
        )
        + 0.10 * decision.get(
            "consistency",
            0.0,
        )
    )

    gis_available = bool(
        decision.get(
            "gis_available",
            False,
        )
    )

    evidence = {
        "reid": reid_component,
        "plate": _clamp01(plate_score),
        "temporal": _clamp01(temporal_score),
        "type": _clamp01(vehicle_type_score),
        "track": _clamp01(track_quality_score),
        "embedding_quality": _clamp01(
            embedding_quality_score
        ),
    }

    weights = {
        "reid": REID_WEIGHT,
        "plate": PLATE_WEIGHT,
        "temporal": TEMPORAL_WEIGHT,
        "type": TYPE_WEIGHT,
        "track": TRACK_WEIGHT,
    }

    if gis_available:
        evidence["gis"] = _clamp01(
            decision.get(
                "gis_score",
                0.0,
            )
        )
        weights["gis"] = GIS_WEIGHT

    # Observation quality modulates appearance evidence instead of
    # becoming a fake independent identity signal.
    evidence["reid"] *= (
        0.60
        + 0.40 * evidence["embedding_quality"]
    )

    usable_weights = {
        key: max(0.0, float(value))
        for key, value in weights.items()
        if float(value) > 0.0
    }

    if not usable_weights:
        return 0.0

    total_weight = sum(
        usable_weights.values()
    )

    total = sum(
        usable_weights[key] * evidence[key]
        for key in usable_weights
    ) / total_weight

    support = max(
        0,
        int(
            decision.get(
                "supporting_count",
                0,
            )
        ),
    )

    # Multi-frame evidence is mandatory for cross-camera identity.
    if support < MIN_SUPPORTING_EMBEDDINGS:
        total *= 0.75

    if (
        evidence["track"]
        < MIN_TRACK_QUALITY_FOR_CORRELATION
    ):
        total *= 0.85

    return _clamp01(total)


# ============================================================
# SAME-CAMERA TRACK RECOVERY
# ============================================================

def _find_same_camera_track_recovery(
    vehicle: dict[str, Any],
    local_evidence: dict[str, Any],
) -> Optional[dict[str, Any]]:

    camera_id = vehicle.get(
        "camera_id"
    )

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    current_embeddings = (
        local_evidence.get(
            "embeddings",
            [],
        )
    )

    if not current_embeddings:
        return None

    candidates = []

    for (
        global_id,
        global_vehicle,
    ) in list(
        _global_vehicles.items()
    ):

        if (
            global_vehicle.get(
                "last_camera_id"
            )
            != camera_id
        ):
            continue

        previous_timestamp = (
            global_vehicle.get(
                "last_seen"
            )
        )

        if previous_timestamp is None:
            continue

        gap = (
            timestamp
            - _safe_timestamp(
                previous_timestamp
            )
        )

        if (
            gap
            < SAME_CAMERA_MIN_GAP_SECONDS
            or gap
            > SAME_CAMERA_MAX_GAP_SECONDS
        ):
            continue

        if not _vehicle_type_compatible(
            vehicle.get(
                "vehicle_type"
            ),
            global_vehicle.get(
                "vehicle_type"
            ),
        ):
            continue

        comparison = (
            _compare_embedding_sets(
                current_embeddings,
                global_vehicle.get(
                    "embeddings",
                    [],
                ),
            )
        )

        if comparison is None:
            continue

        if (
            comparison[
                "supporting_count"
            ]
            < SAME_CAMERA_MIN_SUPPORT
        ):
            continue

        if (
            comparison[
                "best_similarity"
            ]
            < SAME_CAMERA_REID_THRESHOLD
        ):
            continue

        if (
            comparison[
                "consistency"
            ]
            < SAME_CAMERA_MIN_CONSISTENCY
        ):
            continue

        candidates.append(
            {
                "global_vehicle_id":
                    global_id,

                "comparison":
                    comparison,

                "gap":
                    gap,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate:
            (
                candidate[
                    "comparison"
                ][
                    "mean_top_similarity"
                ],
                candidate[
                    "comparison"
                ][
                    "consistency"
                ],
            ),
        reverse=True,
    )

    best = candidates[0]

    comparison = best[
        "comparison"
    ]

    strong = (
        comparison[
            "best_similarity"
        ]
        >= SAME_CAMERA_STRONG_THRESHOLD
        and comparison[
            "consistency"
        ]
        >= SAME_CAMERA_STRONG_CONSISTENCY
        and comparison[
            "supporting_count"
        ]
        >= SAME_CAMERA_MIN_SUPPORT
    )

    return {
        "global_vehicle_id":
            best[
                "global_vehicle_id"
            ],

        "best_similarity":
            comparison[
                "best_similarity"
            ],

        "mean_similarity":
            comparison[
                "mean_top_similarity"
            ],

        "supporting_frames":
            comparison[
                "supporting_count"
            ],

        "consistency":
            comparison[
                "consistency"
            ],

        "strong_match":
            strong,
    }


# ============================================================
# FIND BEST GLOBAL VEHICLE
# ============================================================

def _find_best_match(
    vehicle: dict[str, Any],
    local_evidence: dict[str, Any],
) -> Optional[dict[str, Any]]:

    if local_evidence is None:
        return None

    current_camera = vehicle.get(
        "camera_id"
    )

    current_timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    current_type = vehicle.get(
        "vehicle_type"
    )

    current_embeddings = (
        local_evidence.get(
            "embeddings",
            [],
        )
    )

    if not current_embeddings:
        return None

    candidates = []

    for (
        global_id,
        global_vehicle,
    ) in list(
        _global_vehicles.items()
    ):

        # ----------------------------------------------------
        # Do not compare against the same current camera.
        # Same-camera recovery is handled separately.
        # ----------------------------------------------------

        if (
            global_vehicle.get(
                "last_camera_id"
            )
            == current_camera
        ):
            continue

        # ----------------------------------------------------
        # Vehicle type.
        # ----------------------------------------------------

        if not _vehicle_type_compatible(
            current_type,
            global_vehicle.get(
                "vehicle_type"
            ),
        ):
            continue

        # ----------------------------------------------------
        # Time.
        # ----------------------------------------------------

        if not _time_compatible(
            current_timestamp,
            global_vehicle.get(
                "last_seen"
            ),
        ):
            continue

        temporal_score = _temporal_score(
            current_timestamp,
            global_vehicle.get(
                "last_seen"
            ),
        )

        # ----------------------------------------------------
        # PostGIS physical candidate gate.
        #
        # This intentionally runs BEFORE plate comparison and the
        # expensive multi-frame Re-ID comparison. When PostGIS can
        # prove that a transition is impossible, the candidate is
        # rejected without spending Re-ID compute.
        # ----------------------------------------------------

        postgis_gate = _postgis_candidate_gate(
            vehicle,
            global_vehicle,
        )

        if postgis_gate.get(
            "hard_reject",
            False,
        ):

            decision = {
                "matched": False,
                "strong_match": False,
                "best_similarity": 0.0,
                "mean_top_similarity": 0.0,
                "supporting_count": 0,
                "consistency": 0.0,
                "track_quality_score":
                    _track_quality_score(vehicle),
                "embedding_quality_score":
                    _embedding_quality_score(vehicle),
                "gis_available": True,
                "gis_score": 0.0,
                "gis_direction_score": 0.5,
                "gis_distance_meters":
                    postgis_gate.get(
                        "distance_meters"
                    ),
                "gis_bearing_degrees": None,
                "gis_elapsed_seconds":
                    postgis_gate.get(
                        "elapsed_seconds"
                    ),
                "gis_implied_speed_kmh":
                    postgis_gate.get(
                        "required_speed_kmh"
                    ),
                "gis_travel_time_score": 0.0,
                "gis_feasible": False,
                "gis_reason":
                    postgis_gate.get(
                        "reason",
                        "postgis_infeasible_transition",
                    ),
                "gis_adjusted_score": 0.0,
                "gis_rejected": True,
                "postgis_gate_available":
                    postgis_gate.get(
                        "available",
                        False,
                    ),
                "postgis_source_camera_db_id":
                    postgis_gate.get(
                        "source_camera_db_id"
                    ),
                "postgis_destination_camera_db_id":
                    postgis_gate.get(
                        "destination_camera_db_id"
                    ),
            }

            gis = {
                "available": True,
                "gis_score": 0.0,
                "direction_score": 0.5,
                "distance_meters":
                    postgis_gate.get(
                        "distance_meters"
                    ),
                "bearing_degrees": None,
                "elapsed_seconds":
                    postgis_gate.get(
                        "elapsed_seconds"
                    ),
                "implied_speed_kmh":
                    postgis_gate.get(
                        "required_speed_kmh"
                    ),
                "travel_time_score": 0.0,
                "feasible": False,
                "reason":
                    postgis_gate.get(
                        "reason",
                        "postgis_infeasible_transition",
                    ),
            }

            candidates.append(
                {
                    "global_vehicle_id":
                        global_id,
                    "decision":
                        decision,
                    "comparison":
                        None,
                    "gis":
                        gis,
                    "gis_rejected":
                        True,
                    "temporal_score":
                        temporal_score,
                }
            )

            continue

        # ----------------------------------------------------
        # Plate contradiction.
        #
        # A strong established plate mismatch blocks a candidate.
        # ----------------------------------------------------

        if not _plate_compatible(
            vehicle,
            global_vehicle,
        ):
            continue

        # ----------------------------------------------------
        # Re-ID.
        # ----------------------------------------------------

        stored_embeddings = (
            global_vehicle.get(
                "embeddings",
                [],
            )
        )

        comparison = (
            _compare_embedding_sets(
                current_embeddings,
                stored_embeddings,
            )
        )

        decision = _evaluate_match(
            comparison
        )

        decision[
            "track_quality_score"
        ] = _track_quality_score(
            vehicle
        )

        decision[
            "embedding_quality_score"
        ] = _embedding_quality_score(
            vehicle
        )

        # Extremely poor tracking evidence should not be allowed
        # to create a cross-camera identity even with one lucky
        # appearance similarity.
        if (
            decision[
                "track_quality_score"
            ]
            < HARD_REJECT_LOW_TRACK_QUALITY
        ):
            continue

        if not decision.get(
            "matched",
            False,
        ):
            continue

        # ----------------------------------------------------
        # GIS.
        # ----------------------------------------------------

        gis = _evaluate_gis_for_candidate(
            vehicle,
            global_vehicle,
        )

        # PostGIS is the authoritative database-backed physical gate.
        # The existing gis_engine remains the richer scoring layer for
        # direction/topology/travel evidence. If that layer cannot obtain
        # coordinates but PostGIS already evaluated the transition, retain
        # the authoritative distance/speed evidence rather than discarding it.
        if postgis_gate.get(
            "available",
            False,
        ):
            gis = dict(gis)

            if gis.get(
                "distance_meters"
            ) is None:
                gis[
                    "distance_meters"
                ] = postgis_gate.get(
                    "distance_meters"
                )

            if gis.get(
                "elapsed_seconds"
            ) is None:
                gis[
                    "elapsed_seconds"
                ] = postgis_gate.get(
                    "elapsed_seconds"
                )

            if gis.get(
                "implied_speed_kmh"
            ) is None:
                gis[
                    "implied_speed_kmh"
                ] = postgis_gate.get(
                    "required_speed_kmh"
                )

            gis[
                "postgis_gate_available"
            ] = True

            gis[
                "postgis_gate_reason"
            ] = postgis_gate.get(
                "reason",
                "postgis_evaluated",
            )

        # ----------------------------------------------------
        # Physically impossible transition.
        #
        # This is a hard rejection when GIS is available.
        # ----------------------------------------------------

        if (
            gis.get(
                "available",
                False,
            )
            and not gis.get(
                "feasible",
                True,
            )
        ):

            decision[
                "gis_available"
            ] = True

            decision[
                "gis_score"
            ] = _clamp01(
                gis.get(
                    "gis_score",
                    0.0,
                )
            )

            decision[
                "gis_direction_score"
            ] = _clamp01(
                gis.get(
                    "direction_score",
                    0.5,
                )
            )

            decision[
                "gis_distance_meters"
            ] = gis.get(
                "distance_meters"
            )

            decision[
                "gis_bearing_degrees"
            ] = gis.get(
                "bearing_degrees"
            )

            decision[
                "gis_elapsed_seconds"
            ] = gis.get(
                "elapsed_seconds"
            )

            decision[
                "gis_implied_speed_kmh"
            ] = gis.get(
                "implied_speed_kmh"
            )

            decision[
                "gis_travel_time_score"
            ] = _clamp01(
                gis.get(
                    "travel_time_score",
                    0.0,
                )
            )

            decision[
                "gis_feasible"
            ] = False

            decision[
                "gis_reason"
            ] = gis.get(
                "reason",
                "gis_infeasible_transition",
            )

            decision[
                "gis_adjusted_score"
            ] = 0.0

            decision[
                "gis_rejected"
            ] = True

            candidates.append(
                {
                    "global_vehicle_id":
                        global_id,

                    "decision":
                        decision,

                    "comparison":
                        comparison,

                    "gis":
                        gis,

                    "gis_rejected":
                        True,

                    "temporal_score":
                        temporal_score,
                }
            )

            continue

        # ----------------------------------------------------
        # GIS is supporting evidence.
        # ----------------------------------------------------

        decision[
            "gis_available"
        ] = bool(
            gis.get(
                "available",
                False,
            )
        )

        decision[
            "gis_score"
        ] = _clamp01(
            gis.get(
                "gis_score",
                0.0,
            )
        )

        decision[
            "gis_direction_score"
        ] = _clamp01(
            gis.get(
                "direction_score",
                0.5,
            )
        )

        decision[
            "gis_distance_meters"
        ] = gis.get(
            "distance_meters"
        )

        decision[
            "gis_bearing_degrees"
        ] = gis.get(
            "bearing_degrees"
        )

        decision[
            "gis_elapsed_seconds"
        ] = gis.get(
            "elapsed_seconds"
        )

        decision[
            "gis_implied_speed_kmh"
        ] = gis.get(
            "implied_speed_kmh"
        )

        decision[
            "gis_travel_time_score"
        ] = _clamp01(
            gis.get(
                "travel_time_score",
                0.0,
            )
        )

        decision[
            "gis_feasible"
        ] = bool(
            gis.get(
                "feasible",
                True,
            )
        )

        decision[
            "gis_reason"
        ] = gis.get(
            "reason",
            "gis_unavailable",
        )

        reid_component = (
            0.75
            * decision[
                "mean_top_similarity"
            ]
            +
            0.25
            * decision[
                "consistency"
            ]
        )

        decision[
            "gis_adjusted_score"
        ] = _clamp01(
            (
                (
                    1.0
                    - GIS_CORRELATION_WEIGHT
                )
                * reid_component
                +
                GIS_CORRELATION_WEIGHT
                * decision[
                    "gis_score"
                ]
            )
            if gis.get(
                "available",
                False,
            )
            else reid_component
        )

        candidates.append(
            {
                "global_vehicle_id":
                    global_id,

                "decision":
                    decision,

                "comparison":
                    comparison,

                "gis":
                    gis,

                "gis_rejected":
                    False,

                "temporal_score":
                    temporal_score,
            }
        )

    if not candidates:
        return None

    # --------------------------------------------------------
    # Highest quality candidate first.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda candidate:
            (
                0
                if candidate.get(
                    "gis_rejected",
                    False,
                )
                else 1,

                candidate[
                    "decision"
                ].get(
                    "gis_adjusted_score",
                    0.0,
                ),

                candidate[
                    "decision"
                ].get(
                    "mean_top_similarity",
                    0.0,
                ),

                candidate[
                    "decision"
                ].get(
                    "consistency",
                    0.0,
                ),
            ),
        reverse=True,
    )

    # --------------------------------------------------------
    # If all candidates are GIS rejected, return the strongest
    # rejected candidate for explicit REJECT persistence.
    # --------------------------------------------------------

    feasible_candidates = [
        candidate
        for candidate in candidates
        if not candidate.get(
            "gis_rejected",
            False,
        )
    ]

    if not feasible_candidates:

        rejected = candidates[0]

        return {
            "global_vehicle_id":
                rejected[
                    "global_vehicle_id"
                ],

            "decision":
                rejected[
                    "decision"
                ],

            "comparison":
                rejected[
                    "comparison"
                ],

            "gis":
                rejected[
                    "gis"
                ],

            "gis_rejected":
                True,

            "temporal_score":
                rejected.get(
                    "temporal_score",
                    0.0,
                ),
        }

    best = feasible_candidates[0]

    # --------------------------------------------------------
    # Candidate margin.
    #
    # A close second candidate means identity ambiguity.
    # --------------------------------------------------------

    if len(
        feasible_candidates
    ) > 1:

        second = (
            feasible_candidates[1]
        )

        best_score = _safe_float(
            best[
                "decision"
            ].get(
                "gis_adjusted_score",
                0.0,
            ),
            0.0,
        )

        second_score = _safe_float(
            second[
                "decision"
            ].get(
                "gis_adjusted_score",
                0.0,
            ),
            0.0,
        )

        margin = (
            best_score
            - second_score
        )

    else:

        margin = 1.0

    best[
        "candidate_margin"
    ] = _clamp01(
        margin
    )

    return best


# ============================================================
# GLOBAL VEHICLE UPDATE
# ============================================================

def _update_global_vehicle_from_match(
    global_vehicle: dict[str, Any],
    vehicle: dict[str, Any],
    embedding: list[float],
    decision: dict[str, Any],
) -> None:

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    global_vehicle[
        "last_seen"
    ] = timestamp

    global_vehicle[
        "last_camera_id"
    ] = vehicle.get(
        "camera_id"
    )

    global_vehicle[
        "last_track_id"
    ] = vehicle.get(
        "track_id"
    )

    heading = vehicle.get(
        "heading",
        vehicle.get(
            "direction"
        ),
    )

    if heading is not None:

        global_vehicle[
            "last_heading"
        ] = heading

    if (
        vehicle.get(
            "vehicle_type"
        )
        and not global_vehicle.get(
            "vehicle_type"
        )
    ):

        global_vehicle[
            "vehicle_type"
        ] = vehicle.get(
            "vehicle_type"
        )

    global_vehicle[
        "correlation_count"
    ] += 1

    if decision.get(
        "strong_match",
        False,
    ):

        global_vehicle[
            "strong_match_count"
        ] += 1

    global_vehicle[
        "last_similarity"
    ] = _clamp01(
        decision.get(
            "mean_top_similarity",
            0.0,
        )
    )

    global_vehicle[
        "best_similarity"
    ] = max(
        _clamp01(
            global_vehicle.get(
                "best_similarity",
                0.0,
            )
        ),
        _clamp01(
            decision.get(
                "best_similarity",
                0.0,
            )
        ),
    )

    global_vehicle[
        "last_correlation_score"
    ] = _clamp01(
        decision.get(
            "correlation_score",
            decision.get(
                "gis_adjusted_score",
                0.0,
            ),
        )
    )

    global_vehicle[
        "last_match_type"
    ] = "cross_camera"

    global_vehicle[
        "updated_at"
    ] = timestamp

    _add_global_embedding(
        global_vehicle,
        embedding,
    )

    _add_global_sighting(
        global_vehicle,
        vehicle,
        similarity=decision.get(
            "mean_top_similarity",
            0.0,
        ),
    )

    _record_journey_evidence(
        global_vehicle=global_vehicle,
        vehicle=vehicle,
        similarity=decision.get(
            "mean_top_similarity",
            0.0,
        ),
        match_type="cross_camera",
        supporting_frames=decision.get(
            "supporting_count",
            0,
        ),
        consistency=decision.get(
            "consistency",
            0.0,
        ),
        strong_match=decision.get(
            "strong_match",
            False,
        ),
    )

    plate_result = (
        _record_plate_observation(
            global_vehicle,
            vehicle,
        )
    )

    if plate_result is not None:

        _record_plate_journey_evidence(
            global_vehicle,
            vehicle,
            plate_result,
        )


# ============================================================
# GLOBAL ID CAPACITY
# ============================================================

def _enforce_global_vehicle_limit() -> None:

    if (
        len(_global_vehicles)
        <= MAX_GLOBAL_VEHICLES
    ):
        return

    sortable = sorted(
        _global_vehicles.items(),
        key=lambda item:
            _safe_float(
                item[1].get(
                    "last_seen",
                    0.0,
                ),
                0.0,
            ),
    )

    excess = (
        len(_global_vehicles)
        - MAX_GLOBAL_VEHICLES
    )

    for (
        global_id,
        _,
    ) in sortable[:excess]:

        _global_vehicles.pop(
            global_id,
            None
        )

        for (
            local_key,
            mapped_global_id,
        ) in list(
            _local_to_global.items()
        ):

            if (
                mapped_global_id
                == global_id
            ):

                _local_to_global.pop(
                    local_key,
                    None,
                )


# ============================================================
# GIS IMPOSSIBLE TRANSITION PERSISTENCE
# ============================================================

def _persist_impossible_gis_transition(
    vehicle: dict[str, Any],
    global_vehicle: dict[str, Any],
    gis: dict[str, Any],
    correlation: dict[str, Any],
) -> Any:

    if save_activity_event is None:
        return None

    if not isinstance(
        vehicle,
        dict,
    ):
        return None

    camera_id = vehicle.get(
        "camera_id"
    )

    if camera_id is None:
        return None

    track_id = vehicle.get(
        "track_id"
    )

    global_id = (
        correlation.get(
            "global_vehicle_id"
        )
        if isinstance(
            correlation,
            dict,
        )
        else None
    )

    source_camera = (
        global_vehicle.get(
            "last_camera_id"
        )
        if isinstance(
            global_vehicle,
            dict,
        )
        else None
    )

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    event_id = (
        "GIS-IMPOSSIBLE-"
        f"{source_camera or 'unknown'}-"
        f"{camera_id}-"
        f"{track_id if track_id is not None else 'unknown'}-"
        f"{global_id or 'unknown'}-"
        f"{int(timestamp * 1000)}"
    )

    evidence = {

        "source":
            "vehicleCorrelation",

        "event_category":
            "GIS_TRANSITION_ANOMALY",

        "event_type":
            "IMPOSSIBLE_CAMERA_TRANSITION",

        "source_camera_id":
            source_camera,

        "destination_camera_id":
            camera_id,

        "track_id":
            track_id,

        "global_vehicle_id":
            global_id,

        "distance_meters":
            gis.get(
                "distance_meters"
            ),

        "elapsed_seconds":
            gis.get(
                "elapsed_seconds"
            ),

        "implied_speed_kmh":
            gis.get(
                "implied_speed_kmh"
            ),

        "bearing_degrees":
            gis.get(
                "bearing_degrees"
            ),

        "direction_score":
            gis.get(
                "direction_score",
                0.5,
            ),

        "gis_score":
            gis.get(
                "gis_score",
                0.0,
            ),

        "travel_time_score":
            gis.get(
                "travel_time_score",
                0.0,
            ),

        "reason":
            str(
                gis.get(
                    "reason",
                    "gis_infeasible_transition",
                )
            ),
    }

    try:

        return save_activity_event(
            event_id=event_id,
            camera_id=str(
                camera_id
            ),
            rule_id=(
                "IMPOSSIBLE_CAMERA_TRANSITION"
            ),
            event_type=(
                "IMPOSSIBLE_CAMERA_TRANSITION"
            ),
            severity="HIGH",
            started_at=None,
            ended_at=None,
            track_id=(
                str(track_id)
                if track_id is not None
                else None
            ),
            global_vehicle_id=(
                str(global_id)
                if global_id
                else None
            ),
            confidence=_clamp01(
                gis.get(
                    "gis_score",
                    0.0,
                )
            ),
            status="CONFIRMED",
            evidence=evidence,
            model_name="VehicleCorrelation-GIS",
            model_version="3.0",
            metadata={
                "source_camera_id":
                    source_camera,

                "destination_camera_id":
                    camera_id,

                "gis_feasible":
                    False,

                "persistence_source":
                    "vehicleCorrelation",
            },
        )

    except Exception:

        logger.exception(
            "Failed to persist impossible GIS transition"
        )

        return None


# ============================================================
# PUBLIC CORRELATION METADATA
# ============================================================

def _build_correlation_metadata(
    vehicle: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:

    if not isinstance(
        result,
        dict,
    ):
        return result

    result = dict(
        result
    )

    match_type = str(
        result.get(
            "match_type",
            "unknown",
        )
    )

    matched = bool(
        result.get(
            "matched",
            False,
        )
    )

    global_vehicle_id = result.get(
        "global_vehicle_id"
    )

    plate_score = 0.0

    if global_vehicle_id:

        with _correlation_lock:

            global_vehicle = (
                _global_vehicles.get(
                    global_vehicle_id
                )
            )

            if global_vehicle is not None:

                plate_score = (
                    _plate_evidence_score(
                        vehicle,
                        global_vehicle,
                    )
                )

    reid_score = _clamp01(
        result.get(
            "similarity",
            result.get(
                "mean_similarity",
                0.0,
            ),
        )
    )

    if match_type == "new_global_vehicle":
        reid_score = 1.0

    vehicle_type_score = 0.0

    current_type = vehicle.get(
        "vehicle_type"
    )

    if global_vehicle_id:

        with _correlation_lock:

            gv = _global_vehicles.get(
                global_vehicle_id
            )

            if gv is not None:

                stored_type = gv.get(
                    "vehicle_type"
                )

                vehicle_type_score = (
                    1.0
                    if _vehicle_type_compatible(
                        current_type,
                        stored_type,
                    )
                    else 0.0
                )

    elif current_type is not None:

        vehicle_type_score = 1.0

    track_quality_score = _track_quality_score(
        vehicle
    )

    embedding_quality_score = _embedding_quality_score(
        vehicle
    )

    temporal_score = _clamp01(
        result.get(
            "temporal_score",
            0.0,
        )
    )

    if (
        global_vehicle_id
        and temporal_score == 0.0
    ):

        with _correlation_lock:

            gv = _global_vehicles.get(
                global_vehicle_id
            )

            if gv is not None:

                temporal_score = (
                    _temporal_score(
                        vehicle.get(
                            "timestamp"
                        ),
                        gv.get(
                            "last_seen"
                        ),
                    )
                )

    if match_type == "new_global_vehicle":
        temporal_score = 1.0

    if matched:

        decision = "MATCH"

    elif match_type in {
        "collecting_evidence",
        "new_global_vehicle",
        "same_camera_track_recovery",
    }:

        decision = "UNCERTAIN"

    elif match_type == "gis_rejected":

        decision = "REJECT"

    else:

        decision = str(
            result.get(
                "decision",
                "UNCERTAIN",
            )
        ).upper()

        if decision not in {
            "MATCH",
            "UNCERTAIN",
            "REJECT",
        }:
            decision = "UNCERTAIN"

    correlation_score = _clamp01(
        result.get(
            "correlation_score",
            0.0,
        )
    )

    gis_available = bool(
        result.get(
            "gis_available",
            False,
        )
    )

    gis_score = _clamp01(
        result.get(
            "gis_score",
            0.0,
        )
    )

    # correlation_score is canonical and is never re-weighted here.
    # Re-applying GIS at this stage would double-count GIS evidence.

    reason_map = {

        "existing_local_track":
            "Existing local tracker identity",

        "same_camera_track_recovery":
            "Recovered tracker-fragmented vehicle on same camera",

        "cross_camera":
            "Matched vehicle across cameras using multi-frame Re-ID evidence",

        "collecting_evidence":
            "Insufficient multi-frame evidence for identity correlation",

        "gis_rejected":
            "GIS rejected physically impossible camera transition",

        "new_global_vehicle":
            "Created new global vehicle identity after sufficient local evidence",

    }

    reason = reason_map.get(
        match_type,
        "Vehicle correlation result",
    )

    if plate_score >= 0.80:

        reason += (
            "; supporting ANPR plate evidence agrees"
        )

    if (
        result.get(
            "candidate_margin",
            1.0,
        )
        < CORRELATION_MARGIN_THRESHOLD
    ):

        reason += (
            "; competing candidate evidence is close"
        )

    result[
        "match_type"
    ] = match_type

    result[
        "correlation_type"
    ] = match_type

    result[
        "decision"
    ] = decision

    result[
        "correlation_score"
    ] = correlation_score

    result[
        "reid_score"
    ] = reid_score

    result[
        "plate_score"
    ] = _clamp01(
        plate_score
    )

    result[
        "vehicle_type_score"
    ] = vehicle_type_score

    result[
        "track_quality_score"
    ] = track_quality_score

    result[
        "embedding_quality_score"
    ] = embedding_quality_score

    result[
        "temporal_score"
    ] = temporal_score

    result[
        "gis_score"
    ] = gis_score

    result[
        "gis_direction_score"
    ] = _clamp01(
        result.get(
            "gis_direction_score",
            0.5,
        )
    )

    result[
        "gis_distance_meters"
    ] = result.get(
        "gis_distance_meters"
    )

    result[
        "gis_bearing_degrees"
    ] = result.get(
        "gis_bearing_degrees"
    )

    result[
        "gis_elapsed_seconds"
    ] = result.get(
        "gis_elapsed_seconds"
    )

    result[
        "gis_implied_speed_kmh"
    ] = result.get(
        "gis_implied_speed_kmh"
    )

    result[
        "gis_travel_time_score"
    ] = _clamp01(
        result.get(
            "gis_travel_time_score",
            0.0,
        )
    )

    result[
        "gis_feasible"
    ] = bool(
        result.get(
            "gis_feasible",
            True,
        )
    )

    result[
        "gis_reason"
    ] = str(
        result.get(
            "gis_reason",
            "gis_unavailable",
        )
    )

    result[
        "reason"
    ] = reason

    result[
        "plate_status"
    ] = str(
        result.get(
            "plate_status",
            "UNKNOWN",
        )
    ).upper()

    result[
        "plate_consensus_margin"
    ] = _clamp01(
        result.get(
            "plate_consensus_margin",
            0.0,
        )
    )

    result[
        "plate_support_count"
    ] = max(
        0,
        int(
            _safe_float(
                result.get(
                    "plate_support_count",
                    0,
                ),
                0.0,
            )
        ),
    )

    conflict_candidates = (
        result.get(
            "plate_conflict_candidates",
            [],
        )
    )

    if not isinstance(
        conflict_candidates,
        (list, tuple),
    ):

        conflict_candidates = []

    result[
        "plate_conflict_candidates"
    ] = [
        str(
            item
        )
        for item in conflict_candidates[:10]
    ]

    # --------------------------------------------------------
    # Confidence ladder.
    # --------------------------------------------------------

    if decision == "MATCH":

        if (
            result.get(
                "strong_match",
                False,
            )
            and correlation_score
            >= CORRELATION_STRONG_THRESHOLD
        ):

            confidence_status = (
                "CONFIRMED"
            )

        elif correlation_score >= 0.72:

            confidence_status = (
                "PROBABLE"
            )

        else:

            confidence_status = (
                "POSSIBLE"
            )

    elif decision == "UNCERTAIN":

        if correlation_score >= 0.60:

            confidence_status = (
                "POSSIBLE"
            )

        else:

            confidence_status = (
                "UNKNOWN"
            )

    else:

        confidence_status = (
            "REJECTED"
        )

    result[
        "confidence_status"
    ] = confidence_status

    result[
        "candidate_margin"
    ] = _clamp01(
        result.get(
            "candidate_margin",
            1.0,
        )
    )

    return result


# ============================================================
# PUBLIC CORRELATION API
# ============================================================

def correlate_vehicle(
    vehicle: dict[str, Any],
    embedding_override: Any = None,
) -> Optional[dict[str, Any]]:
    """
    Production vehicle correlation entry point.

    Evidence hierarchy:

        local track
            ↓
        multi-frame Re-ID
            ↓
        vehicle type
            ↓
        temporal feasibility
            ↓
        ANPR evidence
            ↓
        GIS / topology
            ↓
        candidate ranking
            ↓
        MATCH / UNCERTAIN / REJECT
            ↓
        global vehicle identity

    Important:
        Plate alone never creates a cross-camera identity.
        Re-ID alone never overrides an impossible GIS transition.
    """

    if not isinstance(
        vehicle,
        dict,
    ):
        return None

    # --------------------------------------------------------
    # OPTIONAL EMBEDDING OVERRIDE
    # --------------------------------------------------------
    #
    # Used by deterministic validation/replay callers that already
    # have a normalized Re-ID observation available.
    #
    # Normal production callers continue using:
    #
    #     correlate_vehicle(vehicle)
    #
    # and therefore remain completely backward compatible.
    #
    if embedding_override is not None:
        vehicle = dict(vehicle)
        vehicle["embedding"] = embedding_override

    camera_id = vehicle.get(
        "camera_id"
    )

    track_id = vehicle.get(
        "track_id"
    )

    if (
        camera_id is None
        or track_id is None
    ):
        return None

    embedding = vehicle.get(
        "embedding"
    )

    if not _embedding_quality(
        embedding
    ):
        return None

    embedding = _normalize_embedding(
        embedding
    )

    if embedding is None:
        return None

# ============================================================
# PUBLIC STATE ACCESS
# ============================================================

def get_global_vehicle(
    global_vehicle_id: str,
):
    """
    Return a defensive copy.

    Callers cannot mutate the live correlation state.
    """

    if not isinstance(
        global_vehicle_id,
        str,
    ):
        return None

    global_vehicle_id = (
        global_vehicle_id.strip()
    )

    if not global_vehicle_id:
        return None

    with _correlation_lock:

        vehicle = (
            _global_vehicles.get(
                global_vehicle_id
            )
        )

        if vehicle is None:
            return None

        return copy.deepcopy(
            vehicle
        )


def get_all_global_vehicles():
    with _correlation_lock:

        return [
            copy.deepcopy(
                vehicle
            )
            for vehicle
            in _global_vehicles.values()
        ]


def get_global_vehicle_id(
    camera_id,
    track_id,
):
    local_key = (
        f"{camera_id}:{track_id}"
    )

    with _correlation_lock:

        return _local_to_global.get(
            local_key
        )


# ============================================================
# VEHICLE JOURNEY
# ============================================================

def get_vehicle_journey(
    global_vehicle_id,
):
    """
    Return a defensive copy of the observed vehicle journey.

    This is an observed camera journey, not a claim of complete
    physical road travel.
    """

    if not isinstance(
        global_vehicle_id,
        str,
    ):
        return []

    global_vehicle_id = (
        global_vehicle_id.strip()
    )

    if not global_vehicle_id:
        return []

    with _correlation_lock:

        global_vehicle = (
            _global_vehicles.get(
                global_vehicle_id
            )
        )

        if global_vehicle is None:
            return []

        journey = (
            global_vehicle.get(
                "journey",
                [],
            )
        )

        return [
            copy.deepcopy(
                event
            )
            for event
            in journey
            if isinstance(
                event,
                dict,
            )
        ]


# ============================================================
# CAMERA VISITS
# ============================================================

def get_vehicle_camera_visits(
    global_vehicle_id,
):
    if not isinstance(
        global_vehicle_id,
        str,
    ):
        return {}

    global_vehicle_id = (
        global_vehicle_id.strip()
    )

    if not global_vehicle_id:
        return {}

    with _correlation_lock:

        global_vehicle = (
            _global_vehicles.get(
                global_vehicle_id
            )
        )

        if global_vehicle is None:
            return {}

        visits = (
            global_vehicle.get(
                "camera_visits",
                {},
            )
        )

        result = {}

        for (
            camera_id,
            data,
        ) in visits.items():

            if not isinstance(
                data,
                dict,
            ):
                continue

            result[
                camera_id
            ] = {

                "first_seen":
                    data.get(
                        "first_seen"
                    ),

                "last_seen":
                    data.get(
                        "last_seen"
                    ),

                "observation_count":
                    int(
                        data.get(
                            "observation_count",
                            0,
                        )
                    ),

                "track_ids":
                    list(
                        data.get(
                            "track_ids",
                            [],
                        )
                    ),
            }

        return result


# ============================================================
# PLATE EVIDENCE API
# ============================================================

def get_vehicle_plate_evidence(
    global_vehicle_id,
):
    """
    Return sanitized established ANPR evidence.
    """

    if not isinstance(
        global_vehicle_id,
        str,
    ):
        return None

    global_vehicle_id = (
        global_vehicle_id.strip()
    )

    if not global_vehicle_id:
        return None

    with _correlation_lock:

        global_vehicle = (
            _global_vehicles.get(
                global_vehicle_id
            )
        )

        if global_vehicle is None:
            return None

        return {

            "plate":
                global_vehicle.get(
                    "plate"
                ),

            "plate_confidence":
                _clamp01(
                    global_vehicle.get(
                        "plate_confidence",
                        0.0,
                    )
                ),

            "plate_support_count":
                int(
                    global_vehicle.get(
                        "plate_support_count",
                        0,
                    )
                ),

            "plate_match_strength":
                global_vehicle.get(
                    "plate_match_strength",
                    "none",
                ),

            "plate_status":
                str(
                    global_vehicle.get(
                        "plate_status",
                        "UNKNOWN",
                    )
                ).upper(),

            "plate_consensus_margin":
                _clamp01(
                    global_vehicle.get(
                        "plate_consensus_margin",
                        0.0,
                    )
                ),

            "plate_conflict_candidates":
                list(
                    global_vehicle.get(
                        "plate_conflict_candidates",
                        [],
                    )
                ),
        }


# ============================================================
# CLEANUP
# ============================================================

def cleanup_global_vehicles(
    max_age_seconds: Optional[float] = None,
) -> int:
    """
    Remove stale in-memory global identities.

    PostgreSQL remains the durable intelligence history.
    """

    if max_age_seconds is None:

        max_age_seconds = max(
            MAX_SIGHTING_GAP_SECONDS,
            30 * 60,
        )

    max_age_seconds = max(
        1.0,
        float(
            max_age_seconds
        ),
    )

    now = time.time()

    removed = 0

    with _correlation_lock:

        stale_ids = []

        for (
            global_id,
            vehicle,
        ) in _global_vehicles.items():

            last_seen = _safe_float(
                vehicle.get(
                    "last_seen",
                    0.0,
                ),
                0.0,
            )

            if (
                now
                - last_seen
                > max_age_seconds
            ):

                stale_ids.append(
                    global_id
                )

        for global_id in stale_ids:

            _global_vehicles.pop(
                global_id,
                None,
            )

            removed += 1

        # Remove local mappings pointing to deleted identities.
        for (
            local_key,
            global_id,
        ) in list(
            _local_to_global.items()
        ):

            if (
                global_id
                not in _global_vehicles
            ):

                _local_to_global.pop(
                    local_key,
                    None,
                )

        _cleanup_local_track_evidence(
            now
        )

    return removed


# ============================================================
# IDENTITY CORRECTION
# ============================================================

def reassign_global_identity(source_global_id: str, target_global_id: str, local_keys: list[str] | None = None) -> bool:
    """Apply an audited operator correction to live correlation state."""
    source = str(source_global_id or "").strip()
    target = str(target_global_id or "").strip()
    if not source or not target:
        return False
    with _correlation_lock:
        source_state = _global_vehicles.get(source)
        target_state = _global_vehicles.get(target)
        if target_state is None and source_state is not None:
            target_state = copy.deepcopy(source_state)
            target_state["global_vehicle_id"] = target
            _global_vehicles[target] = target_state
        if local_keys:
            allowed = set(str(k) for k in local_keys)
            for key in list(_local_to_global):
                if _local_to_global.get(key) == source and key in allowed:
                    _local_to_global[key] = target
        else:
            for key, value in list(_local_to_global.items()):
                if value == source:
                    _local_to_global[key] = target
        if source != target and source_state is not None:
            source_state["status"] = "CORRECTED"
            source_state["corrected_to"] = target
        return True


# ============================================================
# RESET / TEST SUPPORT
# ============================================================

def clear_correlation_state() -> None:
    """
    Clear all in-memory correlation state.

    Useful for:
        - camera/session restart
        - tests
        - controlled replay
        - development
    """

    with _correlation_lock:

        _global_vehicles.clear()
        _local_to_global.clear()
        _local_track_evidence.clear()


def clear_camera_correlation_state(
    camera_id: str,
) -> None:
    """
    Clear temporary local-track mappings/evidence for one camera.

    Existing global identities are retained because they may have
    observations from other cameras.
    """

    if not camera_id:
        return

    camera_id = str(
        camera_id
    ).strip()

    if not camera_id:
        return

    with _correlation_lock:

        for local_key in list(
            _local_to_global.keys()
        ):

            if local_key.startswith(
                f"{camera_id}:"
            ):

                _local_to_global.pop(
                    local_key,
                    None,
                )

        for local_key in list(
            _local_track_evidence.keys()
        ):

            if local_key.startswith(
                f"{camera_id}:"
            ):

                _local_track_evidence.pop(
                    local_key,
                    None,
                )


# ============================================================
# HEALTH
# ============================================================

def correlation_health() -> dict[str, Any]:
    with _correlation_lock:

        active_global = len(
            _global_vehicles
        )

        active_local_tracks = len(
            _local_track_evidence
        )

        active_mappings = len(
            _local_to_global
        )

        strong_matches = sum(
            int(
                vehicle.get(
                    "strong_match_count",
                    0,
                )
            )
            for vehicle
            in _global_vehicles.values()
        )

    return {

        "engine":
            "VehicleCorrelation",

        "version":
            "3.0",

        "global_vehicle_count":
            active_global,

        "local_track_evidence_count":
            active_local_tracks,

        "local_global_mapping_count":
            active_mappings,

        "strong_match_count":
            strong_matches,

        "reid_frame_threshold":
            REID_FRAME_THRESHOLD,

        "reid_support_threshold":
            REID_SUPPORT_THRESHOLD,

        "reid_strong_threshold":
            REID_STRONG_THRESHOLD,

        "min_supporting_embeddings":
            MIN_SUPPORTING_EMBEDDINGS,

        "min_strong_supporting_embeddings":
            MIN_STRONG_SUPPORTING_EMBEDDINGS,

        "gis_available":
            evaluate_vehicle_transition
            is not None
            or get_camera_gis
            is not None,

        "persistence_available":
            save_correlation_decision
            is not None,

        "decision_states":
            [
                "MATCH",
                "UNCERTAIN",
                "REJECT",
            ],

        "production_evidence":
            [
                "multi_frame_reid",
                "reid_quality",
                "anpr",
                "temporal_feasibility",
                "gis_topology",
                "vehicle_type",
                "track_quality",
                "candidate_margin",
            ],

        "confidence_states":
            [
                "CONFIRMED",
                "PROBABLE",
                "POSSIBLE",
                "UNKNOWN",
                "REJECTED",
            ],
    }


# ============================================================
# OPTIONAL EXPLICIT DECISION PERSISTENCE
# ============================================================

def persist_correlation_decision(
    vehicle: dict[str, Any],
    correlation: dict[str, Any],
):
    """
    Optional helper for callers that want the correlation engine
    itself to persist an auditable decision.

    main.py may also persist the decision. This helper therefore
    remains opt-in and does not automatically write every frame.
    """

    if save_correlation_decision is None:
        return None

    if not isinstance(
        vehicle,
        dict,
    ):
        return None

    if not isinstance(
        correlation,
        dict,
    ):
        return None

    decision = str(
        correlation.get(
            "decision",
            "UNCERTAIN",
        )
    ).upper()

    if decision not in {
        "MATCH",
        "UNCERTAIN",
        "REJECT",
    }:

        decision = "UNCERTAIN"

    try:

        return save_correlation_decision(

            source_camera_id=str(
                vehicle.get(
                    "camera_id",
                    "",
                )
            ),

            source_track_id=(
                str(
                    vehicle.get(
                        "track_id"
                    )
                )
                if vehicle.get(
                    "track_id"
                ) is not None
                else None
            ),

            candidate_global_vehicle_id=(
                str(
                    correlation.get(
                        "global_vehicle_id"
                    )
                )
                if correlation.get(
                    "global_vehicle_id"
                )
                else None
            ),

            decision=decision,

            final_score=_clamp01(
                correlation.get(
                    "correlation_score",
                    0.0,
                )
            ),

            reid_score=_clamp01(
                correlation.get(
                    "reid_score",
                    correlation.get(
                        "similarity",
                        0.0,
                    ),
                )
            ),

            plate_score=_clamp01(
                correlation.get(
                    "plate_score",
                    0.0,
                )
            ),

            vehicle_type_score=_clamp01(
                correlation.get(
                    "vehicle_type_score",
                    0.0,
                )
            ),

            track_quality_score=_clamp01(
                correlation.get(
                    "track_quality_score",
                    _track_quality_score(vehicle),
                )
            ),

            temporal_score=_clamp01(
                correlation.get(
                    "temporal_score",
                    0.0,
                )
            ),

            gis_score=_clamp01(
                correlation.get(
                    "gis_score",
                    0.0,
                )
            ),

            direction_score=_clamp01(
                correlation.get(
                    "gis_direction_score",
                    0.5,
                )
            ),

            reason=str(
                correlation.get(
                    "reason",
                    "Vehicle correlation decision",
                )
            )[:2000],

            model_name=(
                "VehicleCorrelation"
            ),

            model_version=(
                "3.0"
            ),
        )

    except Exception:

        logger.exception(
            "Correlation decision persistence failed"
        )

        return None