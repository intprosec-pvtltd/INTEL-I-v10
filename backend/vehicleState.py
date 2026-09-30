from __future__ import annotations

import copy
import logging
import math
import os
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ============================================================
# OPTIONAL PERSISTENCE
# ============================================================

try:
    from services.intelligence_persistence import (
        save_vehicle_observation,
        save_vehicle_state,
    )
except ImportError:
    save_vehicle_observation = None
    save_vehicle_state = None


# ============================================================
# CONFIGURATION
# ============================================================

STATE_VERSION = os.getenv(
    "VEHICLE_STATE_VERSION",
    "3.0-production",
)

MAX_ACTIVE_VEHICLES = max(
    100,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_ACTIVE_VEHICLES",
            "10000",
        )
    ),
)

MAX_OBSERVATIONS_PER_VEHICLE = max(
    10,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_OBSERVATIONS",
            "200",
        )
    ),
)

MAX_TRANSITIONS_PER_VEHICLE = max(
    10,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_TRANSITIONS",
            "200",
        )
    ),
)

MAX_ACTIVITY_EVENTS_PER_VEHICLE = max(
    10,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_ACTIVITY_EVENTS",
            "200",
        )
    ),
)

MAX_PLATE_HISTORY = max(
    5,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_PLATE_HISTORY",
            "50",
        )
    ),
)

MAX_REID_HISTORY = max(
    5,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_REID_HISTORY",
            "50",
        )
    ),
)

MAX_CAMERA_HISTORY = max(
    10,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_CAMERA_HISTORY",
            "100",
        )
    ),
)

MAX_STATE_HISTORY = max(
    10,
    int(
        os.getenv(
            "VEHICLE_STATE_MAX_STATE_HISTORY",
            "100",
        )
    ),
)

VEHICLE_STATE_TTL_SECONDS = max(
    60.0,
    float(
        os.getenv(
            "VEHICLE_STATE_TTL_SECONDS",
            str(60 * 60),
        )
    ),
)

ACTIVE_VEHICLE_TIMEOUT_SECONDS = max(
    5.0,
    float(
        os.getenv(
            "ACTIVE_VEHICLE_TIMEOUT_SECONDS",
            "30",
        )
    ),
)

UNCERTAIN_RETENTION_SECONDS = max(
    30.0,
    float(
        os.getenv(
            "UNCERTAIN_RETENTION_SECONDS",
            str(10 * 60),
        )
    ),
)

MIN_CONFIRMED_CORRELATION_SCORE = float(
    os.getenv(
        "MIN_CONFIRMED_CORRELATION_SCORE",
        "0.84",
    )
)

MIN_PROBABLE_CORRELATION_SCORE = float(
    os.getenv(
        "MIN_PROBABLE_CORRELATION_SCORE",
        "0.72",
    )
)

MIN_POSSIBLE_CORRELATION_SCORE = float(
    os.getenv(
        "MIN_POSSIBLE_CORRELATION_SCORE",
        "0.55",
    )
)

PLATE_CONFIRMED_THRESHOLD = float(
    os.getenv(
        "PLATE_CONFIRMED_THRESHOLD",
        "0.80",
    )
)

PLATE_STRONG_THRESHOLD = float(
    os.getenv(
        "PLATE_STRONG_THRESHOLD",
        "0.92",
    )
)

REID_CONFIRMED_THRESHOLD = float(
    os.getenv(
        "REID_CONFIRMED_THRESHOLD",
        "0.84",
    )
)

REID_STRONG_THRESHOLD = float(
    os.getenv(
        "REID_STRONG_THRESHOLD",
        "0.90",
    )
)

GIS_CONFIRMED_THRESHOLD = float(
    os.getenv(
        "GIS_CONFIRMED_THRESHOLD",
        "0.65",
    )
)

MAX_SPEED_KMH = max(
    1.0,
    float(
        os.getenv(
            "VEHICLE_STATE_MAX_SPEED_KMH",
            "180",
        )
    ),
)

MAX_REASONABLE_ACCELERATION = max(
    0.1,
    float(
        os.getenv(
            "VEHICLE_STATE_MAX_ACCELERATION",
            "12.0",
        )
    ),
)

MIN_OBSERVATIONS_FOR_ACTIVITY = max(
    1,
    int(
        os.getenv(
            "VEHICLE_STATE_MIN_ACTIVITY_OBSERVATIONS",
            "2",
        )
    ),
)

try:
    MIN_REID_QUALITY_FOR_CORRELATION = max(
        0.0,
        min(
            1.0,
            float(
                os.getenv(
                    "VEHICLE_STATE_MIN_REID_QUALITY",
                    "0.35",
                )
            ),
        ),
    )
except (TypeError, ValueError):
    MIN_REID_QUALITY_FOR_CORRELATION = 0.35

MAX_OBSERVATION_CLOCK_SKEW_SECONDS = max(
    0.0,
    float(
        os.getenv(
            "VEHICLE_STATE_MAX_CLOCK_SKEW_SECONDS",
            "5.0",
        )
    ),
)

MAX_TRANSITION_ELAPSED_SECONDS = max(
    1.0,
    float(
        os.getenv(
            "VEHICLE_STATE_MAX_TRANSITION_ELAPSED_SECONDS",
            str(60 * 60),
        )
    ),
)

MIN_OBSERVATIONS_FOR_CONFIRMED_STATE = max(
    1,
    int(
        os.getenv(
            "VEHICLE_STATE_MIN_CONFIRMED_OBSERVATIONS",
            "2",
        )
    ),
)


# ============================================================
# THREAD SAFE STATE
# ============================================================

_state_lock = threading.RLock()

_vehicle_states: dict[
    str,
    dict[str, Any],
] = {}


# ============================================================
# GENERIC HELPERS
# ============================================================

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

    if not math.isfinite(number):
        return default

    return number


def _clamp01(
    value: Any,
) -> float:

    number = _safe_float(
        value,
        0.0,
    )

    return max(
        0.0,
        min(
            1.0,
            number,
        ),
    )


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


def _normalize_vehicle_type(
    value: Any,
) -> Optional[str]:

    value = _safe_string(
        value,
        64,
    )

    if value is None:
        return None

    value = value.lower()

    aliases = {
        "motorbike": "motorcycle",
        "bike": "motorcycle",
        "lorry": "truck",
    }

    return aliases.get(
        value,
        value,
    )


def _copy(
    value: Any,
) -> Any:

    try:
        return copy.deepcopy(
            value
        )
    except Exception:
        return value


# ============================================================
# ENUM-LIKE STATE VALUES
# ============================================================

STATE_ACTIVE = "ACTIVE"
STATE_INACTIVE = "INACTIVE"
STATE_UNCERTAIN = "UNCERTAIN"
STATE_CONFIRMED = "CONFIRMED"
STATE_PROBABLE = "PROBABLE"
STATE_POSSIBLE = "POSSIBLE"
STATE_REJECTED = "REJECTED"
STATE_COMPLETED = "COMPLETED"


# ============================================================
# CONFIDENCE RESOLUTION
# ============================================================

def _resolve_confidence_status(
    correlation: dict[str, Any],
    vehicle: dict[str, Any],
    state: Optional[dict[str, Any]] = None,
) -> str:

    if not isinstance(
        correlation,
        dict,
    ):
        return STATE_UNCERTAIN

    decision = str(
        correlation.get(
            "decision",
            "",
        )
    ).upper()

    if decision == "REJECT":
        return STATE_REJECTED

    explicit_status = str(
        correlation.get(
            "confidence_status",
            "",
        )
    ).upper()

    if explicit_status in {
        STATE_CONFIRMED,
        STATE_PROBABLE,
        STATE_POSSIBLE,
        STATE_UNCERTAIN,
        STATE_REJECTED,
    }:

        if (
            explicit_status
            == STATE_CONFIRMED
        ):
            return STATE_CONFIRMED

        if (
            explicit_status
            == STATE_PROBABLE
        ):
            return STATE_PROBABLE

        if (
            explicit_status
            == STATE_POSSIBLE
        ):
            return STATE_POSSIBLE

        if (
            explicit_status
            == STATE_REJECTED
        ):
            return STATE_REJECTED

        return STATE_UNCERTAIN

    score = _clamp01(
        correlation.get(
            "correlation_score",
            0.0,
        )
    )

    reid_score = _clamp01(
        correlation.get(
            "reid_score",
            correlation.get(
                "similarity",
                0.0,
            ),
        )
    )

    plate_score = _clamp01(
        correlation.get(
            "plate_score",
            0.0,
        )
    )

    gis_score = _clamp01(
        correlation.get(
            "gis_score",
            0.0,
        )
    )

    strong_match = bool(
        correlation.get(
            "strong_match",
            False,
        )
    )

    if (
        decision == "MATCH"
        and strong_match
        and score
        >= MIN_CONFIRMED_CORRELATION_SCORE
        and reid_score
        >= REID_CONFIRMED_THRESHOLD
    ):

        return STATE_CONFIRMED

    if (
        decision == "MATCH"
        and score
        >= MIN_PROBABLE_CORRELATION_SCORE
    ):

        if (
            plate_score
            >= PLATE_CONFIRMED_THRESHOLD
            or reid_score
            >= REID_CONFIRMED_THRESHOLD
            or gis_score
            >= GIS_CONFIRMED_THRESHOLD
        ):

            return STATE_PROBABLE

    if (
        score
        >= MIN_POSSIBLE_CORRELATION_SCORE
    ):

        return STATE_POSSIBLE

    return STATE_UNCERTAIN


# ============================================================
# EVIDENCE STRENGTH
# ============================================================

def _resolve_evidence_strength(
    correlation: dict[str, Any],
    vehicle: dict[str, Any],
) -> str:

    if not isinstance(
        correlation,
        dict,
    ):
        return "NONE"

    score = _clamp01(
        correlation.get(
            "correlation_score",
            0.0,
        )
    )

    reid = _clamp01(
        correlation.get(
            "reid_score",
            correlation.get(
                "similarity",
                0.0,
            ),
        )
    )

    plate = _clamp01(
        correlation.get(
            "plate_score",
            0.0,
        )
    )

    gis = _clamp01(
        correlation.get(
            "gis_score",
            0.0,
        )
    )

    support = max(
        0,
        int(
            _safe_float(
                correlation.get(
                    "supporting_frames",
                    0,
                ),
                0.0,
            )
        ),
    )

    strong_match = bool(
        correlation.get(
            "strong_match",
            False,
        )
    )

    if (
        strong_match
        and score >= 0.84
        and support >= 4
        and reid >= 0.88
    ):

        return "STRONG"

    if (
        score >= 0.72
        and (
            reid >= 0.80
            or plate >= 0.80
            or gis >= 0.65
        )
    ):

        return "GOOD"

    if score >= 0.55:
        return "MODERATE"

    if (
        reid > 0.0
        or plate > 0.0
        or gis > 0.0
    ):

        return "WEAK"

    return "NONE"


# ============================================================
# OBSERVATION NORMALIZATION
# ============================================================

def _normalize_bbox(
    bbox: Any,
) -> Optional[list[int]]:

    if not isinstance(
        bbox,
        (list, tuple),
    ):
        return None

    if len(bbox) < 4:
        return None

    try:

        result = [
            int(
                float(
                    bbox[0]
                )
            ),
            int(
                float(
                    bbox[1]
                )
            ),
            int(
                float(
                    bbox[2]
                )
            ),
            int(
                float(
                    bbox[3]
                )
            ),
        ]

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None

    x1, y1, x2, y2 = result

    if (
        x2 < x1
        or y2 < y1
    ):
        return None

    return result


def _normalize_center(
    center: Any,
) -> Optional[list[float]]:

    if not isinstance(
        center,
        (list, tuple),
    ):
        return None

    if len(center) < 2:
        return None

    try:

        x = float(
            center[0]
        )

        y = float(
            center[1]
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None

    if not (
        math.isfinite(x)
        and math.isfinite(y)
    ):
        return None

    return [
        x,
        y,
    ]


def _normalize_heading(
    vehicle: dict[str, Any],
) -> Optional[float]:

    heading = vehicle.get(
        "heading",
        vehicle.get(
            "direction"
        ),
    )

    if heading is None:
        return None

    heading = _safe_float(
        heading,
        -1.0,
    )

    if heading < 0.0:
        return None

    return heading % 360.0


def _normalize_observation(
    vehicle: dict[str, Any],
    correlation: dict[str, Any],
) -> dict[str, Any]:

    timestamp = _safe_timestamp(
        vehicle.get(
            "timestamp"
        )
    )

    camera_id = _safe_string(
        vehicle.get(
            "camera_id"
        ),
        128,
    )

    track_id = _safe_string(
        vehicle.get(
            "track_id"
        ),
        128,
    )

    global_vehicle_id = (
        _safe_string(
            correlation.get(
                "global_vehicle_id"
            ),
            128,
        )
    )

    bbox = _normalize_bbox(
        vehicle.get(
            "bbox"
        )
    )

    center = _normalize_center(
        vehicle.get(
            "center"
        )
    )

    confidence = _clamp01(
        vehicle.get(
            "confidence",
            0.0,
        )
    )

    plate = _safe_string(
        correlation.get(
            "plate",
            vehicle.get(
                "plate"
            ),
        ),
        32,
    )

    plate_confidence = _clamp01(
        correlation.get(
            "plate_confidence",
            vehicle.get(
                "plate_confidence",
                0.0,
            ),
        )
    )

    observation = {

        "timestamp":
            timestamp,

        "camera_id":
            camera_id,

        "track_id":
            track_id,

        "global_vehicle_id":
            global_vehicle_id,

        "vehicle_type":
            _normalize_vehicle_type(
                vehicle.get(
                    "vehicle_type"
                )
            ),

        "confidence":
            confidence,

        "bbox":
            bbox,

        "center":
            center,

        "heading":
            _normalize_heading(
                vehicle
            ),

        "plate":
            plate,

        "plate_confidence":
            plate_confidence,

        "plate_status":
            str(
                correlation.get(
                    "plate_status",
                    "UNKNOWN",
                )
            ).upper(),

        "plate_support_count":
            max(
                0,
                int(
                    _safe_float(
                        correlation.get(
                            "plate_support_count",
                            0,
                        ),
                        0.0,
                    )
                ),
            ),

        "reid_score":
            _clamp01(
                correlation.get(
                    "reid_score",
                    correlation.get(
                        "similarity",
                        0.0,
                    ),
                )
            ),

        "best_reid_score":
            _clamp01(
                correlation.get(
                    "best_similarity",
                    correlation.get(
                        "similarity",
                        0.0,
                    ),
                )
            ),

        "reid_consistency":
            _clamp01(
                correlation.get(
                    "consistency",
                    0.0,
                )
            ),

        # Observation quality is deliberately separate from identity
        # similarity/confidence.
        "reid_observation_quality":
            _clamp01(
                correlation.get(
                    "reid_observation_quality",
                    vehicle.get(
                        "reid_observation_quality",
                        0.0,
                    ),
                )
            ),

        "crop_quality":
            _clamp01(
                correlation.get(
                    "crop_quality",
                    vehicle.get(
                        "crop_quality",
                        0.0,
                    ),
                )
            ),

        "embedding_quality":
            _clamp01(
                correlation.get(
                    "embedding_quality",
                    vehicle.get(
                        "embedding_quality",
                        0.0,
                    ),
                )
            ),

        "supporting_frames":
            max(
                0,
                int(
                    _safe_float(
                        correlation.get(
                            "supporting_frames",
                            0,
                        ),
                        0.0,
                    )
                ),
            ),

        "correlation_score":
            _clamp01(
                correlation.get(
                    "correlation_score",
                    0.0,
                )
            ),

        "correlation_decision":
            str(
                correlation.get(
                    "decision",
                    "UNCERTAIN",
                )
            ).upper(),

        "match_type":
            str(
                correlation.get(
                    "match_type",
                    "unknown",
                )
            )[:64],

        "evidence_strength":
            _resolve_evidence_strength(
                correlation,
                vehicle,
            ),

        "confidence_status":
            _resolve_confidence_status(
                correlation,
                vehicle,
            ),

        "temporal_score":
            _clamp01(
                correlation.get(
                    "temporal_score",
                    0.0,
                )
            ),

        "gis_score":
            _clamp01(
                correlation.get(
                    "gis_score",
                    0.0,
                )
            ),

        "gis_direction_score":
            _clamp01(
                correlation.get(
                    "gis_direction_score",
                    0.5,
                )
            ),

        "gis_distance_meters":
            correlation.get(
                "gis_distance_meters"
            ),

        "gis_elapsed_seconds":
            correlation.get(
                "gis_elapsed_seconds"
            ),

        "gis_implied_speed_kmh":
            correlation.get(
                "gis_implied_speed_kmh"
            ),

        "gis_feasible":
            bool(
                correlation.get(
                    "gis_feasible",
                    True,
                )
            ),

        "gis_reason":
            str(
                correlation.get(
                    "gis_reason",
                    "unknown",
                )
            )[:256],

        "candidate_margin":
            _clamp01(
                correlation.get(
                    "candidate_margin",
                    1.0,
                )
            ),

        "strong_match":
            bool(
                correlation.get(
                    "strong_match",
                    False,
                )
            ),

        "model_version":
            STATE_VERSION,
    }

    return observation


# ============================================================
# INITIAL STATE
# ============================================================

def _create_state(
    observation: dict[str, Any],
) -> dict[str, Any]:

    now = observation[
        "timestamp"
    ]

    global_vehicle_id = observation.get(
        "global_vehicle_id"
    )

    state = {

        "state_version":
            STATE_VERSION,

        "global_vehicle_id":
            global_vehicle_id,

        "vehicle_type":
            observation.get(
                "vehicle_type"
            ),

        "state":
            STATE_ACTIVE,

        "confidence_status":
            observation.get(
                "confidence_status",
                STATE_UNCERTAIN,
            ),

        "evidence_strength":
            observation.get(
                "evidence_strength",
                "NONE",
            ),

        "first_seen":
            now,

        "last_seen":
            now,

        "last_observation_at":
            now,

        "last_camera_id":
            observation.get(
                "camera_id"
            ),

        "last_track_id":
            observation.get(
                "track_id"
            ),

        "last_heading":
            observation.get(
                "heading"
            ),

        "last_center":
            observation.get(
                "center"
            ),

        "last_bbox":
            observation.get(
                "bbox"
            ),

        "last_speed_kmh":
            None,

        "last_acceleration_kmh_s":
            None,

        "max_speed_kmh":
            0.0,

        "average_speed_kmh":
            0.0,

        "total_distance_meters":
            0.0,

        "observation_count":
            0,

        "confirmed_observation_count":
            0,

        "probable_observation_count":
            0,

        "possible_observation_count":
            0,

        "uncertain_observation_count":
            0,

        "rejected_observation_count":
            0,

        "strong_match_count":
            0,

        "cross_camera_match_count":
            0,

        "same_camera_recovery_count":
            0,

        "local_track_count":
            0,

        "camera_count":
            0,

        "camera_history":
            [],

        "observations":
            [],

        "transitions":
            [],

        "activity_events":
            [],

        "state_history":
            [],

        "plate":
            observation.get(
                "plate"
            ),

        "plate_confidence":
            observation.get(
                "plate_confidence",
                0.0,
            ),

        "plate_status":
            observation.get(
                "plate_status",
                "UNKNOWN",
            ),

        "plate_support_count":
            observation.get(
                "plate_support_count",
                0,
            ),

        "plate_history":
            [],

        "plate_conflict_candidates":
            [],

        "reid_history":
            [],

        "correlation_history":
            [],

        "gis_history":
            [],

        "last_correlation_score":
            observation.get(
                "correlation_score",
                0.0,
            ),

        "best_correlation_score":
            observation.get(
                "correlation_score",
                0.0,
            ),

        "best_reid_score":
            observation.get(
                "best_reid_score",
                0.0,
            ),

        "best_reid_observation_quality":
            observation.get(
                "reid_observation_quality",
                0.0,
            ),

        "best_crop_quality":
            observation.get(
                "crop_quality",
                0.0,
            ),

        "best_embedding_quality":
            observation.get(
                "embedding_quality",
                0.0,
            ),

        "best_plate_confidence":
            observation.get(
                "plate_confidence",
                0.0,
            ),

        "best_gis_score":
            observation.get(
                "gis_score",
                0.0,
            ),

        "created_at":
            now,

        "updated_at":
            now,
    }

    return state


# ============================================================
# STATE HISTORY
# ============================================================

def _record_state_transition(
    state: dict[str, Any],
    old_state: str,
    new_state: str,
    timestamp: float,
    reason: str,
) -> None:

    if (
        old_state
        == new_state
    ):
        return

    history = state.setdefault(
        "state_history",
        [],
    )

    history.append(
        {
            "timestamp":
                timestamp,

            "from":
                old_state,

            "to":
                new_state,

            "reason":
                str(
                    reason
                )[:512],
        }
    )

    if (
        len(history)
        > MAX_STATE_HISTORY
    ):

        del history[
            :-MAX_STATE_HISTORY
        ]


def _set_state(
    state: dict[str, Any],
    new_state: str,
    timestamp: float,
    reason: str,
) -> None:

    old_state = str(
        state.get(
            "state",
            STATE_UNCERTAIN,
        )
    )

    if (
        new_state
        not in {
            STATE_ACTIVE,
            STATE_INACTIVE,
            STATE_UNCERTAIN,
            STATE_CONFIRMED,
            STATE_PROBABLE,
            STATE_POSSIBLE,
            STATE_REJECTED,
            STATE_COMPLETED,
        }
    ):
        new_state = STATE_UNCERTAIN

    _record_state_transition(
        state,
        old_state,
        new_state,
        timestamp,
        reason,
    )

    state[
        "state"
    ] = new_state


# ============================================================
# CONFIDENCE STATE TRANSITION
# ============================================================

def _update_confidence_state(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> None:

    timestamp = observation[
        "timestamp"
    ]

    current = str(
        state.get(
            "confidence_status",
            STATE_UNCERTAIN,
        )
    )

    incoming = str(
        observation.get(
            "confidence_status",
            STATE_UNCERTAIN,
        )
    )

    incoming_decision = str(
        observation.get(
            "correlation_decision",
            "UNCERTAIN",
        )
    ).upper()

    if incoming == STATE_REJECTED:
        return

    # Never downgrade an already confirmed identity because one
    # weak frame arrived.
    if current == STATE_CONFIRMED:

        if incoming == STATE_CONFIRMED:
            return

        return

    rank = {
        STATE_UNCERTAIN: 0,
        STATE_POSSIBLE: 1,
        STATE_PROBABLE: 2,
        STATE_CONFIRMED: 3,
        STATE_REJECTED: -1,
    }

    current_rank = rank.get(
        current,
        0,
    )

    incoming_rank = rank.get(
        incoming,
        0,
    )

    if incoming_rank >= current_rank:

        state[
            "confidence_status"
        ] = incoming

        return

    # A hard negative/contradiction should not automatically
    # erase a confirmed identity. It is recorded as evidence.
    if (
        incoming_decision
        == "REJECT"
    ):

        state[
            "last_negative_evidence"
        ] = {
            "timestamp":
                timestamp,

            "reason":
                observation.get(
                    "gis_reason",
                    "rejected",
                ),

            "global_vehicle_id":
                observation.get(
                    "global_vehicle_id"
                ),
        }


# ============================================================
# PLATE STATE
# ============================================================

def _update_plate_state(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> None:

    plate = observation.get(
        "plate"
    )

    confidence = _clamp01(
        observation.get(
            "plate_confidence",
            0.0,
        )
    )

    if not plate:
        return

    timestamp = observation[
        "timestamp"
    ]

    history = state.setdefault(
        "plate_history",
        [],
    )

    history.append(
        {
            "plate":
                plate,

            "confidence":
                confidence,

            "status":
                observation.get(
                    "plate_status",
                    "UNKNOWN",
                ),

            "support_count":
                observation.get(
                    "plate_support_count",
                    0,
                ),

            "timestamp":
                timestamp,

            "camera_id":
                observation.get(
                    "camera_id"
                ),
        }
    )

    if (
        len(history)
        > MAX_PLATE_HISTORY
    ):

        del history[
            :-MAX_PLATE_HISTORY
        ]

    # Stable/confirmed plate from the correlation layer.
    incoming_status = str(
        observation.get(
            "plate_status",
            "UNKNOWN",
        )
    ).upper()

    incoming_support = max(
        0,
        int(
            observation.get(
                "plate_support_count",
                0,
            )
        ),
    )

    current_plate = state.get(
        "plate"
    )

    current_confidence = _clamp01(
        state.get(
            "plate_confidence",
            0.0,
        )
    )

    if (
        incoming_status == "STABLE"
        and confidence
        >= PLATE_CONFIRMED_THRESHOLD
        and incoming_support >= 2
    ):

        if (
            current_plate is None
            or confidence
            > current_confidence
        ):

            state[
                "plate"
            ] = plate

            state[
                "plate_confidence"
            ] = confidence

        state[
            "plate_status"
        ] = "STABLE"

        state[
            "plate_support_count"
        ] = max(
            int(
                state.get(
                    "plate_support_count",
                    0,
                )
            ),
            incoming_support,
        )

        state[
            "best_plate_confidence"
        ] = max(
            _clamp01(
                state.get(
                    "best_plate_confidence",
                    0.0,
                )
            ),
            confidence,
        )

    elif (
        current_plate is None
        and confidence
        >= 0.50
    ):

        state[
            "plate"
        ] = plate

        state[
            "plate_confidence"
        ] = confidence

        state[
            "plate_status"
        ] = "PROVISIONAL"

        state[
            "plate_support_count"
        ] = incoming_support

    # Conflicting plate observations are preserved.
    if (
        current_plate is not None
        and plate != current_plate
        and confidence >= 0.75
    ):

        conflicts = state.setdefault(
            "plate_conflict_candidates",
            [],
        )

        if plate not in conflicts:

            conflicts.append(
                plate
            )

        if len(
            conflicts
        ) > 10:

            del conflicts[
                :-10
            ]


# ============================================================
# RE-ID STATE
# ============================================================

def _update_reid_state(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> None:

    score = _clamp01(
        observation.get(
            "reid_score",
            0.0,
        )
    )

    best_score = _clamp01(
        observation.get(
            "best_reid_score",
            score,
        )
    )

    consistency = _clamp01(
        observation.get(
            "reid_consistency",
            0.0,
        )
    )

    support = max(
        0,
        int(
            observation.get(
                "supporting_frames",
                0,
            )
        )
    )

    history = state.setdefault(
        "reid_history",
        []
    )

    history.append(
        {
            "timestamp":
                observation[
                    "timestamp"
                ],

            "camera_id":
                observation.get(
                    "camera_id"
                ),

            "track_id":
                observation.get(
                    "track_id"
                ),

            "score":
                score,

            "best_score":
                best_score,

            "consistency":
                consistency,

            "supporting_frames":
                support,

            "observation_quality":
                _clamp01(
                    observation.get(
                        "reid_observation_quality",
                        0.0,
                    )
                ),

            "crop_quality":
                _clamp01(
                    observation.get(
                        "crop_quality",
                        0.0,
                    )
                ),

            "embedding_quality":
                _clamp01(
                    observation.get(
                        "embedding_quality",
                        0.0,
                    )
                ),
        }
    )

    if (
        len(history)
        > MAX_REID_HISTORY
    ):

        del history[
            :-MAX_REID_HISTORY
        ]

    state[
        "best_reid_score"
    ] = max(
        _clamp01(
            state.get(
                "best_reid_score",
                0.0,
            )
        ),
        best_score,
    )

    state[
        "best_reid_observation_quality"
    ] = max(
        _clamp01(
            state.get(
                "best_reid_observation_quality",
                0.0,
            )
        ),
        _clamp01(
            observation.get(
                "reid_observation_quality",
                0.0,
            )
        ),
    )

    state[
        "best_crop_quality"
    ] = max(
        _clamp01(
            state.get(
                "best_crop_quality",
                0.0,
            )
        ),
        _clamp01(
            observation.get(
                "crop_quality",
                0.0,
            )
        ),
    )

    state[
        "best_embedding_quality"
    ] = max(
        _clamp01(
            state.get(
                "best_embedding_quality",
                0.0,
            )
        ),
        _clamp01(
            observation.get(
                "embedding_quality",
                0.0,
            )
        ),
    )


# ============================================================
# CORRELATION STATE
# ============================================================

def _update_correlation_state(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> None:

    timestamp = observation[
        "timestamp"
    ]

    score = _clamp01(
        observation.get(
            "correlation_score",
            0.0,
        )
    )

    decision = str(
        observation.get(
            "correlation_decision",
            "UNCERTAIN",
        )
    ).upper()

    match_type = str(
        observation.get(
            "match_type",
            "unknown",
        )
    )

    history = state.setdefault(
        "correlation_history",
        []
    )

    history.append(
        {
            "timestamp":
                timestamp,

            "decision":
                decision,

            "score":
                score,

            "match_type":
                match_type,

            "candidate_margin":
                observation.get(
                    "candidate_margin",
                    1.0,
                ),

            "supporting_frames":
                observation.get(
                    "supporting_frames",
                    0,
                ),

            "camera_id":
                observation.get(
                    "camera_id"
                ),
        }
    )

    if (
        len(history)
        > MAX_OBSERVATIONS_PER_VEHICLE
    ):

        del history[
            :-MAX_OBSERVATIONS_PER_VEHICLE
        ]

    state[
        "last_correlation_score"
    ] = score

    state[
        "best_correlation_score"
    ] = max(
        _clamp01(
            state.get(
                "best_correlation_score",
                0.0,
            )
        ),
        score,
    )

    confidence_status = str(
        observation.get(
            "confidence_status",
            STATE_UNCERTAIN,
        )
    ).upper()

    if confidence_status == STATE_CONFIRMED:
        state[
            "confirmed_observation_count"
        ] += 1

    elif confidence_status == STATE_PROBABLE:
        state[
            "probable_observation_count"
        ] += 1

    elif confidence_status == STATE_POSSIBLE:
        state[
            "possible_observation_count"
        ] += 1

    elif decision == "UNCERTAIN":
        state[
            "uncertain_observation_count"
        ] += 1

    elif decision == "REJECT":

        state[
            "rejected_observation_count"
        ] += 1

    if bool(
        observation.get(
            "strong_match",
            False,
        )
    ):

        state[
            "strong_match_count"
        ] += 1

    if match_type == "cross_camera":

        state[
            "cross_camera_match_count"
        ] += 1

    elif match_type == "same_camera_track_recovery":

        state[
            "same_camera_recovery_count"
        ] += 1


# ============================================================
# CAMERA HISTORY
# ============================================================

def _update_camera_history(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> None:

    camera_id = observation.get(
        "camera_id"
    )

    if not camera_id:
        return

    timestamp = observation[
        "timestamp"
    ]

    history = state.setdefault(
        "camera_history",
        [],
    )

    last = (
        history[-1]
        if history
        else None
    )

    if (
        last is not None
        and last.get(
            "camera_id"
        )
        == camera_id
    ):

        last[
            "last_seen"
        ] = timestamp

        last[
            "observation_count"
        ] = int(
            last.get(
                "observation_count",
                0,
            )
        ) + 1

        track_id = observation.get(
            "track_id"
        )

        if (
            track_id
            and track_id
            not in last.get(
                "track_ids",
                [],
            )
        ):

            last.setdefault(
                "track_ids",
                [],
            ).append(
                track_id
            )

        if len(
            last.get(
                "track_ids",
                [],
            )
        ) > 50:

            del last[
                "track_ids"
            ][:-50]

        return

    history.append(
        {
            "camera_id":
                camera_id,

            "first_seen":
                timestamp,

            "last_seen":
                timestamp,

            "observation_count":
                1,

            "track_ids":
                [
                    observation.get(
                        "track_id"
                    )
                ]
                if observation.get(
                    "track_id"
                )
                else [],
        }
    )

    if (
        len(history)
        > MAX_CAMERA_HISTORY
    ):

        del history[
            :-MAX_CAMERA_HISTORY
        ]

    state[
        "camera_count"
    ] = len(
        {
            item.get(
                "camera_id"
            )
            for item in history
            if item.get(
                "camera_id"
            )
        }
    )


# ============================================================
# DISTANCE / SPEED
# ============================================================

def _distance_from_centers(
    previous_center: Optional[list[float]],
    current_center: Optional[list[float]],
) -> Optional[float]:

    if (
        previous_center is None
        or current_center is None
    ):
        return None

    try:

        dx = (
            current_center[0]
            - previous_center[0]
        )

        dy = (
            current_center[1]
            - previous_center[1]
        )

    except (
        TypeError,
        IndexError,
    ):

        return None

    distance = math.sqrt(
        dx * dx
        + dy * dy
    )

    if not math.isfinite(
        distance
    ):
        return None

    return distance


def _calculate_speed(
    previous_center: Optional[list[float]],
    current_center: Optional[list[float]],
    previous_timestamp: Optional[float],
    current_timestamp: float,
) -> Optional[float]:

    if (
        previous_center is None
        or current_center is None
        or previous_timestamp is None
    ):
        return None

    elapsed = (
        current_timestamp
        - previous_timestamp
    )

    if elapsed <= 0.0:
        return None

    pixel_distance = (
        _distance_from_centers(
            previous_center,
            current_center,
        )
    )

    if pixel_distance is None:
        return None

    # Pixel speed is intentionally not presented as physical
    # km/h. Physical speed should come from calibrated GIS/camera
    # geometry. This function only estimates normalized motion.
    normalized_speed = (
        pixel_distance
        / elapsed
    )

    if not math.isfinite(
        normalized_speed
    ):
        return None

    return normalized_speed


def _calculate_acceleration(
    previous_speed: Optional[float],
    current_speed: Optional[float],
    elapsed: Optional[float],
) -> Optional[float]:

    if (
        previous_speed is None
        or current_speed is None
        or elapsed is None
        or elapsed <= 0.0
    ):
        return None

    acceleration = (
        current_speed
        - previous_speed
    ) / elapsed

    if not math.isfinite(
        acceleration
    ):
        return None

    return acceleration


# ============================================================
# TRANSITION
# ============================================================

def _record_transition(
    state: dict[str, Any],
    previous_observation: Optional[dict[str, Any]],
    observation: dict[str, Any],
) -> None:

    if previous_observation is None:
        return

    previous_camera = (
        previous_observation.get(
            "camera_id"
        )
    )

    current_camera = (
        observation.get(
            "camera_id"
        )
    )

    previous_timestamp = _safe_timestamp(
        previous_observation.get(
            "timestamp"
        )
    )

    current_timestamp = _safe_timestamp(
        observation.get(
            "timestamp"
        )
    )

    elapsed = (
        current_timestamp
        - previous_timestamp
    )

    if elapsed < 0.0:
        return

    # A very large gap is not a continuous physical transition.
    # Preserve the observation, but don't derive speed/acceleration
    # from a disconnected time interval.
    transition_continuous = (
        elapsed <= MAX_TRANSITION_ELAPSED_SECONDS
    )

    transition = {

        "from_camera_id":
            previous_camera,

        "to_camera_id":
            current_camera,

        "from_timestamp":
            previous_timestamp,

        "to_timestamp":
            current_timestamp,

        "elapsed_seconds":
            elapsed,

        "transition_continuous":
            transition_continuous,

        "match_type":
            observation.get(
                "match_type"
            ),

        "correlation_score":
            observation.get(
                "correlation_score",
                0.0,
            ),

        "gis_score":
            observation.get(
                "gis_score",
                0.0,
            ),

        "gis_feasible":
            observation.get(
                "gis_feasible",
                True,
            ),

        "gis_reason":
            observation.get(
                "gis_reason"
            ),

        "gis_distance_meters":
            observation.get(
                "gis_distance_meters"
            ),

        "gis_implied_speed_kmh":
            observation.get(
                "gis_implied_speed_kmh"
            ),

        "direction_score":
            observation.get(
                "gis_direction_score",
                0.5,
            ),
    }

    transitions = state.setdefault(
        "transitions",
        []
    )

    transitions.append(
        transition
    )

    if (
        len(transitions)
        > MAX_TRANSITIONS_PER_VEHICLE
    ):

        del transitions[
            :-MAX_TRANSITIONS_PER_VEHICLE
        ]

    # Physical distance is only added when GIS provides it.
    distance = observation.get(
        "gis_distance_meters"
    )

    if distance is not None and transition_continuous:

        distance = _safe_float(
            distance,
            0.0,
        )

        if (
            distance >= 0.0
            and math.isfinite(
                distance
            )
        ):

            state[
                "total_distance_meters"
            ] += distance

            if elapsed > 0.0:

                speed_kmh = (
                    distance
                    / elapsed
                    * 3.6
                )

                if (
                    0.0
                    <= speed_kmh
                    <= MAX_SPEED_KMH
                ):

                    # IMPORTANT: capture the previous physical
                    # speed before storing the current speed.
                    previous_speed = _safe_float(
                        state.get(
                            "last_speed_kmh"
                        ),
                        float("nan"),
                    )

                    state[
                        "last_speed_kmh"
                    ] = speed_kmh

                    state[
                        "max_speed_kmh"
                    ] = max(
                        _safe_float(
                            state.get(
                                "max_speed_kmh",
                                0.0,
                            ),
                            0.0,
                        ),
                        speed_kmh,
                    )

                    if not math.isfinite(
                        previous_speed
                    ):
                        previous_speed = None

                    acceleration = (
                        _calculate_acceleration(
                            previous_speed,
                            speed_kmh,
                            elapsed,
                        )
                    )

                    if (
                        acceleration is not None
                        and abs(
                            acceleration
                        )
                        <= MAX_REASONABLE_ACCELERATION
                    ):

                        state[
                            "last_acceleration_kmh_s"
                        ] = acceleration


# ============================================================
# GIS HISTORY
# ============================================================

def _update_gis_history(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> None:

    if not observation.get(
        "gis_available",
        False,
    ):

        # Keep fallback information only when meaningful.
        if not observation.get(
            "gis_reason"
        ):
            return

    history = state.setdefault(
        "gis_history",
        []
    )

    history.append(
        {
            "timestamp":
                observation[
                    "timestamp"
                ],

            "camera_id":
                observation.get(
                    "camera_id"
                ),

            "gis_score":
                observation.get(
                    "gis_score",
                    0.0,
                ),

            "direction_score":
                observation.get(
                    "gis_direction_score",
                    0.5,
                ),

            "distance_meters":
                observation.get(
                    "gis_distance_meters"
                ),

            "elapsed_seconds":
                observation.get(
                    "gis_elapsed_seconds"
                ),

            "implied_speed_kmh":
                observation.get(
                    "gis_implied_speed_kmh"
                ),

            "travel_time_score":
                observation.get(
                    "gis_travel_time_score",
                    0.0,
                ),

            "feasible":
                bool(
                    observation.get(
                        "gis_feasible",
                        True,
                    )
                ),

            "reason":
                observation.get(
                    "gis_reason"
                ),
        }
    )

    if len(
        history
    ) > MAX_OBSERVATIONS_PER_VEHICLE:

        del history[
            :-MAX_OBSERVATIONS_PER_VEHICLE
        ]


# ============================================================
# ACTIVITY EVENTS
# ============================================================

def _detect_activity_events(
    state: dict[str, Any],
    previous_observation: Optional[dict[str, Any]],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:

    events: list[
        dict[str, Any]
    ] = []

    timestamp = observation[
        "timestamp"
    ]

    global_vehicle_id = observation.get(
        "global_vehicle_id"
    )

    camera_id = observation.get(
        "camera_id"
    )

    # --------------------------------------------------------
    # New camera sighting.
    # --------------------------------------------------------

    if (
        previous_observation is not None
        and previous_observation.get(
            "camera_id"
        )
        != camera_id
    ):

        events.append(
            {
                "event_type":
                    "CAMERA_TRANSITION",

                "timestamp":
                    timestamp,

                "camera_id":
                    camera_id,

                "previous_camera_id":
                    previous_observation.get(
                        "camera_id"
                    ),

                "global_vehicle_id":
                    global_vehicle_id,

                "confidence":
                    observation.get(
                        "correlation_score",
                        0.0,
                    ),

                "evidence":
                    {
                        "match_type":
                            observation.get(
                                "match_type"
                            ),

                        "reid_score":
                            observation.get(
                                "reid_score",
                                0.0,
                            ),

                        "plate_score":
                            observation.get(
                                "plate_confidence",
                                0.0,
                            ),

                        "gis_score":
                            observation.get(
                                "gis_score",
                                0.0,
                            ),

                        "gis_feasible":
                            observation.get(
                                "gis_feasible",
                                True,
                            ),
                    },
            }
        )

    # --------------------------------------------------------
    # Strong correlation event.
    # --------------------------------------------------------

    if bool(
        observation.get(
            "strong_match",
            False,
        )
    ):

        events.append(
            {
                "event_type":
                    "STRONG_VEHICLE_CORRELATION",

                "timestamp":
                    timestamp,

                "camera_id":
                    camera_id,

                "global_vehicle_id":
                    global_vehicle_id,

                "confidence":
                    observation.get(
                        "correlation_score",
                        0.0,
                    ),

                "evidence":
                    {
                        "reid_score":
                            observation.get(
                                "reid_score",
                                0.0,
                            ),

                        "best_reid_score":
                            observation.get(
                                "best_reid_score",
                                0.0,
                            ),

                        "supporting_frames":
                            observation.get(
                                "supporting_frames",
                                0,
                            ),

                        "plate_status":
                            observation.get(
                                "plate_status"
                            ),

                        "gis_score":
                            observation.get(
                                "gis_score",
                                0.0,
                            ),
                    },
            }
        )

    # --------------------------------------------------------
    # Impossible GIS transition.
    # --------------------------------------------------------

    if (
        observation.get(
            "gis_available",
            False,
        )
        and not observation.get(
            "gis_feasible",
            True,
        )
    ):

        events.append(
            {
                "event_type":
                    "IMPOSSIBLE_CAMERA_TRANSITION",

                "timestamp":
                    timestamp,

                "camera_id":
                    camera_id,

                "global_vehicle_id":
                    global_vehicle_id,

                "confidence":
                    1.0
                    - observation.get(
                        "gis_score",
                        0.0,
                    ),

                "severity":
                    "HIGH",

                "evidence":
                    {
                        "gis_score":
                            observation.get(
                                "gis_score",
                                0.0,
                            ),

                        "distance_meters":
                            observation.get(
                                "gis_distance_meters"
                            ),

                        "elapsed_seconds":
                            observation.get(
                                "gis_elapsed_seconds"
                            ),

                        "implied_speed_kmh":
                            observation.get(
                                "gis_implied_speed_kmh"
                            ),

                        "reason":
                            observation.get(
                                "gis_reason"
                            ),
                    },
            }
        )

    # --------------------------------------------------------
    # Plate confirmation.
    # --------------------------------------------------------

    if (
        observation.get(
            "plate_status"
        )
        == "STABLE"
        and observation.get(
            "plate_confidence",
            0.0,
        )
        >= PLATE_CONFIRMED_THRESHOLD
        and observation.get(
            "plate_support_count",
            0,
        )
        >= 2
    ):

        previous_plate = (
            previous_observation.get(
                "plate"
            )
            if previous_observation
            else None
        )

        if previous_plate != observation.get(
            "plate"
        ):

            events.append(
                {
                    "event_type":
                        "PLATE_CONFIRMED",

                    "timestamp":
                        timestamp,

                    "camera_id":
                        camera_id,

                    "global_vehicle_id":
                        global_vehicle_id,

                    "confidence":
                        observation.get(
                            "plate_confidence",
                            0.0,
                        ),

                    "evidence":
                        {
                            "plate":
                                observation.get(
                                    "plate"
                                ),

                            "supporting_frames":
                                observation.get(
                                    "plate_support_count",
                                    0,
                                ),
                        },
                }
            )

    # --------------------------------------------------------
    # High-confidence vehicle identity.
    # --------------------------------------------------------

    if (
        observation.get(
            "confidence_status"
        )
        == STATE_CONFIRMED
        and state.get(
            "confidence_status"
        )
        != STATE_CONFIRMED
    ):

        events.append(
            {
                "event_type":
                    "VEHICLE_IDENTITY_CONFIRMED",

                "timestamp":
                    timestamp,

                "camera_id":
                    camera_id,

                "global_vehicle_id":
                    global_vehicle_id,

                "confidence":
                    observation.get(
                        "correlation_score",
                        0.0,
                    ),

                "evidence":
                    {
                        "reid_score":
                            observation.get(
                                "reid_score",
                                0.0,
                            ),

                        "plate_score":
                            observation.get(
                                "plate_confidence",
                                0.0,
                            ),

                        "gis_score":
                            observation.get(
                                "gis_score",
                                0.0,
                            ),

                        "supporting_frames":
                            observation.get(
                                "supporting_frames",
                                0,
                            ),
                    },
            }
        )

    return events


def _append_activity_events(
    state: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:

    if not events:
        return

    activity_events = state.setdefault(
        "activity_events",
        []
    )

    activity_events.extend(
        events
    )

    if (
        len(activity_events)
        > MAX_ACTIVITY_EVENTS_PER_VEHICLE
    ):

        del activity_events[
            :-MAX_ACTIVITY_EVENTS_PER_VEHICLE
        ]


# ============================================================
# STATE RESOLUTION
# ============================================================

def _resolve_runtime_state(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> str:

    timestamp = observation[
        "timestamp"
    ]

    decision = str(
        observation.get(
            "correlation_decision",
            "UNCERTAIN",
        )
    ).upper()

    confidence_status = str(
        state.get(
            "confidence_status",
            STATE_UNCERTAIN,
        )
    ).upper()

    # Hard rejection applies to the observation, not necessarily
    # to the historical identity.
    if decision == "REJECT":

        state[
            "last_rejection"
        ] = {
            "timestamp":
                timestamp,

            "reason":
                observation.get(
                    "gis_reason",
                    "Correlation rejected",
                ),

            "camera_id":
                observation.get(
                    "camera_id"
                ),
        }

        # Keep historical vehicle identity alive but mark current
        # observation as uncertain/inactive.
        return STATE_UNCERTAIN

    if (
        confidence_status
        == STATE_CONFIRMED
    ):

        return STATE_ACTIVE

    if (
        confidence_status
        == STATE_PROBABLE
    ):

        return STATE_ACTIVE

    if (
        confidence_status
        == STATE_POSSIBLE
    ):

        return STATE_ACTIVE

    return STATE_UNCERTAIN


# ============================================================
# APPLY OBSERVATION
# ============================================================

def _apply_observation(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:

    timestamp = observation[
        "timestamp"
    ]

    previous_observation = (
        state[
            "observations"
        ][-1]
        if state.get(
            "observations"
        )
        else None
    )

    # --------------------------------------------------------
    # Temporal ordering.
    # --------------------------------------------------------

    is_out_of_order = bool(
        previous_observation is not None
        and timestamp
        < _safe_timestamp(
            previous_observation.get(
                "timestamp"
            )
        )
    )

    if is_out_of_order:

        # Don't let delayed frames move the live state backward.
        state.setdefault(
            "out_of_order_observations",
            0,
        )

        state[
            "out_of_order_observations"
        ] += 1

    # --------------------------------------------------------
    # State before update.
    # --------------------------------------------------------

    old_runtime_state = str(
        state.get(
            "state",
            STATE_UNCERTAIN,
        )
    )

    # --------------------------------------------------------
    # Core counters.
    # --------------------------------------------------------

    state[
        "observation_count"
    ] += 1

    # --------------------------------------------------------
    # Local track count.
    # --------------------------------------------------------

    track_key = (
        observation.get(
            "camera_id"
        ),
        observation.get(
            "track_id"
        ),
    )

    known_tracks = state.setdefault(
        "_known_tracks",
        set(),
    )

    if track_key not in known_tracks:

        known_tracks.add(
            track_key
        )

        state[
            "local_track_count"
        ] = len(
            known_tracks
        )

    # --------------------------------------------------------
    # Last physical observation.
    # --------------------------------------------------------

    if not is_out_of_order:

        state[
            "last_observation_at"
        ] = max(
            _safe_timestamp(
                state.get(
                    "last_observation_at",
                    timestamp,
                )
            ),
            timestamp,
        )

        state[
            "last_seen"
        ] = max(
            _safe_timestamp(
                state.get(
                    "last_seen",
                    timestamp,
                )
            ),
            timestamp,
        )

        state[
            "last_camera_id"
        ] = observation.get(
            "camera_id"
        )

        state[
            "last_track_id"
        ] = observation.get(
            "track_id"
        )

        if observation.get(
            "heading"
        ) is not None:

            state[
                "last_heading"
            ] = observation.get(
                "heading"
            )

        if observation.get(
            "center"
        ) is not None:

            state[
                "last_center"
            ] = observation.get(
                "center"
            )

        if observation.get(
            "bbox"
        ) is not None:

            state[
                "last_bbox"
            ] = observation.get(
                "bbox"
            )

    # --------------------------------------------------------
    # Vehicle type.
    # --------------------------------------------------------

    vehicle_type = observation.get(
        "vehicle_type"
    )

    if (
        state.get(
            "vehicle_type"
        ) is None
        and vehicle_type
    ):

        state[
            "vehicle_type"
        ] = vehicle_type

    # --------------------------------------------------------
    # Evidence.
    # --------------------------------------------------------

    _update_confidence_state(
        state,
        observation,
    )

    _update_evidence_metrics(
        state,
        observation,
    )

    _update_plate_state(
        state,
        observation,
    )

    _update_reid_state(
        state,
        observation,
    )

    _update_correlation_state(
        state,
        observation,
    )

    if not is_out_of_order:
        _update_camera_history(
            state,
            observation,
        )

        _update_gis_history(
            state,
            observation,
        )

        _record_transition(
            state,
            previous_observation,
            observation,
        )

    # --------------------------------------------------------
    # Observation history.
    # --------------------------------------------------------

    state[
        "observations"
    ].append(
        _copy(
            observation
        )
    )

    if (
        len(
            state[
                "observations"
            ]
        )
        > MAX_OBSERVATIONS_PER_VEHICLE
    ):

        del state[
            "observations"
        ][
            :-MAX_OBSERVATIONS_PER_VEHICLE
        ]

    # --------------------------------------------------------
    # Activity events.
    # --------------------------------------------------------

    events = _detect_activity_events(
        state,
        previous_observation,
        observation,
    )

    _append_activity_events(
        state,
        events,
    )

    # --------------------------------------------------------
    # Runtime state.
    # --------------------------------------------------------

    runtime_state = _resolve_runtime_state(
        state,
        observation,
    )

    _set_state(
        state,
        runtime_state,
        timestamp,
        reason=(
            observation.get(
                "match_type",
                "observation",
            )
        ),
    )

    # --------------------------------------------------------
    # Updated timestamp.
    # --------------------------------------------------------

    if not is_out_of_order:
        state[
            "updated_at"
        ] = timestamp
    else:
        state[
            "updated_at"
        ] = max(
            _safe_timestamp(
                state.get(
                    "updated_at",
                    timestamp,
                )
            ),
            timestamp,
        )

    # --------------------------------------------------------
    # Public copy must not expose internal set.
    # --------------------------------------------------------

    return _public_state(
        state
    )


# ============================================================
# EVIDENCE METRICS
# ============================================================

def _update_evidence_metrics(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> None:

    score = _clamp01(
        observation.get(
            "correlation_score",
            0.0,
        )
    )

    state[
        "best_correlation_score"
    ] = max(
        _clamp01(
            state.get(
                "best_correlation_score",
                0.0,
            )
        ),
        score,
    )

    plate_confidence = _clamp01(
        observation.get(
            "plate_confidence",
            0.0,
        )
    )

    state[
        "best_plate_confidence"
    ] = max(
        _clamp01(
            state.get(
                "best_plate_confidence",
                0.0,
            )
        ),
        plate_confidence,
    )

    gis_score = _clamp01(
        observation.get(
            "gis_score",
            0.0,
        )
    )

    state[
        "best_gis_score"
    ] = max(
        _clamp01(
            state.get(
                "best_gis_score",
                0.0,
            )
        ),
        gis_score,
    )

    evidence_strength = (
        observation.get(
            "evidence_strength",
            "NONE",
        )
    )

    strength_rank = {
        "NONE": 0,
        "WEAK": 1,
        "MODERATE": 2,
        "GOOD": 3,
        "STRONG": 4,
    }

    current_strength = str(
        state.get(
            "evidence_strength",
            "NONE",
        )
    ).upper()

    incoming_strength = str(
        evidence_strength
    ).upper()

    if strength_rank.get(
        incoming_strength,
        0,
    ) >= strength_rank.get(
        current_strength,
        0,
    ):

        state[
            "evidence_strength"
        ] = incoming_strength


# ============================================================
# PUBLIC STATE SANITIZATION
# ============================================================

def _public_state(
    state: dict[str, Any],
) -> dict[str, Any]:

    result = _copy(
        state
    )

    # Internal set must never be serialized.
    result.pop(
        "_known_tracks",
        None,
    )

    return result


# ============================================================
# MAIN PUBLIC API
# ============================================================

def update_vehicle_state(
    vehicle: dict[str, Any],
    correlation: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """
    Main state-management API.

    Expected caller contract:

        vehicle = {
            camera_id,
            track_id,
            timestamp,
            vehicle_type,
            confidence,
            bbox,
            center,
            heading,
            ...
        }

        correlation = correlate_vehicle(vehicle)

    The function:

        1. validates observation
        2. normalizes correlation evidence
        3. creates/retrieves global state
        4. updates ANPR state
        5. updates Re-ID state
        6. updates GIS state
        7. records camera transition
        8. records activity events
        9. maintains confidence lifecycle
       10. returns a defensive state snapshot
    """

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

    global_vehicle_id = _safe_string(
        correlation.get(
            "global_vehicle_id"
        ),
        128,
    )

    # A rejected correlation can legitimately have no global ID.
    # It should not create a new identity.
    decision = str(
        correlation.get(
            "decision",
            "UNCERTAIN",
        )
    ).upper()

    if (
        global_vehicle_id is None
        and decision == "REJECT"
    ):
        return None

    observation = _normalize_observation(
        vehicle,
        correlation,
    )

    if (
        observation.get(
            "global_vehicle_id"
        )
        is None
    ):

        return None

    with _state_lock:

        state = _vehicle_states.get(
            global_vehicle_id
        )

        if state is None:

            state = _create_state(
                observation
            )

            # Internal track set.
            state[
                "_known_tracks"
            ] = set()

            _vehicle_states[
                global_vehicle_id
            ] = state

        result = _apply_observation(
            state,
            observation,
        )

        _enforce_state_limit()

    # --------------------------------------------------------
    # Persistence outside lock.
    # --------------------------------------------------------

    _persist_observation_safe(
        observation,
        result,
    )

    _persist_state_safe(
        result
    )

    return result


# ============================================================
# ALIAS COMPATIBILITY
# ============================================================

def process_vehicle_state(
    vehicle: dict[str, Any],
    correlation: dict[str, Any],
) -> Optional[dict[str, Any]]:

    return update_vehicle_state(
        vehicle,
        correlation,
    )


def updateVehicleState(
    vehicle: dict[str, Any],
    correlation: dict[str, Any],
) -> Optional[dict[str, Any]]:

    return update_vehicle_state(
        vehicle,
        correlation,
    )


# ============================================================
# READ APIs
# ============================================================

def get_vehicle_state(
    global_vehicle_id: str,
) -> Optional[dict[str, Any]]:

    global_vehicle_id = _safe_string(
        global_vehicle_id,
        128,
    )

    if not global_vehicle_id:
        return None

    with _state_lock:

        state = _vehicle_states.get(
            global_vehicle_id
        )

        if state is None:
            return None

        return _public_state(
            state
        )


def get_all_vehicle_states() -> list[
    dict[str, Any]
]:

    with _state_lock:

        return [
            _public_state(
                state
            )
            for state
            in _vehicle_states.values()
        ]


def get_active_vehicle_states() -> list[
    dict[str, Any]
]:

    now = time.time()

    result = []

    with _state_lock:

        for state in (
            _vehicle_states.values()
        ):

            last_seen = _safe_timestamp(
                state.get(
                    "last_seen"
                )
            )

            if (
                now
                - last_seen
                <= ACTIVE_VEHICLE_TIMEOUT_SECONDS
            ):

                result.append(
                    _public_state(
                        state
                    )
                )

    return result


def get_vehicle_observations(
    global_vehicle_id: str,
) -> list[
    dict[str, Any]
]:

    state = get_vehicle_state(
        global_vehicle_id
    )

    if state is None:
        return []

    return _copy(
        state.get(
            "observations",
            [],
        )
    )


def get_vehicle_transitions(
    global_vehicle_id: str,
) -> list[
    dict[str, Any]
]:

    state = get_vehicle_state(
        global_vehicle_id
    )

    if state is None:
        return []

    return _copy(
        state.get(
            "transitions",
            [],
        )
    )


def get_vehicle_activity_events(
    global_vehicle_id: str,
) -> list[
    dict[str, Any]
]:

    state = get_vehicle_state(
        global_vehicle_id
    )

    if state is None:
        return []

    return _copy(
        state.get(
            "activity_events",
            [],
        )
    )


def get_vehicle_camera_history(
    global_vehicle_id: str,
) -> list[
    dict[str, Any]
]:

    state = get_vehicle_state(
        global_vehicle_id
    )

    if state is None:
        return []

    return _copy(
        state.get(
            "camera_history",
            [],
        )
    )


# ============================================================
# VEHICLE SUMMARY
# ============================================================

def get_vehicle_summary(
    global_vehicle_id: str,
) -> Optional[dict[str, Any]]:

    state = get_vehicle_state(
        global_vehicle_id
    )

    if state is None:
        return None

    return {

        "global_vehicle_id":
            state.get(
                "global_vehicle_id"
            ),

        "vehicle_type":
            state.get(
                "vehicle_type"
            ),

        "state":
            state.get(
                "state"
            ),

        "confidence_status":
            state.get(
                "confidence_status"
            ),

        "evidence_strength":
            state.get(
                "evidence_strength"
            ),

        "first_seen":
            state.get(
                "first_seen"
            ),

        "last_seen":
            state.get(
                "last_seen"
            ),

        "last_camera_id":
            state.get(
                "last_camera_id"
            ),

        "camera_count":
            state.get(
                "camera_count",
                0,
            ),

        "observation_count":
            state.get(
                "observation_count",
                0,
            ),

        "cross_camera_match_count":
            state.get(
                "cross_camera_match_count",
                0,
            ),

        "strong_match_count":
            state.get(
                "strong_match_count",
                0,
            ),

        "correlation_score":
            state.get(
                "last_correlation_score",
                0.0,
            ),

        "best_correlation_score":
            state.get(
                "best_correlation_score",
                0.0,
            ),

        "best_reid_score":
            state.get(
                "best_reid_score",
                0.0,
            ),

        "best_reid_observation_quality":
            state.get(
                "best_reid_observation_quality",
                0.0,
            ),

        "best_crop_quality":
            state.get(
                "best_crop_quality",
                0.0,
            ),

        "best_embedding_quality":
            state.get(
                "best_embedding_quality",
                0.0,
            ),

        "plate":
            state.get(
                "plate"
            ),

        "plate_confidence":
            state.get(
                "plate_confidence",
                0.0,
            ),

        "plate_status":
            state.get(
                "plate_status",
                "UNKNOWN",
            ),

        "plate_support_count":
            state.get(
                "plate_support_count",
                0,
            ),

        "total_distance_meters":
            state.get(
                "total_distance_meters",
                0.0,
            ),

        "max_speed_kmh":
            state.get(
                "max_speed_kmh",
                0.0,
            ),

        "activity_event_count":
            len(
                state.get(
                    "activity_events",
                    [],
                )
            ),
    }


# ============================================================
# IDENTITY / CONTRADICTION ANALYSIS
# ============================================================

def get_vehicle_identity_health(
    global_vehicle_id: str,
) -> Optional[dict[str, Any]]:

    state = get_vehicle_state(
        global_vehicle_id
    )

    if state is None:
        return None

    observations = state.get(
        "observations",
        [],
    )

    if not observations:
        return {
            "global_vehicle_id":
                global_vehicle_id,

            "identity_health":
                "UNKNOWN",

            "contradictions":
                [],
        }

    contradictions = []

    # --------------------------------------------------------
    # Plate contradictions.
    # --------------------------------------------------------

    stable_plate = state.get(
        "plate"
    )

    if stable_plate:

        for observation in observations:

            plate = observation.get(
                "plate"
            )

            confidence = _clamp01(
                observation.get(
                    "plate_confidence",
                    0.0,
                )
            )

            if (
                plate
                and plate
                != stable_plate
                and confidence
                >= PLATE_CONFIRMED_THRESHOLD
            ):

                contradictions.append(
                    {
                        "type":
                            "PLATE_CONTRADICTION",

                        "timestamp":
                            observation.get(
                                "timestamp"
                            ),

                        "expected":
                            stable_plate,

                        "observed":
                            plate,

                        "confidence":
                            confidence,
                    }
                )

    # --------------------------------------------------------
    # GIS contradictions.
    # --------------------------------------------------------

    for transition in state.get(
        "transitions",
        [],
    ):

        if not transition.get(
            "gis_feasible",
            True,
        ):

            contradictions.append(
                {
                    "type":
                        "GIS_CONTRADICTION",

                    "timestamp":
                        transition.get(
                            "to_timestamp"
                        ),

                    "from_camera_id":
                        transition.get(
                            "from_camera_id"
                        ),

                    "to_camera_id":
                        transition.get(
                            "to_camera_id"
                        ),

                    "implied_speed_kmh":
                        transition.get(
                            "gis_implied_speed_kmh"
                        ),

                    "reason":
                        transition.get(
                            "gis_reason"
                        ),
                }
            )

    # --------------------------------------------------------
    # Determine health.
    # --------------------------------------------------------

    contradiction_count = len(
        contradictions
    )

    confirmed_count = int(
        state.get(
            "confirmed_observation_count",
            0,
        )
    )

    strong_count = int(
        state.get(
            "strong_match_count",
            0,
        )
    )

    if contradiction_count >= 3:

        health = "CONFLICTED"

    elif contradiction_count > 0:

        health = "REVIEW"

    elif (
        confirmed_count
        >= MIN_OBSERVATIONS_FOR_CONFIRMED_STATE
        and strong_count > 0
    ):

        health = "HEALTHY"

    elif confirmed_count > 0:

        health = "PROBABLE"

    else:

        health = "UNCERTAIN"

    return {

        "global_vehicle_id":
            global_vehicle_id,

        "identity_health":
            health,

        "contradiction_count":
            contradiction_count,

        "contradictions":
            contradictions[
                -20:
            ],

        "confirmed_observation_count":
            confirmed_count,

        "strong_match_count":
            strong_count,

        "best_reid_score":
            state.get(
                "best_reid_score",
                0.0,
            ),

        "best_reid_observation_quality":
            state.get(
                "best_reid_observation_quality",
                0.0,
            ),

        "best_crop_quality":
            state.get(
                "best_crop_quality",
                0.0,
            ),

        "best_embedding_quality":
            state.get(
                "best_embedding_quality",
                0.0,
            ),

        "best_plate_confidence":
            state.get(
                "best_plate_confidence",
                0.0,
            ),

        "best_gis_score":
            state.get(
                "best_gis_score",
                0.0,
            ),
    }


# ============================================================
# CLEANUP
# ============================================================

def mark_inactive_vehicles(
    timeout_seconds: Optional[float] = None,
) -> int:

    if timeout_seconds is None:

        timeout_seconds = (
            ACTIVE_VEHICLE_TIMEOUT_SECONDS
        )

    timeout_seconds = max(
        1.0,
        float(
            timeout_seconds
        ),
    )

    now = time.time()

    changed = 0

    with _state_lock:

        for state in (
            _vehicle_states.values()
        ):

            last_seen = _safe_timestamp(
                state.get(
                    "last_seen"
                )
            )

            if (
                state.get(
                    "state"
                )
                == STATE_ACTIVE
                and now
                - last_seen
                > timeout_seconds
            ):

                _set_state(
                    state,
                    STATE_INACTIVE,
                    now,
                    "vehicle_observation_timeout",
                )

                state[
                    "updated_at"
                ] = now

                changed += 1

    return changed


def cleanup_vehicle_states(
    max_age_seconds: Optional[float] = None,
) -> int:

    if max_age_seconds is None:

        max_age_seconds = (
            VEHICLE_STATE_TTL_SECONDS
        )

    max_age_seconds = max(
        60.0,
        float(
            max_age_seconds
        ),
    )

    now = time.time()

    removed = 0

    with _state_lock:

        stale_ids = []

        for (
            global_id,
            state,
        ) in _vehicle_states.items():

            last_seen = _safe_timestamp(
                state.get(
                    "last_seen",
                    0.0,
                )
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

            state = _vehicle_states.get(
                global_id
            )

            if state is not None:

                _set_state(
                    state,
                    STATE_COMPLETED,
                    now,
                    "vehicle_state_expired",
                )

            _vehicle_states.pop(
                global_id,
                None,
            )

            removed += 1

    return removed


def _enforce_state_limit() -> None:

    if (
        len(_vehicle_states)
        <= MAX_ACTIVE_VEHICLES
    ):
        return

    sortable = sorted(
        _vehicle_states.items(),
        key=lambda item:
            _safe_timestamp(
                item[1].get(
                    "last_seen",
                    0.0,
                )
            ),
    )

    excess = (
        len(_vehicle_states)
        - MAX_ACTIVE_VEHICLES
    )

    for (
        global_id,
        _,
    ) in sortable[:excess]:

        _vehicle_states.pop(
            global_id,
            None,
        )


# ============================================================
# RESET
# ============================================================

def clear_vehicle_state(
    global_vehicle_id: str,
) -> bool:

    global_vehicle_id = _safe_string(
        global_vehicle_id,
        128,
    )

    if not global_vehicle_id:
        return False

    with _state_lock:

        return (
            _vehicle_states.pop(
                global_vehicle_id,
                None,
            )
            is not None
        )


def clear_all_vehicle_states() -> None:

    with _state_lock:

        _vehicle_states.clear()


# ============================================================
# PERSISTENCE
# ============================================================

def _persist_observation_safe(
    observation: dict[str, Any],
    state: dict[str, Any],
) -> None:

    if save_vehicle_observation is None:
        return

    try:

        save_vehicle_observation(
            global_vehicle_id=observation.get(
                "global_vehicle_id"
            ),

            camera_id=observation.get(
                "camera_id"
            ),

            track_id=observation.get(
                "track_id"
            ),

            observed_at=observation.get(
                "timestamp"
            ),

            vehicle_type=observation.get(
                "vehicle_type"
            ),

            confidence=observation.get(
                "confidence",
                0.0,
            ),

            plate=observation.get(
                "plate"
            ),

            plate_confidence=observation.get(
                "plate_confidence",
                0.0,
            ),

            reid_score=observation.get(
                "reid_score",
                0.0,
            ),

            correlation_score=observation.get(
                "correlation_score",
                0.0,
            ),

            correlation_decision=observation.get(
                "correlation_decision"
            ),

            evidence_strength=observation.get(
                "evidence_strength"
            ),

            metadata={
                "state_version":
                    STATE_VERSION,

                "match_type":
                    observation.get(
                        "match_type"
                    ),

                "supporting_frames":
                    observation.get(
                        "supporting_frames",
                        0,
                    ),

                "reid_consistency":
                    observation.get(
                        "reid_consistency",
                        0.0,
                    ),

                "reid_observation_quality":
                    observation.get(
                        "reid_observation_quality",
                        0.0,
                    ),

                "crop_quality":
                    observation.get(
                        "crop_quality",
                        0.0,
                    ),

                "embedding_quality":
                    observation.get(
                        "embedding_quality",
                        0.0,
                    ),

                "temporal_score":
                    observation.get(
                        "temporal_score",
                        0.0,
                    ),

                "gis_score":
                    observation.get(
                        "gis_score",
                        0.0,
                    ),

                "gis_feasible":
                    observation.get(
                        "gis_feasible",
                        True,
                    ),

                "gis_reason":
                    observation.get(
                        "gis_reason"
                    ),
            },
        )

    except TypeError:

        # Compatibility with an older persistence signature.
        try:

            save_vehicle_observation(
                observation
            )

        except Exception:

            logger.exception(
                "Vehicle observation persistence failed"
            )

    except Exception:

        logger.exception(
            "Vehicle observation persistence failed"
        )


def _persist_state_safe(
    state: dict[str, Any],
) -> None:

    if save_vehicle_state is None:
        return

    try:

        save_vehicle_state(
            global_vehicle_id=state.get(
                "global_vehicle_id"
            ),

            state=state.get(
                "state"
            ),

            confidence_status=state.get(
                "confidence_status"
            ),

            vehicle_type=state.get(
                "vehicle_type"
            ),

            first_seen=state.get(
                "first_seen"
            ),

            last_seen=state.get(
                "last_seen"
            ),

            last_camera_id=state.get(
                "last_camera_id"
            ),

            observation_count=state.get(
                "observation_count",
                0,
            ),

            correlation_score=state.get(
                "last_correlation_score",
                0.0,
            ),

            metadata={
                "state_version":
                    STATE_VERSION,

                "evidence_strength":
                    state.get(
                        "evidence_strength"
                    ),

                "best_correlation_score":
                    state.get(
                        "best_correlation_score",
                        0.0,
                    ),

                "best_reid_score":
                    state.get(
                        "best_reid_score",
                        0.0,
                    ),

                "best_plate_confidence":
                    state.get(
                        "best_plate_confidence",
                        0.0,
                    ),

                "best_gis_score":
                    state.get(
                        "best_gis_score",
                        0.0,
                    ),

                "camera_count":
                    state.get(
                        "camera_count",
                        0,
                    ),

                "cross_camera_match_count":
                    state.get(
                        "cross_camera_match_count",
                        0,
                    ),

                "strong_match_count":
                    state.get(
                        "strong_match_count",
                        0,
                    ),
            },
        )

    except TypeError:

        try:

            save_vehicle_state(
                state
            )

        except Exception:

            logger.exception(
                "Vehicle state persistence failed"
            )

    except Exception:

        logger.exception(
            "Vehicle state persistence failed"
        )


# ============================================================
# HEALTH
# ============================================================

def vehicle_state_health() -> dict[str, Any]:

    now = time.time()

    active = 0
    inactive = 0
    confirmed = 0
    probable = 0
    possible = 0
    uncertain = 0
    rejected = 0

    with _state_lock:

        total = len(
            _vehicle_states
        )

        for state in (
            _vehicle_states.values()
        ):

            runtime_state = str(
                state.get(
                    "state",
                    STATE_UNCERTAIN,
                )
            )

            confidence = str(
                state.get(
                    "confidence_status",
                    STATE_UNCERTAIN,
                )
            )

            last_seen = _safe_timestamp(
                state.get(
                    "last_seen",
                    0.0,
                )
            )

            if (
                now
                - last_seen
                <= ACTIVE_VEHICLE_TIMEOUT_SECONDS
            ):
                active += 1

            else:
                inactive += 1

            if confidence == STATE_CONFIRMED:
                confirmed += 1
            elif confidence == STATE_PROBABLE:
                probable += 1
            elif confidence == STATE_POSSIBLE:
                possible += 1
            elif confidence == STATE_REJECTED:
                rejected += 1
            else:
                uncertain += 1

    return {

        "engine":
            "VehicleState",

        "version":
            STATE_VERSION,

        "total_vehicle_states":
            total,

        "active":
            active,

        "inactive":
            inactive,

        "confirmed":
            confirmed,

        "probable":
            probable,

        "possible":
            possible,

        "uncertain":
            uncertain,

        "rejected":
            rejected,

        "persistence_available":
            save_vehicle_observation
            is not None
            and
            save_vehicle_state
            is not None,
    }


# ============================================================
# BACKGROUND MAINTENANCE
# ============================================================

def maintenance_tick() -> dict[str, int]:

    marked_inactive = (
        mark_inactive_vehicles()
    )

    removed = (
        cleanup_vehicle_states()
    )

    return {

        "marked_inactive":
            marked_inactive,

        "removed":
            removed,
    }


# ============================================================
# OPTIONAL LEGACY ACCESSORS
# ============================================================

def getVehicleState(
    global_vehicle_id: str,
):
    return get_vehicle_state(
        global_vehicle_id
    )


def getVehicleJourney(
    global_vehicle_id: str,
):
    state = get_vehicle_state(
        global_vehicle_id
    )

    if state is None:
        return []

    return _copy(
        state.get(
            "camera_history",
            [],
        )
    )


def getVehicleSummary(
    global_vehicle_id: str,
):
    return get_vehicle_summary(
        global_vehicle_id
    )


# ============================================================
# MODULE HEALTH
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    logger.info(
        "VehicleState %s loaded",
        STATE_VERSION,
    )