from __future__ import annotations

import math
import threading
import os
import json
import time
from typing import Any, Optional

import supervision as sv


# ============================================================
# PRODUCTION TRACKER CONFIGURATION
# ============================================================

DEFAULT_TRACKER_FPS = 25.0

TRACK_ACTIVATION_THRESHOLD = 0.65
MINIMUM_MATCHING_THRESHOLD = 0.85

MIN_LOST_TRACK_BUFFER = 60
MAX_LOST_TRACK_BUFFER = 900

MAX_CAMERA_ID_LENGTH = 256
MAX_TRACK_ID_LENGTH = 64

MAX_TRACK_HISTORY_PER_CAMERA = 5000
TRACK_HISTORY_TTL_SECONDS = 120.0

TRACK_QUALITY_DETECTION_WEIGHT = 0.45
TRACK_QUALITY_STABILITY_WEIGHT = 0.35
TRACK_QUALITY_AGE_WEIGHT = 0.20

NEW_TRACK_QUALITY_FLOOR = 0.55
MATURE_TRACK_FRAMES = 3

MAX_CENTER_JUMP_FACTOR = 2.5
MIN_BBOX_IOU_FOR_STABILITY = 0.05


# ============================================================
# LOGGING
# ============================================================

LOG_PREFIX = "[INTEL-I][TRACKER]"

ENABLE_TRACKER_LOGS = True


def _log(message: str) -> None:
    """
    Central tracker logging function.

    Using print() intentionally so logs are immediately visible
    in Docker/systemd/terminal deployments.
    """
    if not ENABLE_TRACKER_LOGS:
        return

    try:
        print(
            f"{LOG_PREFIX} {message}",
            flush=True,
        )
    except Exception:
        pass


def _log_error(message: str) -> None:
    try:
        print(
            f"{LOG_PREFIX}[ERROR] {message}",
            flush=True,
        )
    except Exception:
        pass


def _log_warning(message: str) -> None:
    try:
        print(
            f"{LOG_PREFIX}[WARNING] {message}",
            flush=True,
        )
    except Exception:
        pass


# ============================================================
# THREAD-SAFE TRACKER REGISTRIES
# ============================================================

trackers: dict[str, sv.ByteTrack] = {}
vehicle_trackers: dict[str, sv.ByteTrack] = {}

_tracker_lock = threading.RLock()
_vehicle_tracker_lock = threading.RLock()

_vehicle_track_history: dict[
    str,
    dict[int, dict[str, Any]],
] = {}

_vehicle_track_history_lock = threading.RLock()


# ============================================================
# VEHICLE CLASS MAPPING
# ============================================================

def _load_vehicle_class_names():
    # Default is the bundled COCO YOLO mapping. For a custom vehicle-only
    # detector, configure VEHICLE_CLASS_NAMES_JSON as {"0":"car","1":"truck"}.
    raw = os.getenv("VEHICLE_CLASS_NAMES_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                result = {}
                for key, value in parsed.items():
                    result[int(key)] = str(value)[:50]
                if result:
                    return result
        except Exception:
            _log_warning("Invalid VEHICLE_CLASS_NAMES_JSON; using COCO defaults")
    return {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }


VEHICLE_CLASS_NAMES = _load_vehicle_class_names()


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_camera_id(camID: str) -> Optional[str]:
    if camID is None:
        return None

    try:
        value = str(camID).strip()
    except Exception:
        return None

    if not value or len(value) > MAX_CAMERA_ID_LENGTH:
        return None

    if any(ord(ch) < 32 for ch in value):
        return None

    return value


def _safe_fps(fps: Any) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError):
        return DEFAULT_TRACKER_FPS

    if not math.isfinite(value):
        return DEFAULT_TRACKER_FPS

    return max(1.0, min(120.0, value))


def _tracker_lost_buffer(fps: float) -> int:
    value = int(round(fps * 2.0))

    return max(
        MIN_LOST_TRACK_BUFFER,
        min(MAX_LOST_TRACK_BUFFER, value),
    )


def _safe_class_id(value: Any) -> Optional[int]:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None

    return result


def _safe_track_id(value: Any) -> Optional[int]:
    result = _safe_class_id(value)

    if result is None or result < 0:
        return None

    if len(str(result)) > MAX_TRACK_ID_LENGTH:
        return None

    return result


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default

    if not math.isfinite(result):
        return default

    return result


def _safe_timestamp(
    timestamp: Any,
) -> Optional[float]:

    try:
        result = float(timestamp)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if not math.isfinite(result) or result <= 0:
        return None

    return result


def _safe_bbox(
    bbox: Any,
) -> Optional[list[float]]:

    if bbox is None:
        return None

    try:
        values = list(bbox)

        if len(values) != 4:
            return None

        x1 = _safe_float(values[0])
        y1 = _safe_float(values[1])
        x2 = _safe_float(values[2])
        y2 = _safe_float(values[3])

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not all(
        math.isfinite(value)
        for value in (
            x1,
            y1,
            x2,
            y2,
        )
    ):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def _clamp01(value: Any) -> float:
    return max(
        0.0,
        min(
            1.0,
            _safe_float(
                value,
                0.0,
            ),
        ),
    )


def _bbox_iou(
    a: Optional[list[float]],
    b: Optional[list[float]],
) -> float:

    if not a or not b:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return _clamp01(
        intersection / union
    )


def _bbox_center(
    bbox: list[float],
) -> tuple[float, float]:

    x1, y1, x2, y2 = bbox

    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
    )


def _bbox_diagonal(
    bbox: list[float],
) -> float:

    x1, y1, x2, y2 = bbox

    return max(
        1.0,
        math.hypot(
            x2 - x1,
            y2 - y1,
        ),
    )


# ============================================================
# TRACK HISTORY / QUALITY
# ============================================================

def _cleanup_vehicle_track_history(
    camera_id: str,
    now: float,
) -> None:

    camera_history = _vehicle_track_history.get(
        camera_id
    )

    if camera_history is None:
        return

    expired = [
        track_id
        for track_id, state
        in camera_history.items()
        if now
        - _safe_float(
            state.get(
                "last_seen",
                0.0,
            ),
            0.0,
        )
        > TRACK_HISTORY_TTL_SECONDS
    ]

    for track_id in expired:

        camera_history.pop(
            track_id,
            None,
        )

        _log(
            f"TRACK_EXPIRED "
            f"camera={camera_id} "
            f"track={track_id}"
        )

    if (
        len(camera_history)
        <= MAX_TRACK_HISTORY_PER_CAMERA
    ):
        return

    ordered = sorted(
        camera_history.items(),
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
        len(camera_history)
        - MAX_TRACK_HISTORY_PER_CAMERA
    )

    for track_id, _ in ordered[:excess]:

        camera_history.pop(
            track_id,
            None,
        )

        _log_warning(
            f"TRACK_HISTORY_LIMIT "
            f"camera={camera_id} "
            f"removed_track={track_id}"
        )


def _update_vehicle_track_quality(
    camera_id: str,
    track_id: Optional[int],
    bbox: list[float],
    detection_confidence: float,
    timestamp: float,
) -> dict[str, Any]:

    """
    Build an application-level track quality signal.

    This is NOT ByteTrack's internal confidence.

    Components:

        detection confidence
        bbox temporal stability
        track maturity
    """

    if track_id is None:

        _log_warning(
            f"UNTRACKED_DETECTION "
            f"camera={camera_id} "
            f"detection_conf={detection_confidence:.3f}"
        )

        return {
            "track_quality": 0.0,
            "tracking_confidence": 0.0,
            "track_age_frames": 0,
            "track_stability": 0.0,
            "track_status": "UNTRACKED",
            "track_iou": 0.0,
        }

    with _vehicle_track_history_lock:

        camera_history = (
            _vehicle_track_history.setdefault(
                camera_id,
                {},
            )
        )

        state = camera_history.get(
            track_id
        )

        is_new_track = state is None

        if state is None:

            state = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "frames": 0,
                "last_bbox": bbox,
                "mean_detection_confidence": 0.0,
                "stability_ema": 0.0,
            }

            camera_history[
                track_id
            ] = state

            _log(
                f"NEW_TRACK "
                f"camera={camera_id} "
                f"track={track_id}"
            )

        previous_bbox = state.get(
            "last_bbox"
        )

        previous_timestamp = _safe_float(
            state.get(
                "last_seen",
                timestamp,
            ),
            timestamp,
        )

        frames = (
            int(
                state.get(
                    "frames",
                    0,
                )
            )
            + 1
        )

        # ----------------------------------------------------
        # BBOX IOU
        # ----------------------------------------------------

        iou = _bbox_iou(
            previous_bbox,
            bbox,
        )

        # ----------------------------------------------------
        # CENTER MOVEMENT
        # ----------------------------------------------------

        if previous_bbox:

            old_center = _bbox_center(
                previous_bbox
            )

            new_center = _bbox_center(
                bbox
            )

            jump = math.hypot(
                new_center[0]
                - old_center[0],
                new_center[1]
                - old_center[1],
            )

            reference_size = max(
                _bbox_diagonal(
                    previous_bbox
                ),
                _bbox_diagonal(
                    bbox
                ),
            )

            jump_ratio = (
                jump / reference_size
            )

            motion_stability = (
                1.0
                if jump_ratio <= 1.0
                else max(
                    0.0,
                    1.0
                    - (
                        (
                            jump_ratio
                            - 1.0
                        )
                        / MAX_CENTER_JUMP_FACTOR
                    ),
                )
            )

        else:

            jump_ratio = 0.0
            motion_stability = 0.0

        # ----------------------------------------------------
        # FRAME STABILITY
        # ----------------------------------------------------

        if previous_bbox is None:

            frame_stability = 0.0

        elif (
            iou
            >= MIN_BBOX_IOU_FOR_STABILITY
        ):

            frame_stability = (
                0.70 * iou
                + 0.30
                * motion_stability
            )

        else:

            frame_stability = (
                0.25
                * motion_stability
            )

        old_stability = _clamp01(
            state.get(
                "stability_ema",
                0.0,
            )
        )

        stability_ema = (
            0.70 * old_stability
            + 0.30
            * _clamp01(
                frame_stability
            )
        )

        # ----------------------------------------------------
        # DETECTION CONFIDENCE EMA
        # ----------------------------------------------------

        old_mean_conf = _clamp01(
            state.get(
                "mean_detection_confidence",
                detection_confidence,
            )
        )

        mean_confidence = (
            0.75
            * old_mean_conf
            + 0.25
            * _clamp01(
                detection_confidence
            )
        )

        # ----------------------------------------------------
        # TRACK MATURITY
        # ----------------------------------------------------

        maturity = min(
            1.0,
            frames
            / float(
                MATURE_TRACK_FRAMES
            ),
        )

        # ----------------------------------------------------
        # FINAL TRACK QUALITY
        # ----------------------------------------------------

        quality = (
            TRACK_QUALITY_DETECTION_WEIGHT
            * mean_confidence
            + TRACK_QUALITY_STABILITY_WEIGHT
            * stability_ema
            + TRACK_QUALITY_AGE_WEIGHT
            * maturity
        )

        # First frame cannot be highly trusted.
        if frames == 1:

            quality = min(
                quality,
                NEW_TRACK_QUALITY_FLOOR,
            )

        # ----------------------------------------------------
        # FRAME GAP
        # ----------------------------------------------------

        if (
            previous_timestamp > 0
            and timestamp
            >= previous_timestamp
        ):

            gap = (
                timestamp
                - previous_timestamp
            )

        else:

            gap = 0.0

        # ----------------------------------------------------
        # UPDATE STATE
        # ----------------------------------------------------

        state.update(
            {
                "first_seen": state.get(
                    "first_seen",
                    timestamp,
                ),
                "last_seen": timestamp,
                "frames": frames,
                "last_bbox": list(
                    bbox
                ),
                "mean_detection_confidence":
                    mean_confidence,
                "stability_ema":
                    stability_ema,
                "last_iou":
                    iou,
                "last_gap_seconds":
                    gap,
                "last_jump_ratio":
                    jump_ratio,
            }
        )

        _cleanup_vehicle_track_history(
            camera_id,
            timestamp,
        )

        status = (
            "MATURE"
            if frames
            >= MATURE_TRACK_FRAMES
            else "WARMING"
        )

        # ----------------------------------------------------
        # IMPORTANT TRACK LOG
        # ----------------------------------------------------

        _log(
            f"TRACK_UPDATE "
            f"camera={camera_id} "
            f"track={track_id} "
            f"frame_age={frames} "
            f"status={status} "
            f"det_conf={detection_confidence:.3f} "
            f"mean_conf={mean_confidence:.3f} "
            f"iou={iou:.3f} "
            f"stability={stability_ema:.3f} "
            f"maturity={maturity:.3f} "
            f"quality={quality:.3f}"
        )

        if is_new_track:

            _log(
                f"TRACK_STARTED "
                f"camera={camera_id} "
                f"track={track_id} "
                f"quality={quality:.3f}"
            )

        elif (
            frames
            == MATURE_TRACK_FRAMES
        ):

            _log(
                f"TRACK_MATURE "
                f"camera={camera_id} "
                f"track={track_id} "
                f"frames={frames} "
                f"quality={quality:.3f}"
            )

        if (
            iou
            < MIN_BBOX_IOU_FOR_STABILITY
            and frames > 1
        ):

            _log_warning(
                f"LOW_TRACK_IOU "
                f"camera={camera_id} "
                f"track={track_id} "
                f"iou={iou:.3f} "
                f"jump_ratio={jump_ratio:.3f}"
            )

        if quality < 0.50:

            _log_warning(
                f"LOW_TRACK_QUALITY "
                f"camera={camera_id} "
                f"track={track_id} "
                f"quality={quality:.3f}"
            )

        return {
            "track_quality": _clamp01(
                quality
            ),
            "tracking_confidence": _clamp01(
                quality
            ),
            "track_age_frames": frames,
            "track_stability": _clamp01(
                stability_ema
            ),
            "track_iou": _clamp01(
                iou
            ),
            "track_status": status,
        }


# ============================================================
# PERSON TRACKER
# ============================================================

def getTracker(
    camID: str,
    fps: int = 25,
) -> sv.ByteTrack:

    safe_cam_id = _safe_camera_id(
        camID
    )

    if safe_cam_id is None:
        raise ValueError(
            "Invalid camera ID"
        )

    safe_fps = _safe_fps(
        fps
    )

    with _tracker_lock:

        tracker = trackers.get(
            safe_cam_id
        )

        if tracker is not None:

            _log(
                f"PERSON_TRACKER_REUSE "
                f"camera={safe_cam_id}"
            )

            return tracker

        lost_buffer = (
            _tracker_lost_buffer(
                safe_fps
            )
        )

        _log(
            f"PERSON_TRACKER_CREATE "
            f"camera={safe_cam_id} "
            f"fps={safe_fps:.1f} "
            f"lost_buffer={lost_buffer}"
        )

        tracker = sv.ByteTrack(
            track_activation_threshold=
                TRACK_ACTIVATION_THRESHOLD,

            minimum_matching_threshold=
                MINIMUM_MATCHING_THRESHOLD,

            lost_track_buffer=
                lost_buffer,

            frame_rate=
                safe_fps,
        )

        trackers[
            safe_cam_id
        ] = tracker

        return tracker


def trackDetections(
    camID: str,
    detections: sv.Detections,
    fps: int = 25,
):

    if detections is None:

        _log_warning(
            f"PERSON_TRACK_SKIP "
            f"camera={camID} "
            f"reason=no_detections"
        )

        return detections

    safe_cam_id = _safe_camera_id(
        camID
    )

    if safe_cam_id is None:

        _log_error(
            f"PERSON_TRACK_INVALID_CAMERA "
            f"camera={camID}"
        )

        return detections

    try:

        detection_count = len(
            detections
        )

    except Exception as exc:

        _log_error(
            f"PERSON_TRACK_COUNT_ERROR "
            f"camera={safe_cam_id} "
            f"error={exc}"
        )

        return detections

    if detection_count == 0:

        return detections

    try:

        tracker = getTracker(
            safe_cam_id,
            fps,
        )

        with _tracker_lock:

            tracked = (
                tracker
                .update_with_detections(
                    detections
                )
            )

        try:
            tracked_count = len(
                tracked
            )
        except Exception:
            tracked_count = 0

        _log(
            f"PERSON_TRACK_RESULT "
            f"camera={safe_cam_id} "
            f"input={detection_count} "
            f"output={tracked_count}"
        )

        return tracked

    except Exception as exc:

        _log_error(
            f"PERSON_TRACK_ERROR "
            f"camera={safe_cam_id} "
            f"error={exc}"
        )

        return detections


def clear_tracker(
    camID: str,
):
    safe_cam_id = _safe_camera_id(
        camID
    )

    if safe_cam_id is None:
        return

    with _tracker_lock:

        existed = (
            safe_cam_id
            in trackers
        )

        trackers.pop(
            safe_cam_id,
            None,
        )

    _log(
        f"PERSON_TRACKER_CLEAR "
        f"camera={safe_cam_id} "
        f"existed={existed}"
    )


# ============================================================
# VEHICLE TRACKER
# ============================================================

def getVehicleTracker(
    camID: str,
    fps: int = 25,
) -> sv.ByteTrack:

    safe_cam_id = _safe_camera_id(
        camID
    )

    if safe_cam_id is None:

        raise ValueError(
            "Invalid camera ID"
        )

    safe_fps = _safe_fps(
        fps
    )

    with _vehicle_tracker_lock:

        tracker = vehicle_trackers.get(
            safe_cam_id
        )

        if tracker is not None:

            _log(
                f"VEHICLE_TRACKER_REUSE "
                f"camera={safe_cam_id}"
            )

            return tracker

        lost_buffer = (
            _tracker_lost_buffer(
                safe_fps
            )
        )

        _log(
            f"VEHICLE_TRACKER_CREATE "
            f"camera={safe_cam_id} "
            f"fps={safe_fps:.1f} "
            f"lost_buffer={lost_buffer} "
            f"activation_threshold="
            f"{TRACK_ACTIVATION_THRESHOLD} "
            f"matching_threshold="
            f"{MINIMUM_MATCHING_THRESHOLD}"
        )

        tracker = sv.ByteTrack(
            track_activation_threshold=
                TRACK_ACTIVATION_THRESHOLD,

            minimum_matching_threshold=
                MINIMUM_MATCHING_THRESHOLD,

            lost_track_buffer=
                lost_buffer,

            frame_rate=
                safe_fps,
        )

        vehicle_trackers[
            safe_cam_id
        ] = tracker

        return tracker


def trackVehicleDetections(
    camID: str,
    detections: sv.Detections,
    fps: int = 25,
):

    if detections is None:

        _log_warning(
            f"VEHICLE_TRACK_SKIP "
            f"camera={camID} "
            f"reason=no_detections"
        )

        return detections

    safe_cam_id = _safe_camera_id(
        camID
    )

    if safe_cam_id is None:

        _log_error(
            f"VEHICLE_TRACK_INVALID_CAMERA "
            f"camera={camID}"
        )

        return detections

    try:

        detection_count = len(
            detections
        )

    except Exception as exc:

        _log_error(
            f"VEHICLE_TRACK_COUNT_ERROR "
            f"camera={safe_cam_id} "
            f"error={exc}"
        )

        return detections

    if detection_count <= 0:

        return detections

    _log(
        f"VEHICLE_TRACK_INPUT "
        f"camera={safe_cam_id} "
        f"detections={detection_count}"
    )

    try:

        tracker = getVehicleTracker(
            safe_cam_id,
            fps,
        )

        with _vehicle_tracker_lock:

            tracked = (
                tracker
                .update_with_detections(
                    detections
                )
            )

        try:
            tracked_count = len(
                tracked
            )
        except Exception:
            tracked_count = 0

        tracker_ids = []

        if (
            tracked.tracker_id
            is not None
        ):

            try:

                tracker_ids = [
                    _safe_track_id(
                        value
                    )
                    for value
                    in tracked.tracker_id
                ]

            except Exception:
                tracker_ids = []

        valid_tracker_ids = [
            value
            for value in tracker_ids
            if value is not None
        ]

        _log(
            f"VEHICLE_TRACK_RESULT "
            f"camera={safe_cam_id} "
            f"input={detection_count} "
            f"output={tracked_count} "
            f"tracked_ids={valid_tracker_ids}"
        )

        return tracked

    except Exception as exc:

        _log_error(
            f"VEHICLE_TRACK_ERROR "
            f"camera={safe_cam_id} "
            f"error={exc}"
        )

        return detections


def clear_vehicle_tracker(
    camID: str,
):

    safe_cam_id = _safe_camera_id(
        camID
    )

    if safe_cam_id is None:
        return

    with _vehicle_tracker_lock:

        existed = (
            safe_cam_id
            in vehicle_trackers
        )

        vehicle_trackers.pop(
            safe_cam_id,
            None,
        )

    with _vehicle_track_history_lock:

        history_count = len(
            _vehicle_track_history.get(
                safe_cam_id,
                {},
            )
        )

        _vehicle_track_history.pop(
            safe_cam_id,
            None,
        )

    _log(
        f"VEHICLE_TRACKER_CLEAR "
        f"camera={safe_cam_id} "
        f"tracker_existed={existed} "
        f"history_removed={history_count}"
    )


def clear_all_vehicle_trackers():

    with _vehicle_tracker_lock:

        tracker_count = len(
            vehicle_trackers
        )

        vehicle_trackers.clear()

    with _vehicle_track_history_lock:

        history_count = sum(
            len(history)
            for history
            in _vehicle_track_history.values()
        )

        _vehicle_track_history.clear()

    _log(
        f"ALL_VEHICLE_TRACKERS_CLEAR "
        f"trackers={tracker_count} "
        f"history_tracks={history_count}"
    )


def clear_all_trackers():

    with _tracker_lock:

        person_count = len(
            trackers
        )

        trackers.clear()

    with _vehicle_tracker_lock:

        vehicle_count = len(
            vehicle_trackers
        )

        vehicle_trackers.clear()

    with _vehicle_track_history_lock:

        history_count = sum(
            len(history)
            for history
            in _vehicle_track_history.values()
        )

        _vehicle_track_history.clear()

    _log(
        f"ALL_TRACKERS_CLEAR "
        f"person_trackers={person_count} "
        f"vehicle_trackers={vehicle_count} "
        f"vehicle_history={history_count}"
    )


def reset_camera_trackers(
    camID: str,
):

    safe_cam_id = _safe_camera_id(
        camID
    )

    if safe_cam_id is None:
        return

    _log(
        f"CAMERA_TRACKER_RESET "
        f"camera={safe_cam_id}"
    )

    clear_tracker(
        safe_cam_id
    )

    clear_vehicle_tracker(
        safe_cam_id
    )


# ============================================================
# VEHICLE TRACK EXTRACTION
# ============================================================

def extractVehicleTracks(
    camID: str,
    detections: sv.Detections,
    timestamp: float,
):

    """
    Convert tracked vehicle detections
    into INTEL-I vehicle records.
    """

    vehicles: list[
        dict[str, Any]
    ] = []

    safe_cam_id = _safe_camera_id(
        camID
    )

    safe_timestamp = _safe_timestamp(
        timestamp
    )

    if (
        safe_cam_id is None
        or safe_timestamp is None
    ):

        _log_error(
            f"VEHICLE_EXTRACTION_INVALID_INPUT "
            f"camera={camID} "
            f"timestamp={timestamp}"
        )

        return vehicles

    if detections is None:

        return vehicles

    try:

        detection_count = len(
            detections
        )

    except Exception as exc:

        _log_error(
            f"VEHICLE_EXTRACTION_COUNT_ERROR "
            f"camera={safe_cam_id} "
            f"error={exc}"
        )

        return vehicles

    if (
        detection_count <= 0
        or detections.class_id
        is None
    ):

        return vehicles

    _log(
        f"VEHICLE_EXTRACTION_START "
        f"camera={safe_cam_id} "
        f"detections={detection_count}"
    )

    for index in range(
        detection_count
    ):

        try:

            class_id = _safe_class_id(
                detections.class_id[
                    index
                ]
            )

        except (
            IndexError,
            TypeError,
            ValueError,
        ):

            continue

        if (
            class_id
            not in VEHICLE_CLASS_NAMES
        ):

            continue

        try:

            raw_bbox = (
                detections.xyxy[
                    index
                ]
            )

        except (
            IndexError,
            TypeError,
        ):

            _log_warning(
                f"VEHICLE_BBOX_MISSING "
                f"camera={safe_cam_id} "
                f"index={index}"
            )

            continue

        bbox = _safe_bbox(
            raw_bbox
        )

        if bbox is None:

            _log_warning(
                f"VEHICLE_BBOX_INVALID "
                f"camera={safe_cam_id} "
                f"index={index}"
            )

            continue

        # ----------------------------------------------------
        # DETECTION CONFIDENCE
        # ----------------------------------------------------

        confidence = 0.0

        if (
            detections.confidence
            is not None
        ):

            try:

                confidence = _clamp01(
                    detections.confidence[
                        index
                    ]
                )

            except (
                IndexError,
                TypeError,
                ValueError,
            ):

                confidence = 0.0

        # ----------------------------------------------------
        # TRACK ID
        # ----------------------------------------------------

        track_id = None

        if (
            detections.tracker_id
            is not None
        ):

            try:

                track_id = _safe_track_id(
                    detections.tracker_id[
                        index
                    ]
                )

            except (
                IndexError,
                TypeError,
                ValueError,
            ):

                track_id = None

        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        x1, y1, x2, y2 = bbox

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        vehicle_type = (
            VEHICLE_CLASS_NAMES[
                class_id
            ]
        )

        # ----------------------------------------------------
        # TRACK QUALITY
        # ----------------------------------------------------

        quality = (
            _update_vehicle_track_quality(
                camera_id=safe_cam_id,
                track_id=track_id,
                bbox=bbox,
                detection_confidence=confidence,
                timestamp=safe_timestamp,
            )
        )

        # ----------------------------------------------------
        # VEHICLE RECORD
        # ----------------------------------------------------

        vehicle = {

            "camera_id":
                safe_cam_id,

            "track_id":
                track_id,

            "class_id":
                class_id,

            "vehicle_type":
                vehicle_type,

            # Detector confidence.
            "confidence":
                confidence,

            "detection_confidence":
                confidence,

            # Geometry.
            "bbox":
                bbox,

            "center":
                [
                    center_x,
                    center_y,
                ],

            "timestamp":
                safe_timestamp,

            # Tracking evidence.
            "track_quality":
                quality[
                    "track_quality"
                ],

            "tracking_confidence":
                quality[
                    "tracking_confidence"
                ],

            "track_age_frames":
                quality[
                    "track_age_frames"
                ],

            "track_stability":
                quality[
                    "track_stability"
                ],

            "track_iou":
                quality[
                    "track_iou"
                ],

            "track_status":
                quality[
                    "track_status"
                ],

            "is_tracked":
                track_id is not None,
        }

        vehicles.append(
            vehicle
        )

        # ----------------------------------------------------
        # VEHICLE LOG
        # ----------------------------------------------------

        _log(
            f"VEHICLE_TRACK "
            f"camera={safe_cam_id} "
            f"track={track_id} "
            f"type={vehicle_type} "
            f"det_conf={confidence:.3f} "
            f"quality="
            f"{quality['track_quality']:.3f} "
            f"stability="
            f"{quality['track_stability']:.3f} "
            f"age="
            f"{quality['track_age_frames']} "
            f"status="
            f"{quality['track_status']} "
            f"bbox="
            f"{[round(v, 1) for v in bbox]}"
        )

    # --------------------------------------------------------
    # EXTRACTION SUMMARY
    # --------------------------------------------------------

    tracked_count = sum(
        1
        for vehicle in vehicles
        if vehicle.get(
            "is_tracked",
            False,
        )
    )

    untracked_count = (
        len(vehicles)
        - tracked_count
    )

    _log(
        f"VEHICLE_EXTRACTION_COMPLETE "
        f"camera={safe_cam_id} "
        f"vehicles={len(vehicles)} "
        f"tracked={tracked_count} "
        f"untracked={untracked_count}"
    )

    return vehicles


# ============================================================
# TRACKER HEALTH / OBSERVABILITY
# ============================================================

def tracker_health() -> dict[str, Any]:

    with _tracker_lock:

        person_camera_count = len(
            trackers
        )

    with _vehicle_tracker_lock:

        vehicle_camera_count = len(
            vehicle_trackers
        )

    with _vehicle_track_history_lock:

        vehicle_history_tracks = sum(
            len(history)
            for history
            in _vehicle_track_history.values()
        )

    health = {

        "engine":
            "INTEL-I ByteTrack",

        "person_tracker_cameras":
            person_camera_count,

        "vehicle_tracker_cameras":
            vehicle_camera_count,

        "vehicle_track_history_count":
            vehicle_history_tracks,

        "track_activation_threshold":
            TRACK_ACTIVATION_THRESHOLD,

        "minimum_matching_threshold":
            MINIMUM_MATCHING_THRESHOLD,

        "min_lost_track_buffer":
            MIN_LOST_TRACK_BUFFER,

        "max_lost_track_buffer":
            MAX_LOST_TRACK_BUFFER,

        "track_quality_enabled":
            True,

        "track_quality_components":
            [
                "detection_confidence",
                "bbox_temporal_stability",
                "track_maturity",
            ],
    }

    _log(
        f"TRACKER_HEALTH "
        f"person_cameras="
        f"{person_camera_count} "
        f"vehicle_cameras="
        f"{vehicle_camera_count} "
        f"vehicle_history="
        f"{vehicle_history_tracks}"
    )

    return health