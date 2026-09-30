from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Dict, Tuple, Optional, Any
import math
import threading
from config import TRACK_HISTORY_SECONDS, STATE_CLEANUP_SECONDS
from poseutils import kp, LEFT_WRIST, RIGHT_WRIST, norm_dist


MAX_HISTORY_SECONDS = max(
    1.0,
    min(float(TRACK_HISTORY_SECONDS), 300.0),
)

MAX_STATE_CLEANUP_SECONDS = max(
    MAX_HISTORY_SECONDS,
    min(float(STATE_CLEANUP_SECONDS), 600.0),
)

MIN_TIME_DELTA = 1e-3

# A backwards timestamp or a very large discontinuity means that
# the temporal context should not be carried into the new segment.
MAX_TIMESTAMP_GAP_SECONDS = max(
    MAX_STATE_CLEANUP_SECONDS,
    MAX_HISTORY_SECONDS * 2.0,
)

MAX_RULE_VOTE_WINDOW_SECONDS = max(
    0.1,
    min(MAX_HISTORY_SECONDS, 30.0),
)


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    hits: int = 0

    centers: deque = field(default_factory=deque)
    boxes: deque = field(default_factory=deque)
    keypoints: deque = field(default_factory=deque)
    speeds: deque = field(default_factory=deque)
    wrist_speeds: deque = field(default_factory=deque)

    zone_entries: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    inside_zones: Dict[str, bool] = field(default_factory=dict)

    rule_start: Dict[str, float] = field(default_factory=dict)

    rule_votes: Dict[str, deque] = field(
        default_factory=lambda: defaultdict(deque)
    )

    last_alert: Dict[str, float] = field(default_factory=dict)

    # Segment ID prevents temporal state from being interpreted as
    # continuous after a stream reconnect or scene discontinuity.
    segment_id: int = 0


# ============================================================
# PAIR STATE
# ============================================================

@dataclass
class PairState:
    last_seen: float

    rule_start: Dict[str, float] = field(default_factory=dict)

    rule_votes: Dict[str, deque] = field(
        default_factory=lambda: defaultdict(deque)
    )

    last_alert: Dict[str, float] = field(default_factory=dict)

    segment_id: int = 0


# ============================================================
# RUNTIME STATE
# ============================================================

class RuntimeState:
    """
    Per-camera temporal state used by EventFusion.

    Design goals:
        - deterministic timestamps supplied by the caller
        - no dependence on frame arrival wall-clock timing
        - bounded history
        - safe handling of timestamp discontinuities
        - explicit camera-segment reset support
        - thread-safe camera/track state access
        - no sensitive logging
    """

    def __init__(self):
        self.tracks: Dict[str, Dict[int, TrackState]] = defaultdict(dict)
        self.pairs: Dict[str, Dict[Tuple[int, int], PairState]] = defaultdict(dict)

        # Logical segment for each camera.
        self.camera_segments: Dict[str, int] = defaultdict(int)

        # Last logical timestamp observed for each camera.
        self.last_camera_ts: Dict[str, float] = {}

        self._lock = threading.RLock()

    # ========================================================
    # INTERNAL VALIDATION
    # ========================================================

    @staticmethod
    def _safe_timestamp(ts: Any) -> Optional[float]:
        try:
            value = float(ts)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value):
            return None

        return value

    @staticmethod
    def _safe_track_id(track_id: Any) -> Optional[int]:
        try:
            value = int(track_id)
        except (TypeError, ValueError):
            return None

        return value

    @staticmethod
    def _safe_camera_id(cam_id: Any) -> Optional[str]:
        if cam_id is None:
            return None

        value = str(cam_id).strip()

        if not value:
            return None

        if len(value) > 256:
            return None

        return value

    @staticmethod
    def _safe_frame_dimension(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 1.0

        if not math.isfinite(result) or result <= 0:
            return 1.0

        return result

    # ========================================================
    # TEMPORAL CLEANUP
    # ========================================================

    @staticmethod
    def _trim_history(history: deque, now: float, window: float):
        while history:
            item = history[0]

            if not isinstance(item, tuple) or len(item) < 1:
                history.popleft()
                continue

            item_ts = RuntimeState._safe_timestamp(item[0])

            if item_ts is None or now - item_ts > window:
                history.popleft()
                continue

            break

    @staticmethod
    def _clear_rule_state(state):
        state.rule_start.clear()
        state.rule_votes.clear()
        state.last_alert.clear()

    # ========================================================
    # CAMERA SEGMENTS
    # ========================================================

    def get_camera_segment(self, cam_id) -> int:
        safe_cam_id = self._safe_camera_id(cam_id)

        if safe_cam_id is None:
            return 0

        with self._lock:
            return int(self.camera_segments.get(safe_cam_id, 0))

    def begin_camera_segment(
        self,
        cam_id,
        ts: Optional[float] = None,
        clear_tracks: bool = True,
    ) -> int:
        """
        Start a new logical video segment.

        Call this after:
            - RTSP reconnect
            - decoder restart
            - camera source restart
            - explicit scene discontinuity

        This prevents rule durations/votes from crossing a stream
        discontinuity.
        """

        safe_cam_id = self._safe_camera_id(cam_id)

        if safe_cam_id is None:
            return 0

        safe_ts = self._safe_timestamp(ts)

        with self._lock:
            next_segment = (
                int(self.camera_segments.get(safe_cam_id, 0)) + 1
            )

            self.camera_segments[safe_cam_id] = next_segment

            if clear_tracks:
                self.tracks.pop(safe_cam_id, None)
                self.pairs.pop(safe_cam_id, None)
            else:
                for state in self.tracks.get(safe_cam_id, {}).values():
                    state.segment_id = next_segment
                    state.first_seen = safe_ts if safe_ts is not None else state.last_seen
                    state.last_seen = (
                        safe_ts if safe_ts is not None else state.last_seen
                    )
                    state.hits = 0
                    state.centers.clear()
                    state.boxes.clear()
                    state.keypoints.clear()
                    state.speeds.clear()
                    state.wrist_speeds.clear()
                    state.zone_entries.clear()
                    state.inside_zones.clear()
                    self._clear_rule_state(state)

                for state in self.pairs.get(safe_cam_id, {}).values():
                    state.segment_id = next_segment
                    state.last_seen = (
                        safe_ts if safe_ts is not None else state.last_seen
                    )
                    self._clear_rule_state(state)

            if safe_ts is not None:
                self.last_camera_ts[safe_cam_id] = safe_ts
            else:
                self.last_camera_ts.pop(safe_cam_id, None)

            return next_segment

    def reset_camera(self, cam_id, ts: Optional[float] = None):
        """
        Fully reset temporal analytics state for a camera.

        This is intentionally separate from cleanup() because cleanup
        is time-based while reset_camera() represents a known stream
        discontinuity.
        """

        return self.begin_camera_segment(
            cam_id=cam_id,
            ts=ts,
            clear_tracks=True,
        )

    def _ensure_camera_segment(self, cam_id: str, ts: float):
        previous_ts = self.last_camera_ts.get(cam_id)

        if previous_ts is not None:
            delta = ts - previous_ts

            # A backwards clock or a large logical gap means the
            # previous temporal context is no longer trustworthy.
            if delta < 0 or delta > MAX_TIMESTAMP_GAP_SECONDS:
                self.begin_camera_segment(
                    cam_id=cam_id,
                    ts=ts,
                    clear_tracks=True,
                )

        self.last_camera_ts[cam_id] = ts

    # ========================================================
    # TRACK UPDATE
    # ========================================================

    def update_track(
        self,
        cam_id,
        track_id,
        box,
        center,
        kpts,
        frame_w,
        frame_h,
        ts,
    ):
        safe_cam_id = self._safe_camera_id(cam_id)
        safe_track_id = self._safe_track_id(track_id)
        safe_ts = self._safe_timestamp(ts)

        if safe_cam_id is None:
            return None

        if safe_track_id is None:
            return None

        if safe_ts is None:
            return None

        if box is None or center is None:
            return None

        frame_w = self._safe_frame_dimension(frame_w)
        frame_h = self._safe_frame_dimension(frame_h)

        with self._lock:
            self._ensure_camera_segment(
                safe_cam_id,
                safe_ts,
            )

            tracks = self.tracks[safe_cam_id]
            current_segment = int(
                self.camera_segments.get(
                    safe_cam_id,
                    0,
                )
            )

            if safe_track_id not in tracks:
                tracks[safe_track_id] = TrackState(
                    first_seen=safe_ts,
                    last_seen=safe_ts,
                    segment_id=current_segment,
                )

            st = tracks[safe_track_id]

            # Never allow a state object from an old segment to
            # silently continue into a new stream segment.
            if st.segment_id != current_segment:
                st = TrackState(
                    first_seen=safe_ts,
                    last_seen=safe_ts,
                    segment_id=current_segment,
                )
                tracks[safe_track_id] = st

            previous_ts = st.last_seen

            # A track timestamp must never move backwards.
            if safe_ts < previous_ts:
                return st

            st.last_seen = safe_ts
            st.hits += 1

            st.centers.append(
                (
                    safe_ts,
                    center,
                )
            )

            st.boxes.append(
                (
                    safe_ts,
                    box,
                )
            )

            if kpts is not None:
                st.keypoints.append(
                    (
                        safe_ts,
                        kpts,
                    )
                )

            self._trim_history(
                st.centers,
                safe_ts,
                MAX_HISTORY_SECONDS,
            )

            self._trim_history(
                st.boxes,
                safe_ts,
                MAX_HISTORY_SECONDS,
            )

            self._trim_history(
                st.keypoints,
                safe_ts,
                MAX_HISTORY_SECONDS,
            )

            # ----------------------------------------------------
            # CENTER SPEED
            # ----------------------------------------------------

            if len(st.centers) >= 2:
                t1, c1 = st.centers[-2]
                t2, c2 = st.centers[-1]

                dt = t2 - t1

                if (
                    math.isfinite(dt)
                    and dt >= MIN_TIME_DELTA
                ):
                    try:
                        speed = (
                            norm_dist(
                                c1,
                                c2,
                                frame_w,
                                frame_h,
                            )
                            / dt
                        )

                        if math.isfinite(float(speed)):
                            st.speeds.append(
                                (
                                    safe_ts,
                                    float(speed),
                                )
                            )
                    except (
                        TypeError,
                        ValueError,
                        ZeroDivisionError,
                    ):
                        pass

            self._trim_history(
                st.speeds,
                safe_ts,
                MAX_HISTORY_SECONDS,
            )

            # ----------------------------------------------------
            # WRIST SPEED
            # ----------------------------------------------------

            if len(st.keypoints) >= 2:
                t1, k1 = st.keypoints[-2]
                t2, k2 = st.keypoints[-1]

                dt = t2 - t1

                if (
                    math.isfinite(dt)
                    and dt >= MIN_TIME_DELTA
                ):
                    wrist_speeds = []

                    for idx in (
                        LEFT_WRIST,
                        RIGHT_WRIST,
                    ):
                        try:
                            p1 = kp(k1, idx)
                            p2 = kp(k2, idx)
                        except Exception:
                            p1 = None
                            p2 = None

                        if p1 and p2:
                            try:
                                wrist_speed = (
                                    norm_dist(
                                        p1,
                                        p2,
                                        frame_w,
                                        frame_h,
                                    )
                                    / dt
                                )

                                if math.isfinite(
                                    float(wrist_speed)
                                ):
                                    wrist_speeds.append(
                                        float(wrist_speed)
                                    )
                            except (
                                TypeError,
                                ValueError,
                                ZeroDivisionError,
                            ):
                                continue

                    if wrist_speeds:
                        st.wrist_speeds.append(
                            (
                                safe_ts,
                                max(wrist_speeds),
                            )
                        )

            self._trim_history(
                st.wrist_speeds,
                safe_ts,
                MAX_HISTORY_SECONDS,
            )

            return st

    # ========================================================
    # ZONE ENTRY
    # ========================================================

    def update_zone_entry(
        self,
        track_state,
        zone_name,
        inside_now,
    ):
        if track_state is None:
            return 0

        if zone_name is None:
            return 0

        zone_key = str(zone_name).strip()

        if not zone_key or len(zone_key) > 256:
            return 0

        previous = track_state.inside_zones.get(
            zone_key,
            False,
        )

        inside_now = bool(inside_now)

        if inside_now and not previous:
            track_state.zone_entries[zone_key] += 1

        track_state.inside_zones[zone_key] = inside_now

        return int(
            track_state.zone_entries[zone_key]
        )

    # ========================================================
    # PAIR STATE
    # ========================================================

    def get_pair(
        self,
        cam_id,
        id1,
        id2,
        ts,
    ):
        safe_cam_id = self._safe_camera_id(cam_id)
        safe_id1 = self._safe_track_id(id1)
        safe_id2 = self._safe_track_id(id2)
        safe_ts = self._safe_timestamp(ts)

        if (
            safe_cam_id is None
            or safe_id1 is None
            or safe_id2 is None
            or safe_ts is None
        ):
            return (
                (
                    0,
                    0,
                ),
                PairState(
                    last_seen=safe_ts or 0.0
                ),
            )

        with self._lock:
            self._ensure_camera_segment(
                safe_cam_id,
                safe_ts,
            )

            pair = tuple(
                sorted(
                    [
                        safe_id1,
                        safe_id2,
                    ]
                )
            )

            current_segment = int(
                self.camera_segments.get(
                    safe_cam_id,
                    0,
                )
            )

            pair_state = self.pairs[
                safe_cam_id
            ].get(pair)

            if pair_state is None:
                pair_state = PairState(
                    last_seen=safe_ts,
                    segment_id=current_segment,
                )
                self.pairs[
                    safe_cam_id
                ][pair] = pair_state

            elif pair_state.segment_id != current_segment:
                pair_state = PairState(
                    last_seen=safe_ts,
                    segment_id=current_segment,
                )
                self.pairs[
                    safe_cam_id
                ][pair] = pair_state

            else:
                pair_state.last_seen = max(
                    pair_state.last_seen,
                    safe_ts,
                )

            return pair, pair_state

    # ========================================================
    # RULE VOTING
    # ========================================================

    def vote(
        self,
        state,
        rule,
        condition,
        ts,
        window=2.0,
    ):
        if state is None:
            return 0.0

        safe_ts = self._safe_timestamp(ts)

        if safe_ts is None:
            return 0.0

        if not rule:
            return 0.0

        rule_key = str(rule).strip()

        if not rule_key or len(rule_key) > 256:
            return 0.0

        try:
            requested_window = float(window)
        except (
            TypeError,
            ValueError,
        ):
            requested_window = 2.0

        if (
            not math.isfinite(
                requested_window
            )
            or requested_window <= 0
        ):
            requested_window = 2.0

        vote_window = min(
            requested_window,
            MAX_RULE_VOTE_WINDOW_SECONDS,
        )

        with self._lock:
            dq = state.rule_votes[
                rule_key
            ]

            dq.append(
                (
                    safe_ts,
                    bool(condition),
                )
            )

            while dq:
                item_ts = self._safe_timestamp(
                    dq[0][0]
                )

                if (
                    item_ts is None
                    or safe_ts - item_ts > vote_window
                ):
                    dq.popleft()
                    continue

                break

            if not dq:
                return 0.0

            positive = sum(
                1
                for _, value in dq
                if bool(value)
            )

            ratio = positive / len(dq)

            if not math.isfinite(
                float(ratio)
            ):
                return 0.0

            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        ratio,
                    ),
                )
            )

    # ========================================================
    # RULE START / DURATION
    # ========================================================

    def start_rule(
        self,
        state,
        rule,
        condition,
        ts,
    ):
        if state is None:
            return

        safe_ts = self._safe_timestamp(ts)

        if safe_ts is None:
            return

        if not rule:
            return

        rule_key = str(rule).strip()

        if not rule_key or len(rule_key) > 256:
            return

        with self._lock:
            if bool(condition):
                state.rule_start.setdefault(
                    rule_key,
                    safe_ts,
                )
            else:
                state.rule_start.pop(
                    rule_key,
                    None,
                )

    def duration(
        self,
        state,
        rule,
        ts,
    ):
        if state is None:
            return 0.0

        safe_ts = self._safe_timestamp(ts)

        if safe_ts is None:
            return 0.0

        if not rule:
            return 0.0

        rule_key = str(rule).strip()

        if not rule_key:
            return 0.0

        with self._lock:
            start = state.rule_start.get(
                rule_key
            )

            if start is None:
                return 0.0

            start = self._safe_timestamp(start)

            if start is None:
                state.rule_start.pop(
                    rule_key,
                    None,
                )
                return 0.0

            duration = safe_ts - start

            # A negative duration means a timestamp discontinuity.
            if duration < 0:
                state.rule_start.pop(
                    rule_key,
                    None,
                )
                return 0.0

            if not math.isfinite(
                duration
            ):
                return 0.0

            return float(
                min(
                    duration,
                    MAX_STATE_CLEANUP_SECONDS,
                )
            )

    # ========================================================
    # ALERT COOLDOWN
    # ========================================================

    def can_alert(
        self,
        state,
        rule,
        cooldown,
        ts,
    ):
        if state is None:
            return False

        safe_ts = self._safe_timestamp(ts)

        if safe_ts is None:
            return False

        if not rule:
            return False

        rule_key = str(rule).strip()

        if not rule_key or len(rule_key) > 256:
            return False

        try:
            safe_cooldown = float(cooldown)
        except (
            TypeError,
            ValueError,
        ):
            safe_cooldown = 0.0

        if (
            not math.isfinite(
                safe_cooldown
            )
            or safe_cooldown < 0
        ):
            safe_cooldown = 0.0

        with self._lock:
            last = state.last_alert.get(
                rule_key
            )

            if last is None:
                state.last_alert[
                    rule_key
                ] = safe_ts
                return True

            last = self._safe_timestamp(
                last
            )

            if last is None:
                state.last_alert[
                    rule_key
                ] = safe_ts
                return True

            elapsed = safe_ts - last

            # Never let a backwards timestamp trigger a duplicate
            # alert by accident.
            if elapsed < 0:
                state.last_alert[
                    rule_key
                ] = safe_ts
                return False

            if elapsed >= safe_cooldown:
                state.last_alert[
                    rule_key
                ] = safe_ts
                return True

            return False

    # ========================================================
    # CAMERA CLEANUP
    # ========================================================

    def cleanup(
        self,
        cam_id,
        ts,
    ):
        safe_cam_id = self._safe_camera_id(cam_id)
        safe_ts = self._safe_timestamp(ts)

        if safe_cam_id is None or safe_ts is None:
            return

        with self._lock:
            tracks = self.tracks.get(
                safe_cam_id
            )

            if tracks:
                expired_tracks = [
                    tid
                    for tid, state in tracks.items()
                    if (
                        safe_ts - state.last_seen
                        > MAX_STATE_CLEANUP_SECONDS
                    )
                ]

                for tid in expired_tracks:
                    tracks.pop(
                        tid,
                        None,
                    )

                if not tracks:
                    self.tracks.pop(
                        safe_cam_id,
                        None,
                    )

            pairs = self.pairs.get(
                safe_cam_id
            )

            if pairs:
                expired_pairs = [
                    pair
                    for pair, state in pairs.items()
                    if (
                        safe_ts - state.last_seen
                        > MAX_STATE_CLEANUP_SECONDS
                    )
                ]

                for pair in expired_pairs:
                    pairs.pop(
                        pair,
                        None,
                    )

                if not pairs:
                    self.pairs.pop(
                        safe_cam_id,
                        None,
                    )

            # If the camera has no active state left, retain only
            # the logical segment marker and latest timestamp.
            self.last_camera_ts[
                safe_cam_id
            ] = safe_ts

    # ========================================================
    # FULL CAMERA REMOVAL
    # ========================================================

    def remove_camera(
        self,
        cam_id,
    ):
        safe_cam_id = self._safe_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return

        with self._lock:
            self.tracks.pop(
                safe_cam_id,
                None,
            )

            self.pairs.pop(
                safe_cam_id,
                None,
            )

            self.camera_segments.pop(
                safe_cam_id,
                None,
            )

            self.last_camera_ts.pop(
                safe_cam_id,
                None,
            )


# ============================================================
# LEGACY ACTIVE STREAM HELPERS
# ============================================================
#
# These are retained for compatibility with existing callers.
# Access is protected because camera workers run in threads.
# The Redis-backed camera state remains the authoritative state
# in the current INTEL-I architecture.


active_streams = {}
_active_streams_lock = threading.RLock()


def get_active_stream_count():
    with _active_streams_lock:
        return len(active_streams)


def is_camera_running(cam_id: str):
    if cam_id is None:
        return False

    safe_cam_id = str(cam_id).strip()

    if not safe_cam_id:
        return False

    with _active_streams_lock:
        return safe_cam_id in active_streams


def add_active_stream(
    cam_id: str,
    thread,
):
    if cam_id is None:
        return

    safe_cam_id = str(cam_id).strip()

    if not safe_cam_id:
        return

    with _active_streams_lock:
        active_streams[
            safe_cam_id
        ] = thread


def remove_active_stream(
    cam_id: str,
):
    if cam_id is None:
        return

    safe_cam_id = str(cam_id).strip()

    if not safe_cam_id:
        return

    with _active_streams_lock:
        active_streams.pop(
            safe_cam_id,
            None,
        )