import math
import os
import time
import logging

from config import *
from poseutils import (human_pose_valid,aggressive_pose_score,visibility_score,norm_dist,hand_regions,boxes_overlap,overlap_ratio,body_horizontal_score)

logger = logging.getLogger(__name__)

try:
    from services.intelligence_persistence import save_activity_event
except Exception:
    save_activity_event = None

def _finite_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def mean_values(items):
    values = []
    for item in items or []:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        value = _finite_float(item[1], None)
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else 0.0


def max_values(items):
    values = []
    for item in items or []:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        value = _finite_float(item[1], None)
        if value is not None:
            values.append(value)
    return max(values) if values else 0.0

def movement_radius_norm(track_state, frame_w, frame_h):
    if track_state is None or len(track_state.centers) < 2:
        return 0.0

    safe_w = max(1.0, _finite_float(frame_w, 1.0))
    safe_h = max(1.0, _finite_float(frame_h, 1.0))

    points = []
    for item in track_state.centers:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        point = item[1]
        if not point or len(point) < 2:
            continue
        x = _finite_float(point[0], None)
        y = _finite_float(point[1], None)
        if x is not None and y is not None:
            points.append((x, y))

    if len(points) < 2:
        return 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    radius = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    diag = math.hypot(safe_w, safe_h)

    return max(0.0, min(1.0, radius / max(1.0, diag)))

def movement_vector(track_state):
    if track_state is None or len(track_state.centers) < 3:
        return None

    try:
        _, p1 = track_state.centers[0]
        _, p2 = track_state.centers[-1]
        dx = _finite_float(p2[0]) - _finite_float(p1[0])
        dy = _finite_float(p2[1]) - _finite_float(p1[1])
    except (TypeError, ValueError, IndexError):
        return None

    mag = math.hypot(dx, dy)
    if not math.isfinite(mag) or mag < 1.0:
        return None

    return dx / mag, dy / mag

def cosine_similarity(v1, v2):
    if not v1 or not v2 or len(v1) < 2 or len(v2) < 2:
        return 0.0

    value = (
        _finite_float(v1[0]) * _finite_float(v2[0])
        + _finite_float(v1[1]) * _finite_float(v2[1])
    )

    if not math.isfinite(value):
        return 0.0

    return max(-1.0, min(1.0, value))

def object_inside_head_area(person_box, obj_box):
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box

    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    head_x1 = px1 - int(pw * 0.10)
    head_y1 = py1 - int(ph * 0.05)
    head_x2 = px2 + int(pw * 0.10)
    head_y2 = py1 + int(ph * 0.45)

    obj_cx = (ox1 + ox2) / 2
    obj_cy = (oy1 + oy2) / 2

    overlap_ok = overlap_ratio(obj_box, (head_x1, head_y1, head_x2, head_y2)) >= 0.15
    center_ok = head_x1 <= obj_cx <= head_x2 and head_y1 <= obj_cy <= head_y2

    return overlap_ok or center_ok


def crime_rule_name(class_name):
    name = str(class_name or "").lower().strip()

    if name in GUN_CLASSES:
        return "Gun Handling"

    if name in KNIFE_CLASSES:
        return "Knife Handling"

    if name in FIGHT_CLASSES:
        return "Fighting"

    if name in ASSAULT_CLASSES:
        return "Assault Risk"

    if name in THEFT_CLASSES:
        return "Theft / Robbery"

    if name in ABDUCTION_CLASSES:
        return "Possible Abduction Risk"

    if name in SUSPICIOUS_CLASSES:
        return "Suspicious Behaviour"

    if name == "hardhat" or name == "mask":
        return "Face Covered Walking"

    return "Crime Activity"


def crime_near_person(crime_box, person_box):
    return overlap_ratio(crime_box, person_box) >= 0.20


def crime_near_hand(crime_box, kpts, person_box):
    if kpts is None:
        return False

    for hand_box in hand_regions(kpts, person_box):
        if boxes_overlap(crime_box, hand_box, min_ratio=0.20):
            return True

    return False


def merge_boxes(a, b):
    if not a or not b or len(a) < 4 or len(b) < 4:
        return None

    values = [
        _finite_float(a[0]), _finite_float(a[1]),
        _finite_float(a[2]), _finite_float(a[3]),
        _finite_float(b[0]), _finite_float(b[1]),
        _finite_float(b[2]), _finite_float(b[3]),
    ]

    return (
        min(values[0], values[4]),
        min(values[1], values[5]),
        max(values[2], values[6]),
        max(values[3], values[7]),
    )


class EventFusion:
    def __init__(self, runtime_state):
        self.state = runtime_state

        # PostgreSQL persistence is deliberately event-level, not frame-level.
        # This set prevents repeated DB writes while the same confirmed event
        # remains active in the current worker.
        self._persisted_event_keys = set()

        self.eventfusion_version = os.getenv(
            "EVENTFUSION_VERSION",
            "1.0",
        )

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def _event_severity(self, rule):
        """
        Map EventFusion rules to durable severity.

        The real-time rule engine remains unchanged; this only supplies
        a normalized severity for the persistent ActivityEvent record.
        """
        critical = {
            "Fall Emergency",
            "Fighting",
            "Assault Risk",
            "Theft / Robbery",
            "Possible Abduction Risk",
            "Chain Snatching Risk",
            "Suspicious Escalation",
            "Gun Handling",
            "Knife Handling",
        }

        high = {
            "Perimeter Breach",
            "Face Covered Walking",
        }

        medium = {
            "Loitering",
            "Repeated Entry Exit",
            "Suspicious Behaviour",
            "Following Behaviour",
        }

        if rule in critical:
            return "CRITICAL"

        if rule in high:
            return "HIGH"

        if rule in medium:
            return "MEDIUM"

        return "LOW"

    def _persist_confirmed_event(
        self,
        cam_id,
        track_id,
        alert,
        ts,
    ):
        """
        Persist one confirmed EventFusion event.

        IMPORTANT:
        This function is called only after an EventFusion rule has
        passed its temporal confirmation threshold. It is therefore
        NOT a per-video-frame database write.

        Redis/runtime_state remains responsible for high-frequency
        rule votes, durations, pair state, and active tracking.
        """
        if save_activity_event is None:
            return

        if not isinstance(alert, dict):
            return

        rule = str(
            alert.get("rule")
            or "UNKNOWN"
        )

        track = str(
            alert.get("track_id")
            or track_id
            or "global"
        )

        camera = str(
            cam_id
            or "unknown"
        )

        event_key = (
            f"{camera}:"
            f"{track}:"
            f"{rule}"
        )

        if event_key in self._persisted_event_keys:
            return

        self._persisted_event_keys.add(event_key)

        confidence = _finite_float(
            alert.get("confidence"),
            0.0,
        )

        evidence = alert.get(
            "evidence",
            {},
        )

        if not isinstance(evidence, dict):
            evidence = {
                "raw": str(evidence),
            }

        # Use stream timestamp as evidence metadata. Persistence time is
        # supplied by the persistence layer/database.
        evidence = {
            **evidence,
            "eventfusion_timestamp": _finite_float(
                ts,
                0.0,
            ),
            "eventfusion_event_key": event_key,
        }

        # ActivityEvent.event_id is unique. Including the camera, track,
        # rule and confirmation timestamp makes the ID deterministic enough
        # for retries while remaining unique for separate occurrences.
        event_id = (
            f"ACT-{camera}-"
            f"{track}-"
            f"{rule}-"
            f"{int(max(0.0, _finite_float(ts, 0.0)) * 1000)}"
        )

        try:
            save_activity_event(
                event_id=event_id,
                camera_id=camera,
                track_id=track,
                rule_id=rule,
                event_type=rule,
                severity=self._event_severity(rule),
                started_at=None,
                ended_at=None,
                confidence=confidence,
                status="CONFIRMED",
                evidence=evidence,
                model_name="EventFusion",
                model_version=self.eventfusion_version,
                metadata={
                    "source": "eventfusion",
                    "persisted_once": True,
                },
            )

        except Exception:
            # Persistence must never stop live CCTV analytics.
            # Remove the key so a later confirmed evaluation can retry.
            self._persisted_event_keys.discard(
                event_key
            )

            logger.exception(
                "Failed to persist ActivityEvent | "
                "camera=%s track=%s rule=%s",
                camera,
                track,
                rule,
            )

    def clear_persistence_key(
        self,
        cam_id,
        track_id,
        rule,
    ):
        """
        Allow a future occurrence of the same rule/track to become a new
        persistent event after the old event has ended.

        The main cleanup path can call this when a track disappears.
        """
        key = (
            f"{cam_id}:"
            f"{track_id}:"
            f"{rule}"
        )

        self._persisted_event_keys.discard(
            key
        )

    def track_ready(self, track_state, ts):
        if track_state is None:
            return False

        current_ts = _finite_float(ts, None)
        first_seen = _finite_float(getattr(track_state, "first_seen", None), None)
        hits = int(getattr(track_state, "hits", 0) or 0)

        if current_ts is None or first_seen is None or hits < MIN_TRACK_HITS:
            return False

        age = current_ts - first_seen

        # A negative age means the logical stream clock moved backwards.
        if age < 0 or not math.isfinite(age):
            return False

        return age >= MIN_TRACK_AGE_SECONDS

    def evaluate_track(
        self,
        cam_id,
        track_id,
        track_state,
        box,
        center,
        kpts,
        frame_w,
        frame_h,
        ts,
        crime_hits,
        restricted_zone=False,
        restricted_zone_name=None,
    ):
        alerts = []

        safe_ts = _finite_float(ts, None)
        if safe_ts is None:
            return alerts

        ts = safe_ts

        if not self.track_ready(track_state, ts):
            return alerts

        pose_ok = human_pose_valid(kpts)
        pose_score = visibility_score(kpts) if kpts is not None else 0.0
        avg_speed = mean_values(track_state.speeds)
        max_wrist_speed = max_values(track_state.wrist_speeds)
        radius = movement_radius_norm(track_state, frame_w, frame_h)
        aggressive_score = aggressive_pose_score(kpts) if pose_ok else 0.0

        # ---------------- RUNNING ----------------
        if pose_ok:
            rule = "Sudden Running"

            condition = avg_speed >= RUNNING_SPEED_NORM and pose_score >= 0.30

            vote = self.state.vote(track_state, rule, condition, ts, window=2.0)
            self.state.start_rule(track_state, rule, vote >= 0.60, ts)

            confidence = min(
                0.96,
                0.35
                + min(avg_speed / max(RUNNING_SPEED_NORM, 0.001), 1.5) * 0.30
                + pose_score * 0.25,
            )

            if (
                self.state.duration(track_state, rule, ts) >= RUNNING_MIN_SECONDS
                and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
            ):
                alerts.append(
                    self.make_alert(
                        rule,
                        track_id,
                        box,
                        confidence,
                        {
                            "avg_speed": round(avg_speed, 3),
                            "pose_score": round(pose_score, 3),
                        },
                    )
                )

        # ---------------- LOITERING ----------------
        if pose_ok:
            rule = "Loitering"
            age = ts - track_state.first_seen

            condition = (
                age >= LOITERING_MIN_SECONDS
                and radius <= LOITERING_RADIUS_NORM
                and avg_speed < RUNNING_SPEED_NORM * 0.35
                and pose_score >= 0.45
            )

            vote = self.state.vote(track_state, rule, condition, ts, window=8.0)
            self.state.start_rule(track_state, rule, vote >= 0.80, ts)

            confidence = min(
                0.94,
                0.30
                + min(age / LOITERING_MIN_SECONDS, 1.5) * 0.22
                + max(0, 1 - radius * 12) * 0.25
                + pose_score * 0.17,
            )

            if (
                self.state.duration(track_state, rule, ts) >= 5.0
                and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
            ):
                alerts.append(
                    self.make_alert(
                        rule,
                        track_id,
                        box,
                        confidence,
                        {
                            "age_seconds": round(age, 1),
                            "movement_radius_norm": round(radius, 3),
                        },
                    )
                )

        # ---------------- FALL ----------------
        if pose_ok:
            rule = "Fall Emergency"
            horizontal_score = body_horizontal_score(kpts)

            condition = (
                horizontal_score >= 0.65
                and avg_speed <= RUNNING_SPEED_NORM * 0.25
                and pose_score >= 0.45
            )

            vote = self.state.vote(track_state, rule, condition, ts, window=4.0)
            self.state.start_rule(track_state, rule, vote >= 0.75, ts)

            confidence = min(
                0.96,
                0.35
                + horizontal_score * 0.35
                + max(0, 1 - avg_speed * 4) * 0.20
                + pose_score * 0.10,
            )

            if (
                self.state.duration(track_state, rule, ts) >= FALL_CONFIRM_SECONDS
                and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.8)
            ):
                alerts.append(
                    self.make_alert(
                        rule,
                        track_id,
                        box,
                        confidence,
                        {
                            "horizontal_score": round(horizontal_score, 3),
                            "avg_speed": round(avg_speed, 3),
                        },
                    )
                )

        # ---------------- RESTRICTED ZONE ----------------
        if restricted_zone and pose_ok:
            rule = "Perimeter Breach"

            condition = pose_score >= 0.45

            vote = self.state.vote(track_state, rule, condition, ts, window=3.0)
            self.state.start_rule(track_state, rule, vote >= 0.70, ts)

            confidence = min(0.95, 0.55 + pose_score * 0.25)

            if (
                self.state.duration(track_state, rule, ts) >= PERIMETER_CONFIRM_SECONDS
                and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
            ):
                alerts.append(
                    self.make_alert(
                        rule,
                        track_id,
                        box,
                        confidence,
                        {
                            "zone": restricted_zone_name,
                            "pose_score": round(pose_score, 3),
                        },
                    )
                )

            entry_count = self.state.update_zone_entry(
                track_state,
                restricted_zone_name or "restricted",
                True,
            )

            rule = "Repeated Entry Exit"
            condition = entry_count >= REPEATED_ENTRY_EXIT_MIN_COUNT

            vote = self.state.vote(track_state, rule, condition, ts, window=10.0)
            self.state.start_rule(track_state, rule, vote >= 0.80, ts)

            confidence = min(0.92, 0.45 + min(entry_count, 8) * 0.06)

            if (
                self.state.duration(track_state, rule, ts) >= 1.0
                and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
            ):
                alerts.append(
                    self.make_alert(
                        rule,
                        track_id,
                        box,
                        confidence,
                        {"entry_count": entry_count},
                    )
                )

        # ---------------- CRIME / PPE ----------------
        for crime in crime_hits:
            class_name = crime.get("name", "").lower().strip()

            if not class_name:
                continue

            if class_name in IGNORED_CRIME_CLASSES:
                continue

            crime_box = crime.get("box")
            crime_conf = _finite_float(crime.get("confidence", 0.0), 0.0)
            crime_conf = max(0.0, min(1.0, crime_conf))

            if crime_box is None:
                continue

            is_face_cover = class_name in FACE_COVER_CLASSES
            is_weapon = class_name in WEAPON_CLASSES
            is_fight = class_name in FIGHT_CLASSES
            is_theft = class_name in THEFT_CLASSES
            is_suspicious = class_name in SUSPICIOUS_CLASSES
            is_assault = class_name in ASSAULT_CLASSES
            is_abduction = class_name in ABDUCTION_CLASSES

            if (
                not is_face_cover
                and not is_weapon
                and not is_fight
                and not is_theft
                and not is_suspicious
                and not is_assault
                and not is_abduction
                and class_name not in CRITICAL_CRIME_CLASSES
            ):
                continue

            if is_face_cover:
                if crime_conf < PPE_CONF:
                    continue

                head_ok = object_inside_head_area(box, crime_box)

                if not head_ok:
                    continue

            else:
                if crime_conf < CRIME_CONF:
                    continue

                if not crime_near_person(crime_box, box):
                    continue

                if not pose_ok:
                    continue

            if is_weapon:
                rule = crime_rule_name(class_name)
                hand_ok = crime_near_hand(crime_box, kpts, box)

                condition = (
                    hand_ok
                    and pose_score >= 0.45
                    and crime_conf >= CRIME_CONF
                )

                base_conf = (
                    0.38
                    + crime_conf * 0.35
                    + pose_score * 0.12
                    + aggressive_score * 0.08
                    + (0.10 if hand_ok else 0)
                )

                confirm_time = WEAPON_CONFIRM_SECONDS

            elif is_fight:
                rule = "Fighting"

                condition = (
                    aggressive_score >= 0.35
                    and max_wrist_speed >= AGGRESSIVE_WRIST_SPEED_NORM * 0.60
                    and pose_score >= 0.45
                )

                base_conf = (
                    0.38
                    + crime_conf * 0.30
                    + aggressive_score * 0.20
                    + min(max_wrist_speed, 0.30) * 0.25
                )

                confirm_time = FIGHTING_MIN_SECONDS

            elif is_theft:
                rule = "Theft / Robbery"

                condition = (
                    pose_score >= 0.45
                    and (
                        max_wrist_speed >= SUSPICIOUS_WRIST_SPEED_NORM
                        or aggressive_score >= 0.25
                    )
                )

                base_conf = (
                    0.36
                    + crime_conf * 0.32
                    + pose_score * 0.12
                    + min(max_wrist_speed, 0.25) * 0.25
                )

                confirm_time = CRIME_CONFIRM_SECONDS

            elif is_assault:
                rule = "Assault Risk"

                condition = (
                    pose_score >= 0.45
                    and (
                        aggressive_score >= 0.25
                        or max_wrist_speed >= SUSPICIOUS_WRIST_SPEED_NORM * 0.70
                    )
                )

                base_conf = (
                    0.36
                    + crime_conf * 0.32
                    + aggressive_score * 0.18
                    + pose_score * 0.12
                )

                confirm_time = CRIME_CONFIRM_SECONDS

            elif is_face_cover:
                rule = "Face Covered Walking"

                confirm_time = FACE_COVER_CONFIRM_SECONDS

                moving_ok = avg_speed >= FACE_MASK_MIN_SPEED_NORM

                condition = (
                    crime_conf >= PPE_CONF
                    and head_ok
                    and moving_ok
                )

                base_conf = min(
                    0.96,
                    0.40
                    + crime_conf * 0.42
                    + pose_score * 0.10
                    + min(avg_speed, 0.15) * 0.35
                )
            elif is_abduction:
                rule = "Possible Abduction Risk"

                condition = (
                    pose_score >= 0.45
                    and crime_conf >= CRIME_CONF
                    and (
                        avg_speed >= ABDUCTION_MIN_SPEED_NORM
                        or max_wrist_speed >= ABDUCTION_WRIST_SPEED_NORM
                        or aggressive_score >= ABDUCTION_AGGRESSIVE_SCORE
                    )
                )

                base_conf = min(
                    0.98,
                    0.40
                    + crime_conf * 0.35
                    + pose_score * 0.10
                    + aggressive_score * 0.12
                    + min(max_wrist_speed, 0.30) * 0.25
                    + min(avg_speed, 0.20) * 0.25,
                )

                confirm_time = ABDUCTION_CONFIRM_SECONDS
            elif is_suspicious:
                rule = "Suspicious Behaviour"

                condition = (
                    pose_score >= 0.45
                    and (
                        max_wrist_speed >= SUSPICIOUS_WRIST_SPEED_NORM * 0.70
                        or avg_speed >= RUNNING_SPEED_NORM * 0.60
                        or aggressive_score >= 0.25
                    )
                )

                base_conf = (
                    0.34
                    + crime_conf * 0.32
                    + pose_score * 0.12
                    + aggressive_score * 0.12
                    + min(avg_speed, 0.30) * 0.18
                )

                confirm_time = 1.5

            else:
                rule = crime_rule_name(class_name)
                condition = pose_score >= 0.45 and crime_conf >= CRIME_CONF
                base_conf = 0.36 + crime_conf * 0.34 + pose_score * 0.13
                confirm_time = CRIME_CONFIRM_SECONDS

            vote = self.state.vote(track_state, rule, condition, ts, window=3.5)
            self.state.start_rule(track_state, rule, vote >= 0.70, ts)

            confidence = min(0.99, base_conf)

            if (
                self.state.duration(track_state, rule, ts) >= confirm_time
                and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
            ):
                alerts.append(
                    self.make_alert(
                        rule,
                        track_id,
                        box,
                        confidence,
                        {
                            "detected_class": class_name,
                            "model_conf": round(crime_conf, 3),
                            "pose_score": round(pose_score, 3),
                            "avg_speed": round(avg_speed, 3),
                            "head_area_match": True if is_face_cover else None,
                            "max_wrist_speed": round(max_wrist_speed, 3),
                            "aggressive_score": round(aggressive_score, 3),
                        },
                    )
                )

        # Persist only newly-confirmed events. The runtime state above
        # continues to handle high-frequency votes/durations.
        for alert in alerts:
            self._persist_confirmed_event(
                cam_id=cam_id,
                track_id=alert.get("track_id"),
                alert=alert,
                ts=ts,
            )

        return alerts

    def evaluate_pair(self, cam_id, id1, data1, id2, data2, frame_w, frame_h, ts):
        alerts = []

        safe_ts = _finite_float(ts, None)
        if safe_ts is None or not data1 or not data2:
            return alerts

        ts = safe_ts

        try:
            st1 = data1["state"]
            st2 = data2["state"]
        except (KeyError, TypeError):
            return alerts

        if not self.track_ready(st1, ts) or not self.track_ready(st2, ts):
            return alerts

        k1 = data1["kpts"]
        k2 = data2["kpts"]
        c1 = data1["center"]
        c2 = data2["center"]
        b1 = data1["box"]
        b2 = data2["box"]

        pose_ok = human_pose_valid(k1) and human_pose_valid(k2)

        if not pose_ok:
            return alerts

        pair_key, pair_state = self.state.get_pair(cam_id, id1, id2, ts)

        distance = norm_dist(c1, c2, frame_w, frame_h)
        wrist_speed = max(max_values(st1.wrist_speeds), max_values(st2.wrist_speeds))
        aggressive = max(aggressive_pose_score(k1), aggressive_pose_score(k2))

        speed1 = mean_values(st1.speeds)
        speed2 = mean_values(st2.speeds)

        # ---------------- CLOSE INTERACTION ----------------
        rule = "Close Interaction"

        condition = (
            distance <= CLOSE_DISTANCE_NORM
            and wrist_speed < SUSPICIOUS_WRIST_SPEED_NORM * 0.85
            and aggressive < 0.30
        )

        vote = self.state.vote(pair_state, rule, condition, ts, window=5.0)
        self.state.start_rule(pair_state, rule, vote >= 0.80, ts)

        confidence = min(
            0.92,
            0.50 + max(0, CLOSE_DISTANCE_NORM - distance) * 3.0,
        )

        if (
            self.state.duration(pair_state, rule, ts) >= CLOSE_INTERACTION_MIN_SECONDS
            and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
        ):
            alerts.append(
                self.make_pair_alert(
                    rule,
                    pair_key,
                    merge_boxes(b1, b2),
                    confidence,
                    {"distance_norm": round(distance, 3)},
                )
            )

        # ---------------- FOLLOWING BEHAVIOUR ----------------
        rule = "Following Behaviour"

        v1 = movement_vector(st1)
        v2 = movement_vector(st2)
        direction_match = cosine_similarity(v1, v2)

        condition = (
            distance <= FOLLOW_DISTANCE_NORM
            and distance > CLOSE_DISTANCE_NORM
            and direction_match >= FOLLOW_DIRECTION_COSINE
            and abs(speed1 - speed2) <= FOLLOW_SPEED_DIFF_MAX
            and speed1 > 0.015
            and speed2 > 0.015
        )

        vote = self.state.vote(pair_state, rule, condition, ts, window=8.0)
        self.state.start_rule(pair_state, rule, vote >= 0.85, ts)

        confidence = min(
            0.92,
            0.35
            + direction_match * 0.30
            + max(0, FOLLOW_DISTANCE_NORM - distance) * 1.4
            + min(speed1 + speed2, 0.20) * 0.8,
        )

        if (
            self.state.duration(pair_state, rule, ts) >= FOLLOW_MIN_SECONDS
            and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.70)
        ):
            alerts.append(
                self.make_pair_alert(
                    rule,
                    pair_key,
                    merge_boxes(b1, b2),
                    confidence,
                    {
                        "distance_norm": round(distance, 3),
                        "direction_match": round(direction_match, 3),
                        "speed1": round(speed1, 3),
                        "speed2": round(speed2, 3),
                    },
                )
            )

        # ---------------- SUSPICIOUS ESCALATION ----------------
        rule = "Suspicious Escalation"

        condition = (
            distance <= CLOSE_DISTANCE_NORM
            and wrist_speed >= SUSPICIOUS_WRIST_SPEED_NORM
            and aggressive >= 0.35
        )

        vote = self.state.vote(pair_state, rule, condition, ts, window=3.5)
        self.state.start_rule(pair_state, rule, vote >= 0.70, ts)

        confidence = min(
            0.98,
            0.45 + aggressive * 0.28 + min(wrist_speed, 0.30) * 0.35,
        )

        if (
            self.state.duration(pair_state, rule, ts) >= SUSPICIOUS_ESCALATION_MIN_SECONDS
            and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
        ):
            alerts.append(
                self.make_pair_alert(
                    rule,
                    pair_key,
                    merge_boxes(b1, b2),
                    confidence,
                    {
                        "distance_norm": round(distance, 3),
                        "wrist_speed": round(wrist_speed, 3),
                        "aggressive_score": round(aggressive, 3),
                    },
                )
            )

        # ---------------- CHAIN SNATCHING ----------------
        rule = "Chain Snatching Risk"

        condition = (
            distance <= CHAIN_SNATCH_DISTANCE_NORM
            and wrist_speed >= CHAIN_SNATCH_WRIST_SPEED_NORM
            and aggressive >= CHAIN_SNATCH_AGGRESSIVE_SCORE
        )

        vote = self.state.vote(pair_state, rule, condition, ts, window=2.5)
        self.state.start_rule(pair_state, rule, vote >= 0.80, ts)

        confidence = min(
            0.97,
            0.38
            + aggressive * 0.28
            + min(wrist_speed, 0.35) * 0.55
            + max(0, CHAIN_SNATCH_DISTANCE_NORM - distance) * 1.5,
        )

        if (
            self.state.duration(pair_state, rule, ts) >= CHAIN_SNATCH_MIN_SECONDS
            and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.75)
        ):
            alerts.append(
                self.make_pair_alert(
                    rule,
                    pair_key,
                    merge_boxes(b1, b2),
                    confidence,
                    {
                        "distance_norm": round(distance, 3),
                        "wrist_speed": round(wrist_speed, 3),
                        "aggressive_score": round(aggressive, 3),
                    },
                )
            )

        # ---------------- FIGHTING ----------------
        rule = "Fighting"

        condition = (
            distance <= FIGHT_DISTANCE_NORM
            and wrist_speed >= AGGRESSIVE_WRIST_SPEED_NORM
            and aggressive >= 0.45
        )

        vote = self.state.vote(pair_state, rule, condition, ts, window=3.0)
        self.state.start_rule(pair_state, rule, vote >= 0.70, ts)

        confidence = min(
            0.99,
            0.47 + aggressive * 0.30 + min(wrist_speed, 0.35) * 0.40,
        )

        if (
            self.state.duration(pair_state, rule, ts) >= FIGHTING_MIN_SECONDS
            and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.5)
        ):
            alerts.append(
                self.make_pair_alert(
                    rule,
                    pair_key,
                    merge_boxes(b1, b2),
                    confidence,
                    {
                        "distance_norm": round(distance, 3),
                        "wrist_speed": round(wrist_speed, 3),
                        "aggressive_score": round(aggressive, 3),
                    },
                )
            )
        rule = "Possible Abduction Risk"

        condition = (
            distance <= ABDUCTION_DISTANCE_NORM
            and wrist_speed >= ABDUCTION_WRIST_SPEED_NORM
            and aggressive >= ABDUCTION_AGGRESSIVE_SCORE
            and (
                speed1 >= ABDUCTION_MIN_SPEED_NORM
                or speed2 >= ABDUCTION_MIN_SPEED_NORM
            )
        )

        vote = self.state.vote(pair_state, rule, condition, ts, window=3.0)

        self.state.start_rule(
            pair_state,
            rule,
            vote >= 0.70,
            ts,
        )

        confidence = min(
            0.98,
            0.42
            + aggressive * 0.25
            + min(wrist_speed, 0.35) * 0.45
            + max(0, ABDUCTION_DISTANCE_NORM - distance) * 1.4
        )

        if (
            self.state.duration(
                pair_state,
                rule,
                ts,
            ) >= ABDUCTION_CONFIRM_SECONDS
            and confidence >= MIN_ALERT_CONFIDENCE.get(rule, 0.70)
        ):
            alerts.append(
                self.make_pair_alert(
                    rule,
                    pair_key,
                    merge_boxes(b1, b2),
                    confidence,
                    {
                        "distance_norm": round(distance, 3),
                        "wrist_speed": round(wrist_speed, 3),
                        "aggressive_score": round(aggressive, 3),
                        "speed1": round(speed1, 3),
                        "speed2": round(speed2, 3),
                        "reason": "forced movement pattern",
                    },
                )
            )

        # Pair events are also persisted only after their rule-specific
        # confirmation duration has been satisfied.
        for alert in alerts:
            self._persist_confirmed_event(
                cam_id=cam_id,
                track_id=alert.get("track_id"),
                alert=alert,
                ts=ts,
            )

        return alerts

    def make_alert(self, rule, track, box, confidence, evidence):
        return {
            "rule": rule,
            "track": str(track),
            "track_id": str(track),
            "box": box,
            "confidence": round(max(0.0, min(1.0, _finite_float(confidence, 0.0))), 3),
            "evidence": evidence,
        }

    def make_pair_alert(self, rule, pair_key, box, confidence, evidence):
        track_id = f"{pair_key[0]}_{pair_key[1]}"

        return {
            "rule": rule,
            "track": track_id,
            "track_id": track_id,
            "box": box,
            "confidence": round(max(0.0, min(1.0, _finite_float(confidence, 0.0))), 3),
            "evidence": evidence,
        }