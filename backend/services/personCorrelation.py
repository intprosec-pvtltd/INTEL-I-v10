from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("person-correlation")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        value = int(default)
    return max(minimum, min(maximum, value))


def _cosine(a, b) -> float:
    if (
        not isinstance(a, list)
        or not isinstance(b, list)
        or len(a) != len(b)
        or not a
    ):
        return 0.0

    try:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) ** 2 for x in a))
        nb = math.sqrt(sum(float(y) ** 2 for y in b))
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if na <= 0.0 or nb <= 0.0:
        return 0.0

    value = dot / (na * nb)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


@dataclass
class Identity:
    global_person_id: str
    user_id: int
    descriptor: list[float]
    upper_color: str
    lower_color: str
    last_camera_id: str
    last_seen: float
    observations: int = 1


@dataclass(frozen=True)
class SpatialGateResult:
    applicable: bool
    feasible: bool
    distance_m: float | None = None
    required_speed_kmh: float | None = None
    reason: str = "not_applicable"
    source_camera_db_id: int | None = None
    destination_camera_db_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "feasible": self.feasible,
            "distance_m": self.distance_m,
            "distance_km": (
                self.distance_m / 1000.0
                if self.distance_m is not None
                else None
            ),
            "required_speed_kmh": self.required_speed_kmh,
            "reason": self.reason,
            "source_camera_db_id": self.source_camera_db_id,
            "destination_camera_db_id": self.destination_camera_db_id,
        }


class PersonCorrelationEngine:
    """
    INTEL-I cross-camera person correlation engine.

    The existing appearance/color/time/topology scoring remains intact.
    PostGIS is used as a geographic feasibility gate before a cross-camera
    identity can be considered a match.

    Important behavior:
      * Same-camera/local-track continuity never performs a spatial query.
      * Cross-camera candidates may be rejected when the required travel
        distance/speed is physically impossible.
      * Camera identifiers are resolved against cameras.id and cameras.cam_id.
      * Tenant isolation is enforced by cameras.user_id during resolution.
      * PostGIS failure policy is configurable so rollout does not silently
        break existing deployments.
    """

    def __init__(self):
        self.threshold = _env_float(
            "PERSON_CORRELATION_THRESHOLD",
            0.78,
            0.50,
            0.99,
        )
        self.max_gap = _env_float(
            "PERSON_CORRELATION_MAX_GAP_SECONDS",
            1800.0,
            30.0,
            86400.0,
        )
        self.max_identities = _env_int(
            "PERSON_CORRELATION_MAX_IDENTITIES",
            10000,
            100,
            100000,
        )

        # Existing GIS/correlation limits.
        self.gis_enabled = _env_bool(
            "GIS_CORRELATION_ENABLED",
            True,
        )
        self.max_camera_distance_km = _env_float(
            "GIS_MAX_CAMERA_DISTANCE_KM",
            100.0,
            0.001,
            20000.0,
        )
        self.max_reasonable_speed_kmh = _env_float(
            "GIS_MAX_REASONABLE_SPEED_KMH",
            180.0,
            0.001,
            5000.0,
        )
        self.max_transition_seconds = _env_float(
            "GIS_MAX_TRANSITION_SECONDS",
            3600.0,
            0.001,
            86400.0,
        )

        # PostGIS rollout controls.
        self.postgis_enabled = _env_bool(
            "PERSON_CORRELATION_POSTGIS_ENABLED",
            True,
        )
        self.postgis_fail_closed = _env_bool(
            "PERSON_CORRELATION_POSTGIS_FAIL_CLOSED",
            False,
        )
        self.postgis_require_location = _env_bool(
            "PERSON_CORRELATION_POSTGIS_REQUIRE_LOCATION",
            False,
        )

        # Small caches prevent repeated DB lookups for camera identifiers and
        # camera-to-camera distance on every new local track.
        self.camera_cache_ttl = _env_float(
            "PERSON_CORRELATION_CAMERA_CACHE_TTL_SECONDS",
            300.0,
            1.0,
            3600.0,
        )
        self.distance_cache_ttl = _env_float(
            "PERSON_CORRELATION_DISTANCE_CACHE_TTL_SECONDS",
            300.0,
            1.0,
            3600.0,
        )
        self.max_distance_cache_entries = _env_int(
            "PERSON_CORRELATION_DISTANCE_CACHE_MAX_ENTRIES",
            10000,
            100,
            100000,
        )

        self._lock = threading.RLock()
        self._track_map: dict[tuple[int, str, str], str] = {}
        self._identities: dict[str, Identity] = {}

        self._camera_db_id_cache: dict[
            tuple[int, str], tuple[float, int | None]
        ] = {}
        self._distance_cache: dict[
            tuple[int, int], tuple[float, float | None]
        ] = {}

    # ------------------------------------------------------------------
    # Public correlation API
    # ------------------------------------------------------------------

    def correlate(
        self,
        *,
        user_id: int,
        camera_id: str,
        local_track_id: str,
        appearance: dict,
        timestamp: float | None = None,
        allowed_transitions: set[tuple[str, str]] | None = None,
    ) -> dict:
        now = float(timestamp if timestamp is not None else time.time())
        if not math.isfinite(now):
            now = time.time()

        normalized_user_id = int(user_id)
        normalized_camera_id = str(camera_id)
        normalized_track_id = str(local_track_id)

        key = (
            normalized_user_id,
            normalized_camera_id,
            normalized_track_id,
        )

        descriptor = list(
            appearance.get("appearance_descriptor") or []
        )

        with self._lock:
            existing_id = self._track_map.get(key)
            if existing_id and existing_id in self._identities:
                identity = self._identities[existing_id]
                identity.last_seen = now
                identity.observations += 1
                return self._result(
                    identity,
                    1.0,
                    "LOCAL_TRACK_CONTINUITY",
                    None,
                    appearance_score=1.0,
                    color_score=1.0,
                    temporal_score=1.0,
                    topology_score=1.0,
                    spatial=None,
                )

            best: Identity | None = None
            best_score = 0.0
            best_appearance_score = 0.0
            best_color_score = 0.0
            best_time_score = 0.0
            best_topology_score = 0.0
            best_spatial: SpatialGateResult | None = None

            for candidate in self._identities.values():
                if candidate.user_id != normalized_user_id:
                    continue

                elapsed = now - candidate.last_seen
                if elapsed < 0.0 or elapsed > self.max_gap:
                    continue

                appearance_score = _cosine(
                    descriptor,
                    candidate.descriptor,
                )

                cross_camera = (
                    candidate.last_camera_id != normalized_camera_id
                )

                topology_known = bool(allowed_transitions)
                topology_ok = (
                    candidate.last_camera_id,
                    normalized_camera_id,
                ) in (allowed_transitions or set())

                # Preserve the existing explicit topology gate.
                if cross_camera and topology_known and not topology_ok:
                    continue

                spatial: SpatialGateResult | None = None

                if cross_camera and self.gis_enabled and self.postgis_enabled:
                    spatial = self._spatial_gate(
                        user_id=normalized_user_id,
                        source_camera_id=candidate.last_camera_id,
                        destination_camera_id=normalized_camera_id,
                        elapsed_seconds=elapsed,
                    )

                    if spatial.applicable and not spatial.feasible:
                        continue

                    if (
                        not spatial.applicable
                        and self.postgis_fail_closed
                    ):
                        continue

                color_score = (
                    int(
                        str(appearance.get("upper_color"))
                        == candidate.upper_color
                    )
                    + int(
                        str(appearance.get("lower_color"))
                        == candidate.lower_color
                    )
                ) / 2.0

                time_score = max(
                    0.0,
                    1.0 - (elapsed / self.max_gap),
                )

                topology_score = (
                    1.0
                    if not cross_camera or topology_ok
                    else 0.0
                )

                # Preserve the original weighting to avoid changing the
                # correlation model while introducing PostGIS.
                score = min(
                    1.0,
                    0.66 * appearance_score
                    + 0.16 * color_score
                    + 0.10 * time_score
                    + 0.08 * topology_score,
                )

                if score > best_score:
                    best = candidate
                    best_score = score
                    best_appearance_score = appearance_score
                    best_color_score = color_score
                    best_time_score = time_score
                    best_topology_score = topology_score
                    best_spatial = spatial

            candidate_id = (
                best.global_person_id
                if best is not None
                else None
            )

            if best is not None and best_score >= self.threshold:
                identity = best
                matched_cross_camera = (
                    identity.last_camera_id
                    != normalized_camera_id
                )

                identity.descriptor = (
                    descriptor or identity.descriptor
                )
                identity.upper_color = str(
                    appearance.get("upper_color")
                    or identity.upper_color
                )
                identity.lower_color = str(
                    appearance.get("lower_color")
                    or identity.lower_color
                )
                identity.last_camera_id = normalized_camera_id
                identity.last_seen = now
                identity.observations += 1

                has_topology = (
                    not matched_cross_camera
                    or bool(allowed_transitions)
                )

                # PostGIS feasibility is a gate, not an artificial confidence
                # boost. This keeps the historical scoring behavior stable.
                decision = (
                    "CONFIRMED"
                    if best_score >= 0.9 and has_topology
                    else "PROBABLE"
                )
            else:
                global_id = (
                    f"GP-{uuid.uuid4().hex[:20].upper()}"
                )
                identity = Identity(
                    global_person_id=global_id,
                    user_id=normalized_user_id,
                    descriptor=descriptor,
                    upper_color=str(
                        appearance.get("upper_color")
                        or "unknown"
                    ),
                    lower_color=str(
                        appearance.get("lower_color")
                        or "unknown"
                    ),
                    last_camera_id=normalized_camera_id,
                    last_seen=now,
                )

                if len(self._identities) >= self.max_identities:
                    oldest = min(
                        self._identities.values(),
                        key=lambda item: item.last_seen,
                    )
                    self._identities.pop(
                        oldest.global_person_id,
                        None,
                    )
                    self._remove_track_mappings_for_identity(
                        oldest.global_person_id
                    )

                self._identities[global_id] = identity
                best_score = 0.0
                best_appearance_score = 0.0
                best_color_score = 0.0
                best_time_score = 0.0
                best_topology_score = 0.0
                best_spatial = None
                decision = "NEW"

            self._track_map[key] = identity.global_person_id

            return self._result(
                identity,
                best_score,
                decision,
                candidate_id,
                appearance_score=best_appearance_score,
                color_score=best_color_score,
                temporal_score=best_time_score,
                topology_score=best_topology_score,
                spatial=best_spatial,
            )

    # ------------------------------------------------------------------
    # PostGIS geographic gate
    # ------------------------------------------------------------------

    def _spatial_gate(
        self,
        *,
        user_id: int,
        source_camera_id: str,
        destination_camera_id: str,
        elapsed_seconds: float,
    ) -> SpatialGateResult:
        if not self.gis_enabled or not self.postgis_enabled:
            return SpatialGateResult(
                applicable=False,
                feasible=True,
                reason="postgis_disabled",
            )

        if elapsed_seconds < 0.0:
            return SpatialGateResult(
                applicable=True,
                feasible=False,
                reason="destination_precedes_source",
            )

        if elapsed_seconds > self.max_transition_seconds:
            return SpatialGateResult(
                applicable=True,
                feasible=False,
                reason="transition_time_exceeded",
            )

        try:
            source_db_id = self._resolve_camera_db_id(
                user_id=user_id,
                camera_identifier=source_camera_id,
            )
            destination_db_id = self._resolve_camera_db_id(
                user_id=user_id,
                camera_identifier=destination_camera_id,
            )

            if source_db_id is None or destination_db_id is None:
                return SpatialGateResult(
                    applicable=False,
                    feasible=not self.postgis_require_location,
                    reason="camera_not_resolved",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            distance_m = self._camera_distance_m(
                source_db_id,
                destination_db_id,
            )

            if distance_m is None:
                return SpatialGateResult(
                    applicable=False,
                    feasible=not self.postgis_require_location,
                    reason="camera_location_unavailable",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            distance_km = distance_m / 1000.0

            if distance_km > self.max_camera_distance_km:
                return SpatialGateResult(
                    applicable=True,
                    feasible=False,
                    distance_m=distance_m,
                    required_speed_kmh=None,
                    reason="camera_distance_exceeded",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            if elapsed_seconds == 0.0:
                if distance_m <= 0.001:
                    required_speed = 0.0
                else:
                    return SpatialGateResult(
                        applicable=True,
                        feasible=False,
                        distance_m=distance_m,
                        required_speed_kmh=None,
                        reason="zero_time_nonzero_distance",
                        source_camera_db_id=source_db_id,
                        destination_camera_db_id=destination_db_id,
                    )
            else:
                required_speed = (
                    distance_km
                    / (elapsed_seconds / 3600.0)
                )

            if required_speed > self.max_reasonable_speed_kmh:
                return SpatialGateResult(
                    applicable=True,
                    feasible=False,
                    distance_m=distance_m,
                    required_speed_kmh=required_speed,
                    reason="required_speed_exceeded",
                    source_camera_db_id=source_db_id,
                    destination_camera_db_id=destination_db_id,
                )

            return SpatialGateResult(
                applicable=True,
                feasible=True,
                distance_m=distance_m,
                required_speed_kmh=required_speed,
                reason="feasible",
                source_camera_db_id=source_db_id,
                destination_camera_db_id=destination_db_id,
            )

        except Exception as exc:
            # Fail-open is the default rollout mode to preserve existing
            # correlation behavior. Production deployments that have complete
            # camera GIS coverage can set
            # PERSON_CORRELATION_POSTGIS_FAIL_CLOSED=true.
            logger.exception(
                "PostGIS person-correlation gate failed for user=%s "
                "source_camera=%s destination_camera=%s",
                user_id,
                source_camera_id,
                destination_camera_id,
            )
            return SpatialGateResult(
                applicable=False,
                feasible=not self.postgis_fail_closed,
                reason=(
                    "postgis_error_fail_closed"
                    if self.postgis_fail_closed
                    else "postgis_error_fail_open"
                ),
            )

    def _resolve_camera_db_id(
        self,
        *,
        user_id: int,
        camera_identifier: str,
    ) -> int | None:
        identifier = str(camera_identifier or "").strip()
        if not identifier:
            return None

        cache_key = (int(user_id), identifier)
        now = time.monotonic()

        cached = self._camera_db_id_cache.get(cache_key)
        if cached is not None:
            expires_at, cached_id = cached
            if expires_at > now:
                return cached_id
            self._camera_db_id_cache.pop(cache_key, None)

        from sqlalchemy import text
        from db.database import SessionLocal

        db = SessionLocal()
        try:
            row = db.execute(
                text(
                    """
                    SELECT id
                    FROM public.cameras
                    WHERE user_id = :user_id
                      AND (
                            cam_id = :camera_identifier
                            OR CAST(id AS TEXT) = :camera_identifier
                      )
                    ORDER BY
                        CASE
                            WHEN cam_id = :camera_identifier THEN 0
                            ELSE 1
                        END,
                        id ASC
                    LIMIT 1
                    """
                ),
                {
                    "user_id": int(user_id),
                    "camera_identifier": identifier,
                },
            ).first()

            resolved_id = (
                int(row[0])
                if row is not None and row[0] is not None
                else None
            )

            self._camera_db_id_cache[cache_key] = (
                now + self.camera_cache_ttl,
                resolved_id,
            )
            return resolved_id
        finally:
            db.close()

    def _camera_distance_m(
        self,
        source_camera_db_id: int,
        destination_camera_db_id: int,
    ) -> float | None:
        source_id = int(source_camera_db_id)
        destination_id = int(destination_camera_db_id)

        if source_id == destination_id:
            return 0.0

        pair = (
            min(source_id, destination_id),
            max(source_id, destination_id),
        )
        now = time.monotonic()

        cached = self._distance_cache.get(pair)
        if cached is not None:
            expires_at, cached_distance = cached
            if expires_at > now:
                return cached_distance
            self._distance_cache.pop(pair, None)

        from db.database import SessionLocal
        from services.postgisSpatial import PostGISSpatialService

        db = SessionLocal()
        try:
            spatial = PostGISSpatialService(db)
            distance = spatial.camera_distance(
                source_id,
                destination_id,
            )
            distance_m = (
                float(distance.distance_m)
                if distance is not None
                else None
            )
        finally:
            db.close()

        if len(self._distance_cache) >= self.max_distance_cache_entries:
            # Drop the oldest-expiring entries first. This cache is only an
            # optimization; removing entries never changes correlation logic.
            stale_keys = sorted(
                self._distance_cache,
                key=lambda key: self._distance_cache[key][0],
            )[: max(1, self.max_distance_cache_entries // 10)]
            for key in stale_keys:
                self._distance_cache.pop(key, None)

        self._distance_cache[pair] = (
            now + self.distance_cache_ttl,
            distance_m,
        )
        return distance_m

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(
        self,
        *,
        result: dict,
        user_id: int,
        camera_id: str,
        local_track_id: str,
        confidence: float,
        timestamp,
        bbox,
        pose,
        appearance: dict,
    ):
        from db.database import SessionLocal
        from db.intelligence_model import (
            Person,
            PersonCorrelationDecision,
            PersonObservation,
        )

        db = SessionLocal()

        try:
            gid = str(result["global_person_id"])

            person = (
                db.query(Person)
                .filter(
                    Person.user_id == int(user_id),
                    Person.global_person_id == gid,
                )
                .first()
            )

            if person is None:
                person = Person(
                    user_id=int(user_id),
                    global_person_id=gid,
                    first_seen_at=timestamp,
                    last_seen_at=timestamp,
                    observation_count=0,
                    metadata_json={
                        "identity_type": "appearance_correlation"
                    },
                )
                db.add(person)
                db.flush()

            person.last_seen_at = timestamp
            person.observation_count = (
                int(person.observation_count or 0) + 1
            )

            spatial_evidence = result.get("spatial")
            if not isinstance(spatial_evidence, dict):
                spatial_evidence = None

            person.metadata_json = {
                **(
                    person.metadata_json
                    if isinstance(person.metadata_json, dict)
                    else {}
                ),
                "confidence_status": result.get(
                    "confidence_status"
                ),
                "last_correlation_score": float(
                    result.get("score", 0.0)
                ),
                "last_camera_id": str(camera_id),
                "last_spatial": spatial_evidence,
            }

            obs = PersonObservation(
                user_id=int(user_id),
                person_id=person.id,
                global_person_id=gid,
                camera_id=str(camera_id),
                local_track_id=str(local_track_id),
                confidence=float(confidence),
                frame_timestamp=timestamp,
                bbox={"xyxy": list(bbox)},
                pose=pose,
                model_name="person_appearance",
                model_version="1",
                metadata_json={
                    **{
                        k: v
                        for k, v in appearance.items()
                        if k != "appearance_descriptor"
                    },
                    "correlation": {
                        "decision": result.get("decision"),
                        "final_score": float(
                            result.get("score", 0.0)
                        ),
                        "spatial": spatial_evidence,
                    },
                },
            )
            db.add(obs)

            decision_evidence = {
                "upper_color": appearance.get("upper_color"),
                "lower_color": appearance.get("lower_color"),
                "color_score": result.get("color_score"),
                "spatial": spatial_evidence,
            }

            db.add(
                PersonCorrelationDecision(
                    user_id=int(user_id),
                    camera_id=str(camera_id),
                    local_track_id=str(local_track_id),
                    global_person_id=gid,
                    candidate_global_person_id=result.get(
                        "candidate_global_person_id"
                    ),
                    appearance_score=float(
                        result.get(
                            "appearance_score",
                            result.get("score", 0.0),
                        )
                    ),
                    temporal_score=self._optional_float(
                        result.get("temporal_score")
                    ),
                    topology_score=self._optional_float(
                        result.get("topology_score")
                    ),
                    final_score=float(
                        result.get("score", 0.0)
                    ),
                    decision=str(result.get("decision")),
                    evidence=decision_evidence,
                )
            )

            db.commit()
            db.refresh(obs)
            return obs

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset_camera(self, camera_id: str) -> None:
        """Forget only camera-local track mappings after a scene cut/loop."""
        normalized = str(camera_id or "").strip()
        if not normalized:
            return

        with self._lock:
            for key in list(self._track_map):
                if key[1] == normalized:
                    self._track_map.pop(key, None)

    def clear_spatial_cache(self) -> None:
        with self._lock:
            self._camera_db_id_cache.clear()
            self._distance_cache.clear()

    def _remove_track_mappings_for_identity(
        self,
        global_person_id: str,
    ) -> None:
        for key, mapped_id in list(self._track_map.items()):
            if mapped_id == global_person_id:
                self._track_map.pop(key, None)

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _result(
        identity: Identity,
        score: float,
        decision: str,
        candidate_id: str | None,
        *,
        appearance_score: float | None = None,
        color_score: float | None = None,
        temporal_score: float | None = None,
        topology_score: float | None = None,
        spatial: SpatialGateResult | None = None,
    ) -> dict:
        return {
            "global_person_id": identity.global_person_id,
            "score": round(float(score), 4),
            "decision": decision,
            "candidate_global_person_id": candidate_id,
            "confidence_status": decision,
            "appearance_score": (
                round(float(appearance_score), 4)
                if appearance_score is not None
                else None
            ),
            "color_score": (
                round(float(color_score), 4)
                if color_score is not None
                else None
            ),
            "temporal_score": (
                round(float(temporal_score), 4)
                if temporal_score is not None
                else None
            ),
            "topology_score": (
                round(float(topology_score), 4)
                if topology_score is not None
                else None
            ),
            "spatial": (
                spatial.as_dict()
                if spatial is not None
                else None
            ),
        }


person_correlation_engine = PersonCorrelationEngine()