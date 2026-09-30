from __future__ import annotations

"""
Production validation for INTEL-I person correlation + PostGIS.

This validator is intentionally NON-DESTRUCTIVE:
- it reads camera/PostGIS data;
- it exercises PersonCorrelationEngine.correlate() in memory;
- it does NOT call persist();
- it does NOT change cameras, persons, observations, or decisions.

Validation:
1. PostGIS readiness.
2. Select three real geolocated cameras belonging to the same user.
3. Verify A -> B is geographically feasible.
4. Verify the same appearance correlates A -> B to the same GlobalPersonID.
5. Verify B -> C is geographically impossible at a deliberately tiny elapsed time.
6. Verify PostGIS rejects the old identity and creates a new GlobalPersonID.
7. Verify same-camera/local-track continuity remains intact.
"""

import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env", override=True)

from db.database import SessionLocal  # noqa: E402
from services.personCorrelation import PersonCorrelationEngine  # noqa: E402
from services.postgisSpatial import PostGISSpatialService  # noqa: E402


@dataclass(frozen=True)
class CameraRow:
    id: int
    user_id: int
    cam_id: str
    camera_name: str
    latitude: float
    longitude: float


def fail(message: str) -> None:
    raise AssertionError(message)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"{label:<34}: {status}{suffix}")
    if not condition:
        fail(f"{label} failed{suffix}")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_geolocated_cameras(db) -> list[CameraRow]:
    rows = db.execute(
        text(
            """
            SELECT
                id,
                user_id,
                cam_id,
                camera_name,
                latitude,
                longitude
            FROM public.cameras
            WHERE location IS NOT NULL
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
            ORDER BY user_id, id
            """
        )
    ).mappings().all()

    cameras: list[CameraRow] = []
    for row in rows:
        lat = finite_float(row["latitude"])
        lon = finite_float(row["longitude"])
        if lat is None or lon is None:
            continue
        cameras.append(
            CameraRow(
                id=int(row["id"]),
                user_id=int(row["user_id"]),
                cam_id=str(row["cam_id"]),
                camera_name=str(row["camera_name"] or ""),
                latitude=lat,
                longitude=lon,
            )
        )
    return cameras


def distance_m(spatial: PostGISSpatialService, a: CameraRow, b: CameraRow) -> float:
    result = spatial.camera_distance(a.id, b.id)
    if result is None:
        fail(f"No PostGIS distance returned for cameras {a.id} and {b.id}.")
    value = finite_float(result.distance_m)
    if value is None:
        fail(f"Invalid PostGIS distance returned for cameras {a.id} and {b.id}.")
    return value


def choose_three_cameras(
    spatial: PostGISSpatialService,
    cameras: list[CameraRow],
    engine: PersonCorrelationEngine,
) -> tuple[CameraRow, CameraRow, CameraRow, float, float]:
    """
    Select A/B/C from one tenant.

    A -> B must:
      - have non-zero distance;
      - be within GIS_MAX_CAMERA_DISTANCE_KM;
      - be feasible within the correlation max gap at a conservative speed.

    B -> C must:
      - have non-zero distance;
      - be within GIS_MAX_CAMERA_DISTANCE_KM.

    The impossible B -> C case is then created using a deliberately tiny
    elapsed time so required speed exceeds GIS_MAX_REASONABLE_SPEED_KMH.
    """
    by_user: dict[int, list[CameraRow]] = {}
    for camera in cameras:
        by_user.setdefault(camera.user_id, []).append(camera)

    best: tuple[
        float,
        CameraRow,
        CameraRow,
        CameraRow,
        float,
        float,
    ] | None = None

    for user_id, group in by_user.items():
        if len(group) < 3:
            continue

        for a in group:
            for b in group:
                if b.id == a.id:
                    continue

                ab = distance_m(spatial, a, b)
                if ab <= 1.0:
                    continue
                if ab / 1000.0 > engine.max_camera_distance_km:
                    continue

                # Prefer a close A/B pair for the valid transition.
                valid_elapsed = max(
                    60.0,
                    (ab / 1000.0)
                    / max(1.0, engine.max_reasonable_speed_kmh * 0.50)
                    * 3600.0,
                )
                if valid_elapsed >= min(engine.max_gap, engine.max_transition_seconds):
                    continue

                for c in group:
                    if c.id in {a.id, b.id}:
                        continue

                    bc = distance_m(spatial, b, c)
                    if bc <= 1.0:
                        continue
                    if bc / 1000.0 > engine.max_camera_distance_km:
                        continue

                    # Prefer a more distant C so the impossible test is obvious.
                    rank = bc - ab
                    candidate = (rank, a, b, c, ab, bc)
                    if best is None or candidate[0] > best[0]:
                        best = candidate

    if best is None:
        fail(
            "Could not select three suitable same-user geolocated cameras. "
            "Need at least three distinct locations within configured GIS distance limits."
        )

    _, a, b, c, ab, bc = best
    return a, b, c, ab, bc


def appearance_fixture() -> dict[str, Any]:
    # Deterministic normalized-ish descriptor. The exact same descriptor is
    # intentionally supplied at all three cameras so only geography can reject
    # the impossible transition.
    descriptor = [
        0.314159,
        0.271828,
        0.161803,
        0.577215,
        0.141421,
        0.173205,
        0.223607,
        0.244949,
        0.264575,
        0.282843,
        0.316228,
        0.331662,
        0.346410,
        0.360555,
        0.374166,
        0.387298,
    ]
    return {
        "appearance_descriptor": descriptor,
        "upper_color": "validation-blue",
        "lower_color": "validation-black",
    }


def main() -> int:
    banner("INTEL-I — 3-CAMERA POSTGIS PERSON CORRELATION VALIDATION")

    db = SessionLocal()
    try:
        spatial = PostGISSpatialService(db)
        readiness = spatial.readiness()

        print("PostGIS readiness:", readiness)
        check(
            "POSTGIS READINESS",
            bool(readiness.available),
            str(readiness.postgis_version or ""),
        )
        check("LOCATION COLUMN", bool(readiness.location_column))
        check("SPATIAL INDEX", bool(readiness.spatial_index))
        check("SYNC TRIGGER", bool(readiness.sync_trigger))

        cameras = load_geolocated_cameras(db)
        check(
            "GEOLOCATED CAMERA COUNT",
            len(cameras) >= 3,
            f"{len(cameras)} cameras",
        )

        engine = PersonCorrelationEngine()

        # This test specifically validates the PostGIS gate. Do not silently
        # pass if the feature has been disabled in the environment.
        check(
            "GIS CORRELATION ENABLED",
            bool(engine.gis_enabled),
            f"GIS_CORRELATION_ENABLED={engine.gis_enabled}",
        )
        check(
            "PERSON POSTGIS ENABLED",
            bool(engine.postgis_enabled),
            f"PERSON_CORRELATION_POSTGIS_ENABLED={engine.postgis_enabled}",
        )

        a, b, c, ab_m, bc_m = choose_three_cameras(
            spatial,
            cameras,
            engine,
        )

    finally:
        db.close()

    banner("SELECTED REAL DATABASE CAMERAS")
    print(
        f"A: id={a.id} user={a.user_id} cam_id={a.cam_id} "
        f"name={a.camera_name!r} lat={a.latitude} lon={a.longitude}"
    )
    print(
        f"B: id={b.id} user={b.user_id} cam_id={b.cam_id} "
        f"name={b.camera_name!r} lat={b.latitude} lon={b.longitude}"
    )
    print(
        f"C: id={c.id} user={c.user_id} cam_id={c.cam_id} "
        f"name={c.camera_name!r} lat={c.latitude} lon={c.longitude}"
    )
    print(f"A -> B distance: {ab_m:.2f} m ({ab_m / 1000.0:.4f} km)")
    print(f"B -> C distance: {bc_m:.2f} m ({bc_m / 1000.0:.4f} km)")

    # ------------------------------------------------------------------
    # Build transition timings from the real distances/configuration.
    # ------------------------------------------------------------------

    valid_elapsed = max(
        60.0,
        (ab_m / 1000.0)
        / max(1.0, engine.max_reasonable_speed_kmh * 0.50)
        * 3600.0,
    )

    valid_elapsed = min(
        valid_elapsed,
        engine.max_gap * 0.80,
        engine.max_transition_seconds * 0.80,
    )

    if valid_elapsed <= 0.0:
        fail("Could not derive a valid A -> B elapsed time.")

    valid_speed = (ab_m / 1000.0) / (valid_elapsed / 3600.0)

    # Pick an elapsed time that requires at least 2x the configured maximum
    # reasonable speed. Keep it > 0 so we test the speed gate itself.
    impossible_elapsed = (
        (bc_m / 1000.0)
        / max(1.0, engine.max_reasonable_speed_kmh * 2.0)
        * 3600.0
    )
    impossible_elapsed = max(0.001, impossible_elapsed)

    impossible_speed = (
        (bc_m / 1000.0)
        / (impossible_elapsed / 3600.0)
    )

    banner("SPATIAL FEASIBILITY")
    print(
        f"Configured max reasonable speed : "
        f"{engine.max_reasonable_speed_kmh:.2f} km/h"
    )
    print(
        f"Configured max camera distance   : "
        f"{engine.max_camera_distance_km:.2f} km"
    )
    print(
        f"Configured max transition time   : "
        f"{engine.max_transition_seconds:.2f} s"
    )
    print(
        f"A -> B elapsed / required speed  : "
        f"{valid_elapsed:.3f} s / {valid_speed:.3f} km/h"
    )
    print(
        f"B -> C elapsed / required speed  : "
        f"{impossible_elapsed:.6f} s / {impossible_speed:.3f} km/h"
    )

    gate_ab = engine._spatial_gate(
        user_id=a.user_id,
        source_camera_id=a.cam_id,
        destination_camera_id=b.cam_id,
        elapsed_seconds=valid_elapsed,
    )
    gate_bc = engine._spatial_gate(
        user_id=b.user_id,
        source_camera_id=b.cam_id,
        destination_camera_id=c.cam_id,
        elapsed_seconds=impossible_elapsed,
    )

    print("A -> B gate:", gate_ab.as_dict())
    print("B -> C gate:", gate_bc.as_dict())

    check(
        "A -> B GATE APPLICABLE",
        gate_ab.applicable,
        gate_ab.reason,
    )
    check(
        "A -> B FEASIBLE",
        gate_ab.feasible,
        gate_ab.reason,
    )
    check(
        "B -> C GATE APPLICABLE",
        gate_bc.applicable,
        gate_bc.reason,
    )
    check(
        "B -> C IMPOSSIBLE REJECTED",
        not gate_bc.feasible,
        gate_bc.reason,
    )
    check(
        "B -> C SPEED EXCEEDS LIMIT",
        (
            gate_bc.required_speed_kmh is not None
            and gate_bc.required_speed_kmh
            > engine.max_reasonable_speed_kmh
        ),
        (
            f"required={gate_bc.required_speed_kmh:.3f} km/h"
            if gate_bc.required_speed_kmh is not None
            else "required speed unavailable"
        ),
    )

    # ------------------------------------------------------------------
    # Actual in-memory correlation pipeline.
    # ------------------------------------------------------------------

    banner("PERSON CORRELATION PIPELINE")

    appearance = appearance_fixture()
    base_time = time.time()

    result_a = engine.correlate(
        user_id=a.user_id,
        camera_id=a.cam_id,
        local_track_id="POSTGIS-VALIDATION-A",
        appearance=appearance,
        timestamp=base_time,
        allowed_transitions=None,
    )

    result_b = engine.correlate(
        user_id=b.user_id,
        camera_id=b.cam_id,
        local_track_id="POSTGIS-VALIDATION-B",
        appearance=appearance,
        timestamp=base_time + valid_elapsed,
        allowed_transitions=None,
    )

    result_c = engine.correlate(
        user_id=c.user_id,
        camera_id=c.cam_id,
        local_track_id="POSTGIS-VALIDATION-C",
        appearance=appearance,
        timestamp=base_time + valid_elapsed + impossible_elapsed,
        allowed_transitions=None,
    )

    print("CAMERA A RESULT:", result_a)
    print("CAMERA B RESULT:", result_b)
    print("CAMERA C RESULT:", result_c)

    gid_a = result_a["global_person_id"]
    gid_b = result_b["global_person_id"]
    gid_c = result_c["global_person_id"]

    check(
        "CAMERA A CREATES IDENTITY",
        result_a["decision"] == "NEW",
        f"decision={result_a['decision']} gid={gid_a}",
    )

    check(
        "A -> B SAME GLOBAL PERSON",
        gid_b == gid_a,
        f"A={gid_a} B={gid_b}",
    )

    check(
        "A -> B MATCH DECISION",
        result_b["decision"] in {"PROBABLE", "CONFIRMED"},
        f"decision={result_b['decision']} score={result_b['score']}",
    )

    spatial_b = result_b.get("spatial") or {}
    check(
        "A -> B POSTGIS EVIDENCE",
        (
            spatial_b.get("applicable") is True
            and spatial_b.get("feasible") is True
        ),
        str(spatial_b),
    )

    # Since the descriptor/color are intentionally identical, C would match
    # the previous identity without the geographic gate. A different GID here
    # demonstrates that PostGIS rejected that candidate.
    check(
        "B -> C GLOBAL ID ISOLATION",
        gid_c != gid_b,
        f"B={gid_b} C={gid_c}",
    )

    check(
        "CAMERA C BECOMES NEW",
        result_c["decision"] == "NEW",
        f"decision={result_c['decision']} gid={gid_c}",
    )

    # A rejected candidate is not returned as the selected candidate because
    # it is removed before scoring. The direct gate checks above prove why.
    check(
        "IMPOSSIBLE CANDIDATE REMOVED",
        result_c.get("candidate_global_person_id") is None,
        f"candidate={result_c.get('candidate_global_person_id')}",
    )

    # ------------------------------------------------------------------
    # Same-camera/local-track continuity must remain unaffected.
    # ------------------------------------------------------------------

    continuity_time = (
        base_time
        + valid_elapsed
        + impossible_elapsed
        + 1.0
    )

    result_c_continuity = engine.correlate(
        user_id=c.user_id,
        camera_id=c.cam_id,
        local_track_id="POSTGIS-VALIDATION-C",
        appearance=appearance,
        timestamp=continuity_time,
        allowed_transitions=None,
    )

    print("CAMERA C CONTINUITY:", result_c_continuity)

    check(
        "LOCAL TRACK SAME GLOBAL ID",
        result_c_continuity["global_person_id"] == gid_c,
        (
            f"before={gid_c} "
            f"after={result_c_continuity['global_person_id']}"
        ),
    )

    check(
        "LOCAL TRACK CONTINUITY",
        result_c_continuity["decision"]
        == "LOCAL_TRACK_CONTINUITY",
        f"decision={result_c_continuity['decision']}",
    )

    check(
        "LOCAL TRACK SCORE",
        float(result_c_continuity["score"]) == 1.0,
        f"score={result_c_continuity['score']}",
    )

    banner("FINAL RESULT")
    print("FINAL RESULT                       : PASSED")
    print()
    print(
        "PostGIS accepted the physically feasible A -> B transition, "
        "rejected the geographically impossible B -> C transition, "
        "prevented the identical appearance descriptor from reusing the "
        "previous GlobalPersonID, and preserved local-track continuity."
    )
    print()
    print("Database writes performed          : NONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        banner("FINAL RESULT")
        print("FINAL RESULT                       : FAILED")
        print(f"REASON                             : {exc}")
        raise SystemExit(1)
    except Exception:
        banner("FINAL RESULT")
        print("FINAL RESULT                       : ERROR")
        traceback.print_exc()
        raise SystemExit(2)
