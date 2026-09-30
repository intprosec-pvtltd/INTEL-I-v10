from __future__ import annotations

"""
Production validation for INTEL-I vehicle correlation + PostGIS.

NON-DESTRUCTIVE:
- reads cameras/PostGIS data only;
- exercises the production vehicleCorrelation engine in memory;
- does not persist correlation decisions;
- does not modify cameras, vehicles, observations, or journeys.

Validation:
1. PostGIS readiness.
2. Select three real, distinct, same-user geolocated cameras.
3. Verify A -> B is physically feasible.
4. Feed identical strong ANPR + Re-ID evidence at A and B and require the same GlobalVehicleID.
5. Make B -> C physically impossible while keeping the identity evidence identical.
6. Verify PostGIS prevents reuse of the old GlobalVehicleID.
7. Verify same-camera/local-track continuity remains intact.
"""

import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env", override=True)

from db.database import SessionLocal  # noqa: E402
from services.postgisSpatial import PostGISSpatialService  # noqa: E402

import vehicleCorrelation as vc  # noqa: E402


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
    print(f"{label:<38}: {status}{suffix}")
    if not condition:
        fail(f"{label} failed{suffix}")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
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
              AND cam_id IS NOT NULL
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


def distance_m(
    spatial: PostGISSpatialService,
    a: CameraRow,
    b: CameraRow,
) -> float:
    result = spatial.camera_distance(a.id, b.id)
    if result is None:
        fail(f"No PostGIS distance for cameras {a.id} and {b.id}.")
    value = finite_float(result.distance_m)
    if value is None:
        fail(f"Invalid PostGIS distance for cameras {a.id} and {b.id}.")
    return value


def choose_three_cameras(
    spatial: PostGISSpatialService,
    cameras: list[CameraRow],
) -> tuple[CameraRow, CameraRow, CameraRow, float, float]:
    by_user: dict[int, list[CameraRow]] = {}
    for camera in cameras:
        by_user.setdefault(camera.user_id, []).append(camera)

    best = None

    for _, group in by_user.items():
        if len(group) < 3:
            continue

        for a in group:
            for b in group:
                if b.id == a.id:
                    continue

                ab = distance_m(spatial, a, b)
                if ab <= 1.0:
                    continue
                if ab / 1000.0 > vc.POSTGIS_VEHICLE_GATE_MAX_DISTANCE_KM:
                    continue

                valid_elapsed = max(
                    60.0,
                    (ab / 1000.0)
                    / max(
                        1.0,
                        vc.POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH * 0.50,
                    )
                    * 3600.0,
                )

                if valid_elapsed >= min(
                    vc.MAX_SIGHTING_GAP_SECONDS,
                    vc.POSTGIS_VEHICLE_GATE_MAX_TRANSITION_SECONDS,
                ):
                    continue

                for c in group:
                    if c.id in {a.id, b.id}:
                        continue

                    bc = distance_m(spatial, b, c)
                    if bc <= 1.0:
                        continue
                    if bc / 1000.0 > vc.POSTGIS_VEHICLE_GATE_MAX_DISTANCE_KM:
                        continue

                    rank = bc - ab
                    candidate = (rank, a, b, c, ab, bc)
                    if best is None or candidate[0] > best[0]:
                        best = candidate

    if best is None:
        fail(
            "Could not select three suitable same-user geolocated cameras. "
            "Need three distinct locations within configured PostGIS limits."
        )

    _, a, b, c, ab, bc = best
    return a, b, c, ab, bc


def embedding_fixture() -> list[float]:
    # Deterministic 512-D unit vector. Exactly the same appearance evidence is
    # used at A/B/C so geography is the only reason B -> C may be rejected.
    rng = np.random.default_rng(20260904)
    vector = rng.normal(0.0, 1.0, 512).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector.tolist()


def vehicle_fixture(
    *,
    camera: CameraRow,
    track_id: str,
    timestamp: float,
    embedding: list[float],
) -> dict[str, Any]:
    return {
        "camera_id": camera.cam_id,
        "track_id": track_id,
        "timestamp": timestamp,
        "vehicle_type": "car",
        "embedding": embedding,
        "plate": "GJ01AB1234",
        "stable_plate": "GJ01AB1234",
        "plate_status": "STABLE",
        "plate_confidence": 0.99,
        "stable_plate_confidence": 0.99,
        "track_quality": 0.95,
        "track_quality_score": 0.95,
        "embedding_quality": 0.95,
        "embedding_quality_score": 0.95,
    }


def correlate_observation(
    vehicle: dict[str, Any],
    embedding: list[float],
) -> dict[str, Any]:
    """
    Use the production public correlation entry point.

    correlate_vehicle accepts the observation dict plus embedding in current
    INTEL-I. Keep this tiny adapter in one place so a signature mismatch is
    immediately obvious instead of silently bypassing production code.
    """
    return vc.correlate_vehicle(
        vehicle,
        embedding,
    )


def reset_in_memory_state() -> None:
    # The validator is intentionally isolated from any earlier process state.
    if hasattr(vc, "clear_vehicle_correlation_state"):
        vc.clear_vehicle_correlation_state()
        return

    # Fallback for current production module internals.
    with vc._correlation_lock:
        vc._global_vehicles.clear()
        vc._local_to_global.clear()
        vc._local_track_evidence.clear()


def main() -> int:
    banner("INTEL-I — 3-CAMERA POSTGIS VEHICLE CORRELATION VALIDATION")

    check(
        "VEHICLE POSTGIS GATE ENABLED",
        bool(vc.POSTGIS_VEHICLE_GATE_ENABLED),
        f"POSTGIS_VEHICLE_GATE_ENABLED={vc.POSTGIS_VEHICLE_GATE_ENABLED}",
    )

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

        a, b, c, ab_m, bc_m = choose_three_cameras(
            spatial,
            cameras,
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

    valid_elapsed = max(
        60.0,
        (ab_m / 1000.0)
        / max(
            1.0,
            vc.POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH * 0.50,
        )
        * 3600.0,
    )

    valid_elapsed = min(
        valid_elapsed,
        vc.MAX_SIGHTING_GAP_SECONDS * 0.80,
        vc.POSTGIS_VEHICLE_GATE_MAX_TRANSITION_SECONDS * 0.80,
    )

    if valid_elapsed <= 0.0:
        fail("Could not derive a valid A -> B elapsed time.")

    valid_speed = (
        (ab_m / 1000.0)
        / (valid_elapsed / 3600.0)
    )

    # Require exactly 2x the configured maximum speed for B -> C.
    impossible_elapsed = (
        (bc_m / 1000.0)
        / max(
            1.0,
            vc.POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH * 2.0,
        )
        * 3600.0
    )
    impossible_elapsed = max(0.001, impossible_elapsed)

    impossible_speed = (
        (bc_m / 1000.0)
        / (impossible_elapsed / 3600.0)
    )

    banner("DIRECT POSTGIS CANDIDATE GATE")

    base_time = time.time()

    synthetic_a_state = {
        "last_camera_id": a.cam_id,
        "last_seen": base_time,
    }
    synthetic_b_state = {
        "last_camera_id": b.cam_id,
        "last_seen": base_time + valid_elapsed,
    }

    gate_ab = vc._postgis_candidate_gate(
        {
            "camera_id": b.cam_id,
            "timestamp": base_time + valid_elapsed,
        },
        synthetic_a_state,
    )

    gate_bc = vc._postgis_candidate_gate(
        {
            "camera_id": c.cam_id,
            "timestamp": (
                base_time
                + valid_elapsed
                + impossible_elapsed
            ),
        },
        synthetic_b_state,
    )

    print("A -> B gate:", gate_ab)
    print("B -> C gate:", gate_bc)
    print(
        f"Configured max speed             : "
        f"{vc.POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH:.2f} km/h"
    )
    print(
        f"Configured max camera distance   : "
        f"{vc.POSTGIS_VEHICLE_GATE_MAX_DISTANCE_KM:.2f} km"
    )
    print(
        f"Configured max transition time   : "
        f"{vc.POSTGIS_VEHICLE_GATE_MAX_TRANSITION_SECONDS:.2f} s"
    )
    print(
        f"A -> B elapsed / required speed  : "
        f"{valid_elapsed:.3f} s / {valid_speed:.3f} km/h"
    )
    print(
        f"B -> C elapsed / required speed  : "
        f"{impossible_elapsed:.6f} s / {impossible_speed:.3f} km/h"
    )

    check(
        "A -> B POSTGIS AVAILABLE",
        gate_ab.get("available") is True,
        str(gate_ab),
    )
    check(
        "A -> B POSTGIS FEASIBLE",
        (
            gate_ab.get("feasible") is True
            and gate_ab.get("hard_reject") is False
        ),
        str(gate_ab),
    )
    check(
        "B -> C POSTGIS AVAILABLE",
        gate_bc.get("available") is True,
        str(gate_bc),
    )
    check(
        "B -> C POSTGIS HARD REJECT",
        (
            gate_bc.get("feasible") is False
            and gate_bc.get("hard_reject") is True
        ),
        str(gate_bc),
    )
    check(
        "B -> C SPEED EXCEEDS LIMIT",
        (
            finite_float(
                gate_bc.get("required_speed_kmh")
            ) is not None
            and float(
                gate_bc["required_speed_kmh"]
            )
            > vc.POSTGIS_VEHICLE_GATE_MAX_SPEED_KMH
        ),
        str(gate_bc.get("required_speed_kmh")),
    )

    banner("PRODUCTION VEHICLE CORRELATION PIPELINE")

    reset_in_memory_state()
    embedding = embedding_fixture()

    vehicle_a = vehicle_fixture(
        camera=a,
        track_id="POSTGIS-VEHICLE-A",
        timestamp=base_time,
        embedding=embedding,
    )
    vehicle_b = vehicle_fixture(
        camera=b,
        track_id="POSTGIS-VEHICLE-B",
        timestamp=base_time + valid_elapsed,
        embedding=embedding,
    )
    vehicle_c = vehicle_fixture(
        camera=c,
        track_id="POSTGIS-VEHICLE-C",
        timestamp=(
            base_time
            + valid_elapsed
            + impossible_elapsed
        ),
        embedding=embedding,
    )

    result_a = correlate_observation(
        vehicle_a,
        embedding,
    )
    result_b = correlate_observation(
        vehicle_b,
        embedding,
    )
    result_c = correlate_observation(
        vehicle_c,
        embedding,
    )

    print("CAMERA A RESULT:", result_a)
    print("CAMERA B RESULT:", result_b)
    print("CAMERA C RESULT:", result_c)

    gid_a = result_a.get("global_vehicle_id")
    gid_b = result_b.get("global_vehicle_id")
    gid_c = result_c.get("global_vehicle_id")

    check(
        "CAMERA A CREATES GLOBAL ID",
        bool(gid_a),
        f"gid={gid_a}",
    )

    check(
        "A -> B SAME GLOBAL VEHICLE",
        gid_b == gid_a,
        f"A={gid_a} B={gid_b}",
    )

    check(
        "A -> B CROSS-CAMERA MATCH",
        bool(result_b.get("matched")),
        (
            f"matched={result_b.get('matched')} "
            f"type={result_b.get('match_type')} "
            f"score={result_b.get('correlation_score')}"
        ),
    )

    check(
        "A -> B NOT LOCAL CONTINUITY",
        result_b.get("match_type") != "existing_local_track",
        f"match_type={result_b.get('match_type')}",
    )

    # C has the same plate, vehicle type and exact same Re-ID embedding.
    # A different GlobalVehicleID therefore demonstrates that the physically
    # impossible candidate was removed before identity fusion/Re-ID matching.
    check(
        "B -> C GLOBAL ID ISOLATION",
        gid_c != gid_b,
        f"B={gid_b} C={gid_c}",
    )

    check(
        "C NOT MATCHED TO OLD VEHICLE",
        not bool(result_c.get("matched")),
        (
            f"matched={result_c.get('matched')} "
            f"type={result_c.get('match_type')}"
        ),
    )

    # Same camera + same local track after C must still preserve the new ID.
    continuity_vehicle = vehicle_fixture(
        camera=c,
        track_id="POSTGIS-VEHICLE-C",
        timestamp=(
            base_time
            + valid_elapsed
            + impossible_elapsed
            + 1.0
        ),
        embedding=embedding,
    )

    continuity = correlate_observation(
        continuity_vehicle,
        embedding,
    )

    print("CAMERA C CONTINUITY:", continuity)

    check(
        "LOCAL TRACK SAME GLOBAL ID",
        continuity.get("global_vehicle_id") == gid_c,
        (
            f"before={gid_c} "
            f"after={continuity.get('global_vehicle_id')}"
        ),
    )

    check(
        "LOCAL TRACK CONTINUITY",
        continuity.get("match_type") == "existing_local_track",
        f"match_type={continuity.get('match_type')}",
    )

    check(
        "LOCAL TRACK MATCHED",
        bool(continuity.get("matched")),
        f"matched={continuity.get('matched')}",
    )

    banner("FINAL RESULT")
    print("FINAL RESULT                       : PASSED")
    print()
    print(
        "PostGIS accepted the feasible A -> B vehicle transition, "
        "rejected the physically impossible B -> C transition before "
        "cross-camera identity matching, prevented identical ANPR/Re-ID "
        "evidence from reusing the old GlobalVehicleID, and preserved "
        "same-camera local-track continuity."
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
