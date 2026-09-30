"""Deterministic no-GPU preflight for three-camera person correlation policy."""
import json
import time

from services.personCorrelation import PersonCorrelationEngine


def main():
    engine = PersonCorrelationEngine()
    descriptor = [0.0] * 96
    descriptor[7] = 1.0
    appearance = {"valid": True, "upper_color": "blue", "lower_color": "black", "appearance_descriptor": descriptor}
    results = []
    transitions = {("CAM-01", "CAM-02"), ("CAM-02", "CAM-03")}
    for index, camera in enumerate(("CAM-01", "CAM-02", "CAM-03")):
        results.append(engine.correlate(user_id=1, camera_id=camera, local_track_id=f"T-{index+1}", appearance=appearance, timestamp=time.time() + index * 20, allowed_transitions=transitions))
    ids = {row["global_person_id"] for row in results}
    report = {"ok": len(ids) == 1, "camera_count": 3, "global_person_ids": sorted(ids), "results": results}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
