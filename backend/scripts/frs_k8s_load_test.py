"""Concurrent multi-camera load test for an in-cluster FRS Service."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time

import cv2
import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("FRS_REMOTE_URL", "http://frs-worker:9202"))
    parser.add_argument("--token", default=os.getenv("FRS_INTERNAL_TOKEN", ""))
    parser.add_argument("--image", required=True)
    parser.add_argument("--cameras", type=int, default=50)
    parser.add_argument("--requests-per-camera", type=int, default=20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--max-p95-ms", type=float, default=1000.0)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit("Unable to decode --image")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise SystemExit("Unable to encode --image")
    body = encoded.tobytes()
    headers = {"X-Intel-I-FRS-Token": args.token}

    jobs = [(camera, sequence) for camera in range(args.cameras) for sequence in range(args.requests_per_camera)]

    def request(job):
        camera, sequence = job
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{args.url.rstrip('/')}/internal/frs/observe",
                headers=headers,
                data={"user_id": args.user_id, "camera_id": f"LOAD-{camera:04d}", "track_id": f"T-{sequence % 5}", "timestamp": time.time(), "detector_confidence": 0.99},
                files={"face_image": ("face.jpg", body, "image/jpeg")},
                timeout=10,
            )
            response.raise_for_status()
            return (time.perf_counter() - started) * 1000.0, True
        except Exception:
            return (time.perf_counter() - started) * 1000.0, False

    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(request, jobs))
    wall_seconds = time.perf_counter() - wall_started
    latencies = sorted(item[0] for item in results)
    successes = sum(1 for _, success in results if success)
    error_rate = 1.0 - successes / max(1, len(results))
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    report = {
        "cameras": args.cameras, "requests": len(results), "successes": successes,
        "error_rate": round(error_rate, 6), "throughput_rps": round(len(results) / wall_seconds, 3),
        "latency_ms": {"mean": round(statistics.mean(latencies), 3), "p95": round(p95, 3), "max": round(max(latencies), 3)},
        "accepted": error_rate <= args.max_error_rate and p95 <= args.max_p95_ms,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
