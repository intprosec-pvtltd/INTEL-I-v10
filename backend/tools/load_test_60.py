"""Bounded camera traffic generator. Run only against a test deployment.

Synthetic, repeated file/RTSP, or one source per line. Exercises the central
API contract; real camera-worker ingest must also be tested via camera CRUD.
Metrics are observed, never fabricated when a telemetry endpoint is absent.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path


async def run(args):
    import cv2
    import httpx
    import numpy as np
    import psutil
    sources = args.sources.read_text().splitlines() if args.sources else [args.source]
    sources = [s.strip() for s in sources if s.strip()]
    if not sources:
        raise ValueError("no sources configured")
    samples, errors, telemetry = [], [], []
    counters = {"read_frames": 0, "submitted": 0, "shed_ticks": 0, "decode_failures": 0}
    stop = time.monotonic() + args.seconds
    headers = {"X-Intel-I-Inference-Token": os.environ["INFERENCE_INTERNAL_TOKEN"]}
    limits = httpx.Limits(max_connections=args.cameras + 2, max_keepalive_connections=args.cameras + 2)
    async with httpx.AsyncClient(timeout=args.timeout, headers=headers, limits=limits) as client:
        async def camera(index):
            source = sources[index % len(sources)]
            capture = None
            if source != "synthetic":
                capture = cv2.VideoCapture()
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                await asyncio.to_thread(capture.open, source)
            next_tick = time.monotonic()
            try:
                while time.monotonic() < stop:
                    await asyncio.sleep(max(0, next_tick - time.monotonic()))
                    now = time.monotonic()
                    missed = max(0, int((now - next_tick) * args.fps))
                    counters["shed_ticks"] += missed
                    next_tick = now + 1 / args.fps
                    if capture is None:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(frame, f"TEST CAMERA {index}", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    else:
                        ok, frame = await asyncio.to_thread(capture.read)
                        if not ok:
                            counters["decode_failures"] += 1
                            if Path(source).is_file():
                                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                    counters["read_frames"] += 1
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if not ok:
                        continue
                    started = time.monotonic()
                    counters["submitted"] += 1
                    try:
                        response = await client.post(args.url.rstrip("/") + "/v1/infer",
                            data={"camera_id": f"LOAD_{index:03}", "user_id": args.user_id,
                                  "timestamp": time.time(), "source_type": "rtsp", "frame_count": counters["read_frames"]},
                            files={"frame": ("frame.jpg", encoded.tobytes(), "image/jpeg")})
                        response.raise_for_status()
                        result = response.json()
                        if not result.get("ok") or result.get("camera_id") != f"LOAD_{index:03}":
                            raise ValueError("result routing mismatch")
                        samples.append((time.monotonic() - started) * 1000)
                    except Exception as exc:
                        errors.append(type(exc).__name__)  # Do not log source URLs/credentials.
            finally:
                if capture:
                    capture.release()

        async def monitor():
            while time.monotonic() < stop:
                row = {"time": time.time(), "generator_cpu_percent": psutil.cpu_percent(), "generator_ram_percent": psutil.virtual_memory().percent()}
                for key, endpoint in [("inference", args.url.rstrip("/") + "/health/ready"), ("triton_prometheus", args.metrics)]:
                    if endpoint:
                        try:
                            response = await client.get(endpoint)
                            response.raise_for_status()
                            row[key] = response.json() if key == "inference" else response.text
                        except Exception as exc:
                            row[key] = {"unavailable": type(exc).__name__}
                telemetry.append(row)
                await asyncio.sleep(min(5, max(0, stop - time.monotonic())))
        await asyncio.gather(monitor(), *(camera(i) for i in range(args.cameras)))
    percentiles = dict(zip(["p50", "p95", "p99"], np.percentile(samples, [50, 95, 99]).tolist())) if samples else None
    result = {"cameras": args.cameras, "requested_ai_fps_per_camera": args.fps,
              "seconds": args.seconds, **counters, "successful_frames": len(samples),
              "achieved_ai_fps_total": len(samples) / args.seconds, "errors": len(errors),
              "request_latency_ms": percentiles, "telemetry": telemetry,
              "capacity_confirmed": False,
              "limitations": "Synthetic scenes do not exercise downstream AI. This is central-service traffic, not full worker ingest. Alert latency requires timestamped ground-truth incidents."}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "telemetry"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:9300")
    p.add_argument("--cameras", type=int, default=60)
    p.add_argument("--fps", type=float, default=5)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--source", default="synthetic")
    p.add_argument("--sources", type=Path)
    p.add_argument("--user-id", type=int, required=True)
    p.add_argument("--metrics", default="")
    p.add_argument("--output", type=Path, default=Path("load-60-results.json"))
    a = p.parse_args()
    if not 1 <= a.cameras <= 200 or not 0 < a.fps <= 30 or not 0 < a.seconds <= 3600:
        p.error("invalid camera count, FPS or duration")
    asyncio.run(run(a))
