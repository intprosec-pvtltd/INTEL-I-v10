#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark the INTEL-I vehicle detector at 1/5/10/20/50 logical streams.

For repeatable GPU capacity tests, a local MP4 can be used as the source for
multiple logical streams. The benchmark measures decode/read rate, inference
latency, output FPS, queue depth, dropped frames and NVIDIA GPU utilization.
For an actual deployment acceptance test, repeat the same benchmark with the
real RTSP camera set because network/decoder behavior is site-specific.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def nvidia_metrics():
    try:
        p = subprocess.run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=2, check=False)
        if p.returncode != 0 or not p.stdout.strip():
            return {}
        a = [float(x.strip()) for x in p.stdout.splitlines()[0].split(",")]
        return {"gpu_util_percent": a[0], "memory_used_mb": a[1], "memory_total_mb": a[2], "temperature_c": a[3]}
    except Exception:
        return {}


def run_case(model, source, streams, duration, sample_fps, device, imgsz, batch):
    stop = threading.Event()
    lock = threading.Lock()
    latest_frames = [None] * streams
    pending = [False] * streams
    read_counts = [0] * streams
    dropped = [0] * streams
    max_queue_depth = 0
    latencies = []
    output_frames = 0
    started = time.perf_counter()

    def reader(index):
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return
        period = 1.0 / max(0.1, sample_fps)
        next_due = time.perf_counter()
        try:
            while not stop.is_set():
                now = time.perf_counter()
                if now < next_due:
                    time.sleep(min(0.01, next_due - now))
                    continue
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with lock:
                    if pending[index]:
                        dropped[index] += 1
                    latest_frames[index] = frame
                    pending[index] = True
                    read_counts[index] += 1
                next_due = max(next_due + period, now)
        finally:
            cap.release()

    def inference_loop():
        nonlocal output_frames, max_queue_depth
        while not stop.is_set():
            batch_frames = []
            owners = []
            with lock:
                max_queue_depth = max(max_queue_depth, sum(1 for value in pending if value))
                for index in range(streams):
                    if pending[index] and latest_frames[index] is not None:
                        owners.append(index)
                        batch_frames.append(latest_frames[index])
                        pending[index] = False
            if not batch_frames:
                time.sleep(0.002)
                continue
            for offset in range(0, len(batch_frames), max(1, batch)):
                current = batch_frames[offset:offset + max(1, batch)]
                t0 = time.perf_counter()
                model(current, imgsz=imgsz, device=device, verbose=False)
                elapsed = time.perf_counter() - t0
                per_frame = elapsed / max(1, len(current))
                with lock:
                    latencies.append(per_frame)
                    output_frames += len(current)

    with ThreadPoolExecutor(max_workers=min(streams, 32)) as pool:
        futures = [pool.submit(reader, i) for i in range(streams)]
        inf = threading.Thread(target=inference_loop, daemon=True)
        inf.start()
        time.sleep(max(5.0, duration))
        stop.set()
        inf.join(timeout=5)
        for future in futures:
            future.cancel()

    elapsed = time.perf_counter() - started
    total_input = sum(read_counts)
    total_dropped = sum(dropped)
    avg_latency = statistics.fmean(latencies) if latencies else 0.0
    return {
        "streams": streams,
        "duration_s": round(elapsed, 3),
        "input_frames": total_input,
        "input_fps_total": round(total_input / max(elapsed, 1e-6), 3),
        "output_frames": output_frames,
        "output_fps_total": round(output_frames / max(elapsed, 1e-6), 3),
        "output_fps_per_stream": round(output_frames / max(elapsed * streams, 1e-6), 3),
        "dropped_frames": total_dropped,
        "drop_rate": round(total_dropped / max(1, total_input + total_dropped), 6),
        "max_queue_depth": max_queue_depth,
        "avg_inference_seconds_per_frame": round(avg_latency, 6),
        "gpu": nvidia_metrics(),
        "note": "This capacity test uses a local source for repeatability. Run the final acceptance test against real RTSP streams and the production DeepStream pipeline.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--model", default=os.getenv("VEHICLE_MODEL", "models/yolov8m.pt"))
    ap.add_argument("--device", default=os.getenv("DEVICE", "0"))
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--sample-fps", type=float, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--streams", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    ap.add_argument("--output", type=Path, default=Path("vehicle_capacity_report.json"))
    args = ap.parse_args()

    model = YOLO(args.model)
    results = []
    for streams in args.streams:
        print(f"\n=== {streams} logical streams ===")
        result = run_case(model, args.source, streams, args.duration, args.sample_fps, args.device, args.imgsz, args.batch)
        results.append(result)
        print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
