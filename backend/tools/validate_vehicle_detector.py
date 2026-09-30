#!/usr/bin/env python3
from __future__ import annotations

"""Validate the exact INTEL-I vehicle detector before deployment.

Usage:
    python tools/validate_vehicle_detector.py data/vehicle.yaml --model models/vehicle.pt

The report records model checksum/version and Ultralytics validation metrics.
Use a held-out dataset representing the actual CCTV conditions rather than
training data.
"""

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", default="models/yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--output", type=Path, default=Path("vehicle_detector_validation.json"))
    args = ap.parse_args()

    model_path = Path(args.model)
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        verbose=False,
    )

    box = getattr(metrics, "box", None)
    report = {
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path) if model_path.is_file() else None,
            "names": getattr(model, "names", {}),
        },
        "validation": {
            "data": str(args.data),
            "imgsz": args.imgsz,
            "device": args.device,
            "conf": args.conf,
            "map50": float(getattr(box, "map50", 0.0)) if box else 0.0,
            "map50_95": float(getattr(box, "map", 0.0)) if box else 0.0,
            "precision": float(getattr(box, "mp", 0.0)) if box else 0.0,
            "recall": float(getattr(box, "mr", 0.0)) if box else 0.0,
        },
        "deployment_note": "Repeat validation by day/night/rain/occlusion/camera-angle cohorts before production rollout.",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
