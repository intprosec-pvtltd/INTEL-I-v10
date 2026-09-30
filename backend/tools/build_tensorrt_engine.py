#!/usr/bin/env python3
from __future__ import annotations

"""Build a reproducible TensorRT engine for an exact INTEL-I model.

The script intentionally requires the real model file and the NVIDIA/TensorRT
build environment. It never creates a fake engine. Ultralytics performs the
model export; the resulting .engine is then checksum recorded in the manifest.
"""

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ai.tensorrt.manifest import sha256_file, gpu_name, tensorrt_version


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["vehicle_detector", "plate_detector"])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("models/tensorrt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="Use FP16. Recommended for NVIDIA production inference.")
    p.add_argument("--workspace", type=int, default=4, help="TensorRT workspace in GiB.")
    p.add_argument("--device", default="0")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"Source model not found: {model}")
    if model.suffix.lower() != ".pt":
        raise SystemExit("This builder expects the validated Ultralytics .pt source model")
    if args.imgsz < 320 or args.imgsz > 1536:
        raise SystemExit("imgsz must be between 320 and 1536")
    if args.workspace < 1 or args.workspace > 64:
        raise SystemExit("workspace must be between 1 and 64 GiB")

    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"PyTorch is required in the build environment: {exc}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required to build a TensorRT engine")

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    engine = outdir / f"{args.role}.engine"
    if engine.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing engine: {engine}; use --force")

    from ultralytics import YOLO
    yolo = YOLO(str(model))
    exported = yolo.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        workspace=args.workspace,
        nms=False,
        dynamic=False,
        simplify=True,
    )
    exported_path = Path(str(exported)).resolve()
    if not exported_path.is_file():
        raise SystemExit(f"Ultralytics export reported success but engine was not found: {exported_path}")
    if exported_path != engine:
        if engine.exists():
            engine.unlink()
        exported_path.replace(engine)

    source_hash = sha256_file(model)
    engine_hash = sha256_file(engine)
    try:
        names = yolo.names
    except Exception:
        names = {}
    entry = {
        "role": args.role,
        "source_model": str(model),
        "engine_path": str(engine),
        "sha256": engine_hash,
        "source_sha256": source_hash,
        "format": "TensorRT",
        "precision": "FP16" if args.half else "FP32",
        "workspace_mb": args.workspace * 1024,
        "input_size": [args.imgsz, args.imgsz],
        "dynamic_shapes": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "Ultralytics export -> TensorRT",
        "builder_version": str(getattr(__import__('ultralytics'), '__version__', 'unknown')),
        "platform": platform.platform(),
        "gpu_name": gpu_name(),
        "source_classes": names,
        "nms": False,
        "status": "BUILT_NOT_RUNTIME_VALIDATED",
    }
    manifest_path = outdir / "manifest.json"
    manifest = {"schema_version": 1, "engines": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["engines"] = [e for e in manifest.get("engines", []) if e.get("role") != args.role]
    manifest["engines"].append(entry)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(manifest_path)
    print(json.dumps({"engine": str(engine), "manifest": str(manifest_path), "entry": entry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
