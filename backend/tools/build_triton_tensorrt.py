"""Build YOLO engines on the deployment GPU; never downloads missing weights.

Uses the same Ultralytics exporter/runtime as the existing project, preserving
task metadata, letterboxing, class names, pose outputs and NMS. No INT8 path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace-gb", type=float, default=2)
    parser.add_argument("--output", type=Path, default=Path("models/tensorrt/triton-engines.json"))
    args = parser.parse_args()
    if not 1 <= args.batch <= 32 or args.imgsz % 32 or args.workspace_gb <= 0:
        parser.error("invalid batch/image size/workspace")
    names = ["yolov8m.pt", "yolov8m-pose.pt", "Suspicious_Activities_nano.pt", "best.pt", "license_plate_yolov8m.pt"]
    for name in names:
        if not (args.models / name).is_file():
            parser.error(f"Provision existing model: {args.models / name}")
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("Build requires target NVIDIA GPU and matching TensorRT runtime")
    records = {}
    for name in names:
        source = args.models / name
        model = YOLO(str(source))
        engine = Path(model.export(format="engine", half=True, dynamic=True,
                                 batch=args.batch, imgsz=args.imgsz, workspace=args.workspace_gb,
                                 device=0, nms=False))
        # Engine construction succeeding is not an accuracy acceptance test.
        records[name] = {"path": str(engine.resolve()), "sha256": sha256(engine),
                         "source_sha256": sha256(source), "task": model.task,
                         "names": model.names, "batch": args.batch, "imgsz": args.imgsz,
                         "accuracy_validated": False}
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "precision": "fp16", "models": records}, indent=2))
    os.replace(temporary, args.output)
    print(f"Built {len(records)} engines; accuracy comparison still required: {args.output}")


if __name__ == "__main__":
    main()
