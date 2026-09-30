"""Export the configured INTEL-I primary detector to a dynamic FP16 TensorRT engine.

The generated .engine is a deployment artifact and is intentionally excluded
from source packages. Run this on the target NVIDIA GPU/runtime combination.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MODEL_PERSON", "models/yolov8m.pt"))
    parser.add_argument("--batch", type=int, default=int(os.getenv("INFERENCE_BATCH_SIZE", "16")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("PRIMARY_DETECTOR_IMGSZ", "640")))
    parser.add_argument("--workspace", type=float, default=float(os.getenv("TENSORRT_WORKSPACE_GB", "4")))
    args=parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to export TensorRT")
    source=Path(args.model)
    if not source.exists():
        raise SystemExit(f"Model not found: {source}")
    from ultralytics import YOLO
    model=YOLO(str(source))
    result=model.export(format="engine",half=True,dynamic=True,batch=max(1,min(32,args.batch)),imgsz=args.imgsz,device=0,workspace=args.workspace)
    print(f"TensorRT FP16 export complete: {result}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
