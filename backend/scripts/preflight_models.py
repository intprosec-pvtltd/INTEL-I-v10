"""Fail startup when enabled INTEL-I model assets are missing or unverified."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from model_manifest import BACKEND_ROOT, load_manifest, resolve_path, sha256_file

load_dotenv(BACKEND_ROOT / ".env")


def enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "true" if default else "false")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


failures: list[str] = []
warnings: list[str] = []
checked = 0
require_pinned = enabled("MODEL_REQUIRE_PINNED_SHA256", os.getenv("ENV", "dev").strip().lower() == "prod")

# Controlled artifacts: checksum verification is mandatory when a checksum is pinned.
for spec in load_manifest():
    if not spec.enabled:
        continue
    checked += 1
    path = spec.path
    if not path.is_file():
        failures.append(f"{spec.component}: missing {path}")
        continue
    if path.stat().st_size > spec.max_bytes:
        failures.append(f"{spec.component}: exceeds maximum allowed size: {path}")
        continue
    expected = spec.expected_sha256
    if expected:
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            failures.append(f"{spec.component}: SHA-256 mismatch: {path}")
    else:
        message = f"{spec.component}: checksum not pinned; set the model SHA-256 env"
        if require_pinned:
            failures.append(message)
        else:
            warnings.append(message)

# Existing core assets that are not yet part of the controlled manifest.
required = {
    "person detector": os.getenv("MODEL_PERSON", "models/yolov8m.pt"),
    "pose detector": os.getenv("MODEL_POSE", "models/yolov8m-pose.pt"),
    "crime detector": os.getenv("MODEL_CRIME", "models/Suspicious_Activities_nano.pt"),
    "PPE detector": os.getenv("MODEL_PPE", "models/best.pt"),
    "vehicle detector": os.getenv("VEHICLE_MODEL", "models/yolov8m.pt"),
    "ANPR plate detector": os.getenv("ANPR_PLATE_MODEL", "models/license_plate_yolov8m.pt"),
    "vehicle Re-ID model": os.getenv("VEHICLE_REID_MODEL", "models/osnet_ain_x1_0_vehicle_reid.onnx"),
}
ocr_engine = os.getenv("ANPR_OCR_ENGINE", "awiros_paddle").strip().lower()
if ocr_engine == "awiros_paddle":
    required.update({
        "Awiros ANPR OCR weights": os.getenv(
            "ANPR_AWIROS_MODEL_PATH",
            "models/awiros_anpr_ocr/model.safetensors",
        ),
        "Awiros ANPR dictionary": os.getenv(
            "ANPR_AWIROS_DICT_PATH",
            "models/awiros_anpr_ocr/en_dict.txt",
        ),
        "PaddleOCR source tree": os.getenv(
            "ANPR_AWIROS_PADDLEOCR_DIR",
            "vendor/PaddleOCR",
        ),
    })
else:
    required.update({
        "ANPR OCR model": os.getenv(
            "ANPR_OCR_ONNX_MODEL",
            "models/en_PP-OCRv5_rec_mobile.onnx",
        ),
        "ANPR dictionary": os.getenv(
            "ANPR_OCR_DICT",
            "models/ppocrv5_en_dict.txt",
        ),
    })
if enabled("VEHICLE_ATTRIBUTE_ENABLED", True):
    required["vehicle attribute model"] = os.getenv(
        "VEHICLE_ATTRIBUTE_MODEL_PATH", "models/PP-LCNet_x1_0_vehicle_attribute_infer"
    )
for label, raw in required.items():
    path = resolve_path(raw)
    checked += 1
    if not path.exists():
        failures.append(f"{label}: missing {path}")

if ocr_engine == "awiros_paddle":
    paddle_root = resolve_path(required["PaddleOCR source tree"])
    if paddle_root.exists() and not (paddle_root / "ppocr" / "__init__.py").is_file():
        failures.append(
            f"PaddleOCR source tree: missing {paddle_root / 'ppocr' / '__init__.py'}"
        )

attribute_raw = required.get("vehicle attribute model")
if attribute_raw:
    attribute_dir = resolve_path(attribute_raw)
    if attribute_dir.is_dir():
        for filename in ("inference.json", "inference.pdiparams", "inference.yml"):
            candidate = attribute_dir / filename
            if not candidate.is_file():
                failures.append(f"vehicle attribute asset: missing {candidate}")

for warning in warnings:
    print(f"[models] WARNING: {warning}")
if failures:
    raise SystemExit("Required model assets failed preflight:\n- " + "\n- ".join(failures))
print(f"[models] Preflight passed ({checked} assets)")
