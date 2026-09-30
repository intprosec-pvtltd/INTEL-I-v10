from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scripts.model_manifest import load_manifest, sha256_file


def model_deployment_status() -> dict[str, Any]:
    items = []
    all_ready = True
    for spec in load_manifest():
        enabled = spec.enabled
        path = spec.path
        present = path.is_file()
        expected = spec.expected_sha256
        checksum_valid = None
        actual_sha = None
        if present and expected:
            actual_sha = sha256_file(path)
            checksum_valid = actual_sha.lower() == expected.lower()
        configured = bool(str(path))
        ready = (not enabled) or (configured and present and bool(expected) and checksum_valid is True)
        if enabled and not ready:
            all_ready = False
        requested_device = None
        actual_device = None
        if spec.id == "darkir_384":
            requested_device = os.getenv("DARKIR_DEVICE", "cuda")
        elif spec.id == "edsr_x4":
            requested_device = os.getenv("EDSR_DEVICE", "cpu")
            # Current EDSR4x wrapper explicitly reports CPU for OpenCV backend.
            try:
                from services.aiEnhancementModel import EDSR4x
                probe = EDSR4x(str(path), requested_device)
                probe.load()
                actual_device = probe.status().get("loaded_device")
            except Exception:
                actual_device = None
        elif spec.id in {"yunet_2023mar", "sface_2021dec"}:
            requested_device = os.getenv("PERSON_FACE_DEVICE", "cpu")
            actual_device = "cpu/opencv"
        items.append({
            "id": spec.id,
            "component": spec.component,
            "enabled": enabled,
            "configured": configured,
            "path": str(path),
            "present": present,
            "checksum_pinned": bool(expected),
            "checksum_valid": checksum_valid,
            "requested_device": requested_device,
            "actual_device": actual_device,
            "ready": ready,
        })
    return {"status": "READY" if all_ready else "NOT_READY", "models": items}
