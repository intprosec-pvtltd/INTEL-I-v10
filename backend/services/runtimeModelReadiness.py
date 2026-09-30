from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.model_manifest import load_manifest, sha256_file

_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {"verified_at": None, "models": [], "status": "NOT_VERIFIED"}


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _verify_darkir(path: Path) -> tuple[bool, str | None, str | None]:
    from services.darkIR import DarkIRRestorer
    verifier = DarkIRRestorer(
        path,
        os.getenv("DARKIR_DEVICE", "auto"),
        max_width=320,
        max_height=240,
        strict=True,
        expected_sha256=os.getenv("DARKIR_MODEL_SHA256", ""),
    )
    image = np.full((64, 96, 3), 24, dtype=np.uint8)
    result = verifier.enhance(image)
    ok = bool(result.get("model_used")) and isinstance(result.get("frame"), np.ndarray)
    status = verifier.status()
    return ok, status.get("actual_device") or status.get("device") or status.get("loaded_device"), None if ok else "darkir_inference_failed"


def _verify_edsr(path: Path) -> tuple[bool, str | None, str | None]:
    from services.aiEnhancementModel import EDSR4x
    verifier = EDSR4x(path, os.getenv("EDSR_DEVICE", "cpu"))
    if not verifier.load():
        return False, verifier.status().get("loaded_device"), "edsr_load_failed"
    # Keep startup self-test deliberately tiny. It proves executable graph integrity,
    # not throughput; throughput is covered by the camera benchmark acceptance suite.
    image = np.full((8, 16, 3), 128, dtype=np.uint8)
    output = verifier.upscale(image)
    ok = isinstance(output, np.ndarray) and output.shape[:2] == (32, 64)
    return ok, verifier.status().get("loaded_device"), None if ok else "edsr_inference_failed"


def _verify_yunet(path: Path) -> tuple[bool, str | None, str | None]:
    threshold = float(os.getenv("PERSON_FACE_MIN_DET_SCORE", "0.70"))
    detector = cv2.FaceDetectorYN.create(str(path), "", (64, 64), threshold, 0.3, 5000)
    detector.setInputSize((64, 64))
    detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    return True, "cpu/opencv", None


def _verify_sface(path: Path) -> tuple[bool, str | None, str | None]:
    recognizer = cv2.FaceRecognizerSF.create(str(path), "")
    # SFace accepts its canonical aligned 112x112 face tensor. A blank image is
    # sufficient for an executable-graph smoke test; biometric accuracy is tested separately.
    feature = recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
    ok = isinstance(feature, np.ndarray) and feature.size > 0 and np.isfinite(feature).all()
    return ok, "cpu/opencv", None if ok else "sface_inference_failed"


_VERIFIERS = {
    "darkir_384": _verify_darkir,
    "edsr_x4": _verify_edsr,
    "yunet_2023mar": _verify_yunet,
    "sface_2021dec": _verify_sface,
}


def verify_models(*, force: bool = False) -> dict[str, Any]:
    global _CACHE
    with _LOCK:
        if _CACHE.get("verified_at") and not force:
            return dict(_CACHE)
        strict_sha = _bool("MODEL_REQUIRE_PINNED_SHA256", os.getenv("ENV", "dev").lower() == "prod")
        rows: list[dict[str, Any]] = []
        all_ready = True
        for spec in load_manifest():
            configured = bool(str(spec.path))
            present = spec.path.is_file()
            expected = spec.expected_sha256
            checksum_verified = False
            checksum_actual = None
            if present and expected:
                checksum_actual = sha256_file(spec.path)
                checksum_verified = checksum_actual.lower() == expected.lower()
            elif present and not strict_sha:
                checksum_verified = True

            loaded = False
            inference_verified = False
            actual_device = None
            error = None
            if spec.enabled and configured and present and checksum_verified:
                verifier = _VERIFIERS.get(spec.id)
                if verifier is None:
                    error = "no_runtime_verifier"
                else:
                    try:
                        inference_verified, actual_device, error = verifier(spec.path)
                        loaded = bool(inference_verified or error not in {"darkir_load_failed", "edsr_load_failed"})
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {str(exc)[:240]}"
            ready = (not spec.enabled) or (
                configured and present and checksum_verified and loaded and inference_verified
            )
            if spec.enabled and not ready:
                all_ready = False
            rows.append({
                "id": spec.id,
                "component": spec.component,
                "enabled": spec.enabled,
                "configured": configured,
                "model_present": present,
                "checksum_pinned": bool(expected),
                "checksum_verified": checksum_verified,
                "loaded": loaded,
                "inference_verified": inference_verified,
                "requested_device": (
                    os.getenv("DARKIR_DEVICE", "auto") if spec.id == "darkir_384" else
                    os.getenv("EDSR_DEVICE", "cpu") if spec.id == "edsr_x4" else
                    os.getenv("PERSON_FACE_DEVICE", "cpu")
                ),
                "actual_device": actual_device,
                "ready": ready,
                "error": error,
            })
        _CACHE = {
            "status": "READY" if all_ready else "NOT_READY",
            "verified_at": time.time(),
            "models": rows,
        }
        return dict(_CACHE)


def cached_model_readiness() -> dict[str, Any]:
    with _LOCK:
        return dict(_CACHE)
