from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai.tensorrt.manifest import sha256_file, validate_manifest


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "artifact.engine"
    p.write_bytes(b"intel-i")
    expected = hashlib.sha256(b"intel-i").hexdigest()
    assert sha256_file(p) == expected


def test_manifest_rejects_missing_artifacts(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"engines": [{
        "role": "vehicle_detector",
        "source_model": str(tmp_path / "model.pt"),
        "engine_path": str(tmp_path / "vehicle_detector.engine"),
        "sha256": "0" * 64,
        "source_sha256": "0" * 64,
        "format": "TensorRT",
        "precision": "FP16",
    }]}), encoding="utf-8")
    result = validate_manifest(manifest, require_engine=True)
    assert result["ok"] is False
    assert any("engine missing" in e for e in result["engines"][0]["errors"])


def test_manifest_accepts_matching_hashes(tmp_path: Path):
    source = tmp_path / "model.pt"
    engine = tmp_path / "vehicle_detector.engine"
    source.write_bytes(b"source")
    engine.write_bytes(b"engine")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"engines": [{
        "role": "vehicle_detector",
        "source_model": str(source),
        "engine_path": str(engine),
        "sha256": sha256_file(engine),
        "source_sha256": sha256_file(source),
        "format": "TensorRT",
        "precision": "FP16",
    }]}), encoding="utf-8")
    result = validate_manifest(manifest, require_engine=True)
    assert result["ok"] is True
