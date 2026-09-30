from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_ROLES = {"vehicle_detector", "plate_detector", "vehicle_reid"}


@dataclass(frozen=True)
class EngineArtifact:
    role: str
    source_model: str
    engine_path: str
    sha256: str
    source_sha256: str
    format: str
    precision: str
    workspace_mb: int
    input_size: tuple[int, int]
    dynamic_shapes: dict[str, list[int]] | None
    created_at_utc: str
    builder: str
    builder_version: str
    platform: str
    gpu_name: str | None
    status: str = "VALIDATED"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def gpu_name() -> str | None:
    output = _command_output([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"
    ])
    return output.splitlines()[0].strip() if output else None


def tensorrt_version() -> str | None:
    try:
        import tensorrt as trt  # type: ignore
        return str(getattr(trt, "__version__", "unknown"))
    except Exception:
        return None


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("TensorRT manifest must be a JSON object")
    return data


def validate_manifest_entry(entry: dict[str, Any], *, require_engine: bool = True) -> tuple[bool, list[str]]:
    errors: list[str] = []
    role = str(entry.get("role", "")).strip()
    if role not in SUPPORTED_ROLES:
        errors.append(f"unsupported role: {role!r}")

    for field in ("source_model", "engine_path", "sha256", "source_sha256", "format", "precision"):
        if not str(entry.get(field, "")).strip():
            errors.append(f"missing field: {field}")

    engine = Path(str(entry.get("engine_path", "")))
    source = Path(str(entry.get("source_model", "")))
    if require_engine and not engine.is_file():
        errors.append(f"engine missing: {engine}")
    if not source.is_file():
        errors.append(f"source model missing: {source}")

    if engine.is_file() and entry.get("sha256"):
        actual = sha256_file(engine)
        if actual.lower() != str(entry["sha256"]).lower():
            errors.append("engine SHA-256 mismatch")
    if source.is_file() and entry.get("source_sha256"):
        actual = sha256_file(source)
        if actual.lower() != str(entry["source_sha256"]).lower():
            errors.append("source model SHA-256 mismatch")

    return not errors, errors


def validate_manifest(manifest_path: Path, *, require_engine: bool = True) -> dict[str, Any]:
    data = load_manifest(manifest_path)
    entries = data.get("engines")
    if not isinstance(entries, list):
        return {"ok": False, "errors": ["manifest.engines must be a list"], "engines": []}
    results = []
    all_ok = True
    for entry in entries:
        if not isinstance(entry, dict):
            results.append({"ok": False, "errors": ["engine entry must be an object"]})
            all_ok = False
            continue
        ok, errors = validate_manifest_entry(entry, require_engine=require_engine)
        results.append({"role": entry.get("role"), "ok": ok, "errors": errors})
        all_ok = all_ok and ok
    return {"ok": all_ok, "errors": [], "engines": results}
