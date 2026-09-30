from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BACKEND_ROOT / "config" / "model_manifest.json"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(raw: str) -> Path:
    p = Path(str(raw or "").strip())
    if p.is_absolute():
        return p
    return (BACKEND_ROOT / p).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    id: str
    component: str
    enabled_env: str
    path_env: str
    default_path: str
    sha256: str
    size_bytes: int | None
    max_bytes: int
    url_env: str
    default_url: str
    allowed_hosts_env: str
    required_when_enabled: bool

    @property
    def enabled(self) -> bool:
        return env_bool(self.enabled_env, False)

    @property
    def path(self) -> Path:
        return resolve_path(os.getenv(self.path_env, self.default_path))

    @property
    def expected_sha256(self) -> str:
        override = os.getenv(f"{self.id.upper()}_SHA256", "").strip().lower()
        generic_override = os.getenv({
            "edsr_x4": "EDSR_MODEL_SHA256",
            "darkir_384": "DARKIR_MODEL_SHA256",
            "yunet_2023mar": "FACE_DETECTOR_MODEL_SHA256",
            "sface_2021dec": "FACE_RECOGNIZER_MODEL_SHA256",
        }.get(self.id, ""), "").strip().lower()
        return override or generic_override or self.sha256.strip().lower()

    @property
    def url(self) -> str:
        return os.getenv(self.url_env, self.default_url).strip()


def load_manifest(path: Path = MANIFEST_PATH) -> list[ModelSpec]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    for item in data.get("models", []):
        specs.append(ModelSpec(
            id=str(item["id"]),
            component=str(item.get("component") or item["id"]),
            enabled_env=str(item["enabled_env"]),
            path_env=str(item["path_env"]),
            default_path=str(item["default_path"]),
            sha256=str(item.get("sha256") or ""),
            size_bytes=int(item["size_bytes"]) if item.get("size_bytes") is not None else None,
            max_bytes=int(item.get("max_bytes") or 1024 * 1024 * 1024),
            url_env=str(item.get("url_env") or ""),
            default_url=str(item.get("default_url") or ""),
            allowed_hosts_env=str(item.get("allowed_hosts_env") or "MODEL_DOWNLOAD_ALLOWED_HOSTS"),
            required_when_enabled=bool(item.get("required_when_enabled", True)),
        ))
    return specs
