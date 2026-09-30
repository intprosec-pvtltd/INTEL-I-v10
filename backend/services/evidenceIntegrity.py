from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def build_evidence_manifest(
    alert_id: str,
    snapshot_path: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = {
        "alert_id": str(alert_id),
        "metadata": metadata,
    }

    if snapshot_path:
        path = Path(snapshot_path)
        manifest["snapshot"] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    return manifest
