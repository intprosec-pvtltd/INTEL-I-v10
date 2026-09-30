from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"sha256": sha256_bytes(value), "size": len(value)}
    return str(value)


def canonical_manifest_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return sha256_bytes(encoded)
