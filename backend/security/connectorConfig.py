from __future__ import annotations

import json
from typing import Any

from security.cameraSource import encrypt_camera_source, decrypt_camera_source


MAX_CONNECTOR_CONFIG_BYTES = 16384


def encrypt_connector_config(config: dict[str, Any] | None) -> str | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("Connector configuration must be an object")

    encoded = json.dumps(
        config,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if len(encoded.encode("utf-8")) > MAX_CONNECTOR_CONFIG_BYTES:
        raise ValueError("Connector configuration is too large")

    return encrypt_camera_source(encoded)


def decrypt_connector_config(encrypted: str | None) -> dict[str, Any]:
    if not encrypted:
        return {}

    raw = decrypt_camera_source(encrypted)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid encrypted connector configuration") from exc

    if not isinstance(value, dict):
        raise ValueError("Connector configuration must be an object")

    return value
