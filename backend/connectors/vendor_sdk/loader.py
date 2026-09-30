from __future__ import annotations

import importlib
import os
from typing import Any

from connectors.base import ConnectorResult
from connectors.vendor_sdk.adapter import VendorSDKAdapter


def _allowed_modules() -> set[str]:
    raw = os.getenv(
        "INTEL_I_SDK_MODULE_ALLOWLIST",
        "connectors.vendor_sdk_plugins",
    )
    return {
        item.strip()
        for item in raw.split(",")
        if item.strip()
    }


def _load_class(module_name: str, class_name: str):
    allowed = _allowed_modules()
    if module_name not in allowed and not any(
        module_name.startswith(prefix + ".") for prefix in allowed
    ):
        raise ValueError("Vendor SDK module is not allowlisted")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        raise ValueError("Vendor SDK adapter class was not found")

    if not issubclass(cls, VendorSDKAdapter):
        raise ValueError("Vendor SDK adapter does not implement the INTEL-I SDK contract")

    return cls


class VendorSDKConnector(VendorSDKAdapter):
    connector_type = "vendor_sdk"

    def resolve(self) -> ConnectorResult:
        module_name = str(self.config.get("module") or "").strip()
        class_name = str(self.config.get("class_name") or "").strip()
        if not module_name or not class_name:
            raise ValueError("Vendor SDK module and class_name are required")

        adapter_cls = _load_class(module_name, class_name)
        adapter_config = self.config.get("config") or {}
        if not isinstance(adapter_config, dict):
            raise ValueError("Vendor SDK config must be an object")

        adapter = adapter_cls(adapter_config)
        return adapter.resolve()

    def health_check(self) -> dict[str, Any]:
        try:
            result = self.resolve()
            return {"ok": True, "source_type": result.source_type}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:160]}
