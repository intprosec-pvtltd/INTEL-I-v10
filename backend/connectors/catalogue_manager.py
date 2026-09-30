"""Registry for catalogue-level camera integrations."""
from __future__ import annotations

from typing import Any

from connectors.catalogue import CameraCatalogueConnector
from connectors.sentinel import SentinelCatalogueConnector
from connectors.vms import GenericVMSCatalogueConnector


CATALOGUE_PROVIDER_TYPES = frozenset({"sentinel", "generic_vms", "generic_nvr"})


def normalize_catalogue_provider(value: Any) -> str:
    provider = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if provider == "generic_nvr":
        provider = "generic_vms"
    if provider not in {"sentinel", "generic_vms"}:
        raise ValueError("Unsupported camera catalogue provider")
    return provider


def create_catalogue_connector(
    provider_type: str,
    config: dict[str, Any],
    *,
    transport=None,
) -> CameraCatalogueConnector:
    provider = normalize_catalogue_provider(provider_type)
    if provider == "sentinel":
        return SentinelCatalogueConnector(config, transport=transport)
    return GenericVMSCatalogueConnector(config, transport=transport)

