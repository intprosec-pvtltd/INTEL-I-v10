"""Mapping-driven REST catalogue adapter for VMS and NVR platforms."""
from __future__ import annotations

from typing import Any

from connectors.catalogue import CameraCatalogueConnector, CatalogueResult
from connectors.http_catalogue import (
    MAX_CAMERAS_PER_SYNC,
    MAX_CATALOGUE_PAGES,
    auth_headers,
    first_value,
    generic_camera_from_mapping,
    http_client,
    nested_get,
    parse_json_response,
    safe_endpoint,
    validate_base_url,
)


class GenericVMSCatalogueConnector(CameraCatalogueConnector):
    provider_type = "generic_vms"

    def __init__(self, config: dict[str, Any], *, transport=None):
        super().__init__(config)
        self.base_url = validate_base_url(self.config.get("base_url"))
        self.endpoint = safe_endpoint(self.base_url, self.config.get("catalogue_path") or "/api/cameras")
        self.transport = transport

    def fetch_catalogue(self) -> CatalogueResult:
        headers, auth = auth_headers(self.config)
        mapping = self.config.get("mapping") if isinstance(self.config.get("mapping"), dict) else {}
        items_path = str(mapping.get("items") or self.config.get("items_json_path") or "cameras")
        method = str(self.config.get("method") or "GET").strip().upper()
        if method not in {"GET", "POST"}:
            raise ValueError("VMS catalogue method must be GET or POST")

        cameras = []
        seen_ids: set[str] = set()
        cursor = None
        page = 1
        etag = None
        revision = None

        with http_client(self.config, transport=self.transport) as client:
            while page <= MAX_CATALOGUE_PAGES and len(cameras) < MAX_CAMERAS_PER_SYNC:
                params = dict(self.config.get("query") or {})
                if cursor:
                    params[str(self.config.get("cursor_parameter") or "cursor")] = cursor
                elif self.config.get("page_parameter"):
                    params[str(self.config["page_parameter"])] = page
                response = client.request(
                    method,
                    self.endpoint,
                    params=params,
                    json=self.config.get("body") if method == "POST" else None,
                    headers=headers,
                    auth=auth,
                )
                payload = parse_json_response(response)
                etag = response.headers.get("etag") or etag
                revision = first_value(payload, ("revision", "version", "data.revision"), revision)
                items = nested_get(payload, items_path, payload if isinstance(payload, list) else [])
                if not isinstance(items, list):
                    raise RuntimeError("Configured VMS camera list path did not resolve to a list")
                camera_mapping = {key: value for key, value in mapping.items() if key != "items"}
                for raw in items:
                    camera = generic_camera_from_mapping(raw, camera_mapping)
                    if camera and camera.external_id not in seen_ids:
                        cameras.append(camera)
                        seen_ids.add(camera.external_id)
                        if len(cameras) >= MAX_CAMERAS_PER_SYNC:
                            break

                cursor_path = str(self.config.get("next_cursor_json_path") or "next_cursor")
                cursor = nested_get(payload, cursor_path)
                has_more = bool(nested_get(payload, str(self.config.get("has_more_json_path") or "has_more"), False))
                if cursor or (has_more and self.config.get("page_parameter") and items):
                    page += 1
                    continue
                break

        return CatalogueResult(
            cameras=tuple(cameras),
            fetched_pages=page,
            source_revision=str(revision)[:128] if revision is not None else None,
            etag=str(etag)[:256] if etag else None,
        )

