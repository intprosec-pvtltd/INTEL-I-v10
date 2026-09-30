"""Sentinel Government camera catalogue adapter."""
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
    parse_json_response,
    safe_endpoint,
    validate_base_url,
)


class SentinelCatalogueConnector(CameraCatalogueConnector):
    provider_type = "sentinel"

    def __init__(self, config: dict[str, Any], *, transport=None):
        super().__init__(config)
        self.base_url = validate_base_url(self.config.get("base_url"))
        self.endpoint = safe_endpoint(self.base_url, self.config.get("ingest_path") or "/api/ingest")
        self.transport = transport

    @staticmethod
    def _camera_items(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        value = first_value(payload, ("cameras", "data.cameras", "data.items", "items", "results", "camera_catalogue"), [])
        return value if isinstance(value, list) else []

    def fetch_catalogue(self) -> CatalogueResult:
        headers, auth = auth_headers(self.config)
        method = str(self.config.get("method") or "GET").strip().upper()
        if method not in {"GET", "POST"}:
            raise ValueError("Sentinel ingest method must be GET or POST")
        body = self.config.get("body")
        if body is not None and not isinstance(body, dict):
            raise ValueError("Sentinel request body must be a JSON object")

        cameras = []
        seen_ids: set[str] = set()
        page = 1
        cursor = None
        revision = None
        etag = None

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
                    json=body if method == "POST" else None,
                    headers=headers,
                    auth=auth,
                )
                payload = parse_json_response(response)
                etag = response.headers.get("etag") or etag
                revision = first_value(payload, ("revision", "version", "data.revision", "data.version"), revision)

                items = self._camera_items(payload)
                for raw in items:
                    camera = generic_camera_from_mapping(raw, self.config.get("mapping"))
                    if camera and camera.external_id not in seen_ids:
                        cameras.append(camera)
                        seen_ids.add(camera.external_id)
                        if len(cameras) >= MAX_CAMERAS_PER_SYNC:
                            break

                cursor = first_value(payload, ("next_cursor", "pagination.next_cursor", "data.next_cursor"))
                has_more = bool(first_value(payload, ("has_more", "pagination.has_more", "data.has_more"), False))
                if cursor:
                    page += 1
                    continue
                if has_more and self.config.get("page_parameter") and items:
                    page += 1
                    continue
                break

        return CatalogueResult(
            cameras=tuple(cameras),
            fetched_pages=page,
            source_revision=str(revision)[:128] if revision is not None else None,
            etag=str(etag)[:256] if etag else None,
        )

