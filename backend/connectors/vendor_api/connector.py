from __future__ import annotations

import ipaddress
import json
import socket
from typing import Any
from urllib.parse import quote, urljoin, urlparse, urlunparse

import httpx

from connectors.base import CameraConnector, ConnectorResult


ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(value: Any, field: str) -> str:
    url = str(value or "").strip()
    if not url or len(url) > 2048:
        raise ValueError(f"Invalid {field}")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES or not parsed.hostname:
        raise ValueError(f"{field} must use http:// or https://")
    if parsed.username or parsed.password:
        raise ValueError(f"{field} must not contain credentials")
    if any(c in url for c in "\r\n"):
        raise ValueError(f"Invalid {field}")
    return url


def _nested_get(data: Any, path: str | None) -> Any:
    if not path:
        return data
    current = data
    for part in str(path).split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


def _is_safe_stream_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"rtsp", "http", "https"}:
        return False
    return bool(parsed.hostname)


def _apply_stream_auth(url: str, config: dict[str, Any]) -> str:
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        return url

    username = str(config.get("stream_username") or "")
    password = str(config.get("stream_password") or "")
    if not username:
        return url

    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"

    return urlunparse((
        parsed.scheme,
        netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


class VendorAPIConnector(CameraConnector):
    connector_type = "vendor_api"

    def _headers(self) -> dict[str, str]:
        headers = {}
        raw = self.config.get("headers") or {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                k = str(key).strip()
                v = str(value)
                if k and len(k) <= 128 and len(v) <= 4096:
                    headers[k] = v

        auth_type = str(self.config.get("auth_type") or "none").lower()
        username = str(self.config.get("username") or "")
        password = str(self.config.get("password") or "")
        token = str(self.config.get("token") or "")
        api_key = str(self.config.get("api_key") or "")

        if auth_type == "bearer" and token:
            headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "api_key" and api_key:
            header_name = str(self.config.get("api_key_header") or "X-API-Key")
            if 1 <= len(header_name) <= 128:
                headers[header_name] = api_key

        return headers

    def _auth(self):
        auth_type = str(self.config.get("auth_type") or "none").lower()
        if auth_type == "basic":
            username = str(self.config.get("username") or "")
            password = str(self.config.get("password") or "")
            if not username:
                raise ValueError("Vendor API username is required for basic authentication")
            return (username, password)
        return None

    def _client(self) -> httpx.Client:
        timeout = min(max(float(self.config.get("timeout_seconds", 8)), 1.0), 30.0)
        return httpx.Client(
            timeout=httpx.Timeout(timeout, connect=timeout),
            follow_redirects=False,
            headers=self._headers(),
        )

    def _resolve_direct_stream(self) -> str | None:
        value = self.config.get("stream_url")
        if not value:
            return None
        stream_url = _apply_stream_auth(str(value).strip(), self.config)
        if not _is_safe_stream_url(stream_url):
            raise ValueError("Vendor API stream_url is invalid")
        return stream_url

    def resolve(self) -> ConnectorResult:
        direct = self._resolve_direct_stream()
        if direct:
            return ConnectorResult(
                source=direct,
                source_type=(
                    "rtsp"
                    if direct.lower().startswith("rtsp://")
                    else "hls"
                    if direct.lower().split("?", 1)[0].endswith(".m3u8")
                    else "http"
                ),
                metadata={"connector_type": "vendor_api", "resolved_source_type": "direct"},
            )

        base_url = _validate_url(self.config.get("base_url"), "Vendor API base_url")
        stream_path = str(self.config.get("stream_path") or "").strip()
        if not stream_path.startswith("/"):
            stream_path = "/" + stream_path if stream_path else "/"
        endpoint = urljoin(base_url.rstrip("/") + "/", stream_path.lstrip("/"))

        method = str(self.config.get("method") or "GET").upper()
        if method not in {"GET", "POST"}:
            raise ValueError("Vendor API stream method must be GET or POST")

        body = self.config.get("body")
        if body is not None and not isinstance(body, dict):
            raise ValueError("Vendor API body must be a JSON object")

        with self._client() as client:
            response = client.request(
                method,
                endpoint,
                json=body if method == "POST" else None,
                auth=self._auth(),
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Vendor API returned HTTP {response.status_code}")

            if len(response.content) > 2 * 1024 * 1024:
                raise RuntimeError("Vendor API response is too large")

            content_type = response.headers.get("content-type", "").lower()
            if "json" in content_type:
                data = response.json()
            else:
                try:
                    data = json.loads(response.text)
                except json.JSONDecodeError:
                    data = response.text.strip()

        stream_url = _nested_get(data, self.config.get("stream_json_path") or "stream_url")
        if not isinstance(stream_url, str):
            raise RuntimeError("Vendor API did not return a valid stream URL")
        stream_url = _apply_stream_auth(stream_url.strip(), self.config)
        if not _is_safe_stream_url(stream_url):
            raise RuntimeError("Vendor API did not return a valid stream URL")

        return ConnectorResult(
            source=stream_url.strip(),
            source_type=(
                "rtsp"
                if stream_url.lower().startswith("rtsp://")
                else "hls"
                if stream_url.lower().split("?", 1)[0].endswith(".m3u8")
                else "http"
            ),
            metadata={
                "connector_type": "vendor_api",
                "resolved_source_type": (
                    "rtsp"
                    if stream_url.lower().startswith("rtsp://")
                    else "hls"
                    if stream_url.lower().split("?", 1)[0].endswith(".m3u8")
                    else "http"
                ),
            },
        )

    def health_check(self) -> dict[str, Any]:
        try:
            self.resolve()
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:160]}
