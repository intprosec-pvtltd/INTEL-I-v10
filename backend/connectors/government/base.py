from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx


class GovernmentConnectorError(RuntimeError): pass
class GovernmentConnectorNotConfigured(GovernmentConnectorError): pass


def _safe_host(url: str, allowed_hosts: set[str]) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise GovernmentConnectorError("government connector requires HTTPS")
    host = parsed.hostname.lower()
    if host not in allowed_hosts:
        raise GovernmentConnectorError("government connector host is not allowlisted")
    # Prevent a DNS rebinding/configuration mistake from silently targeting local metadata.
    try:
        for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM):
            ip = ipaddress.ip_address(item[4][0])
            if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
                raise GovernmentConnectorError("government connector resolved to a forbidden address")
    except socket.gaierror as exc:
        raise GovernmentConnectorError("government connector DNS resolution failed") from exc
    return host


@dataclass(frozen=True)
class ConnectorResult:
    provider: str
    request_id: str
    records: list[dict[str, Any]]
    latency_ms: float


class GovernmentDataConnector:
    """Production transport boundary for authorized government data systems.

    Provider-specific field contracts must be supplied from the authorized API
    specification. This class deliberately does not guess VAHAN/SARATHI/CCTNS/
    AFIS/NAFIS endpoints or schemas.
    """
    def __init__(self, provider: str):
        self.provider = provider.upper()
        prefix = f"GOV_{self.provider}_"
        self.enabled = os.getenv(prefix + "ENABLED", "false").lower() in {"1","true","yes","on"}
        self.base_url = os.getenv(prefix + "BASE_URL", "").strip().rstrip("/")
        self.search_path = os.getenv(prefix + "SEARCH_PATH", "").strip()
        self.token = os.getenv(prefix + "TOKEN", "")
        self.ca = os.getenv(prefix + "CA_CERT", "").strip() or True
        self.client_cert = os.getenv(prefix + "CLIENT_CERT", "").strip() or None
        self.client_key = os.getenv(prefix + "CLIENT_KEY", "").strip() or None
        self.timeout = max(2.0, min(30.0, float(os.getenv(prefix + "TIMEOUT_SECONDS", "8"))))
        self.allowed_hosts = {x.strip().lower() for x in os.getenv(prefix + "ALLOWED_HOSTS", "").split(",") if x.strip()}

    def readiness(self) -> dict[str, Any]:
        configured = bool(self.base_url and self.search_path and self.allowed_hosts)
        return {"provider": self.provider, "enabled": self.enabled, "configured": configured, "ready": (not self.enabled) or configured}

    def search(self, *, criteria: dict[str, Any], purpose: str, actor_user_id: int) -> ConnectorResult:
        if not self.enabled:
            raise GovernmentConnectorNotConfigured(f"{self.provider} connector is disabled")
        if not self.base_url or not self.search_path or not self.allowed_hosts:
            raise GovernmentConnectorNotConfigured(f"{self.provider} connector is not configured")
        _safe_host(self.base_url, self.allowed_hosts)
        if not purpose.strip() or len(purpose) > 250:
            raise GovernmentConnectorError("an auditable query purpose is required")
        # Strict bounded JSON only; never forward arbitrary headers/URLs from a user.
        clean = json.loads(json.dumps(criteria, default=str))
        if not isinstance(clean, dict) or len(json.dumps(clean)) > 16_000:
            raise GovernmentConnectorError("invalid or oversized connector criteria")
        request_id = hashlib.sha256(f"{self.provider}:{actor_user_id}:{time.time_ns()}".encode()).hexdigest()[:24]
        headers = {"Accept": "application/json", "X-INTEL-I-Request-ID": request_id, "X-INTEL-I-Purpose": purpose[:250]}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        cert = (self.client_cert, self.client_key) if self.client_cert and self.client_key else None
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout, verify=self.ca, cert=cert, follow_redirects=False) as client:
            response = client.post(f"{self.base_url}/{self.search_path.lstrip('/')}", json={"criteria": clean}, headers=headers)
            response.raise_for_status()
            if int(response.headers.get("content-length", "0") or 0) > 2_000_000:
                raise GovernmentConnectorError("government connector response too large")
            body = response.json()
        records = body.get("records", []) if isinstance(body, dict) else []
        if not isinstance(records, list):
            raise GovernmentConnectorError("government connector returned invalid schema")
        return ConnectorResult(self.provider, request_id, records[:500], (time.perf_counter()-started)*1000.0)
