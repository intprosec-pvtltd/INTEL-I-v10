from __future__ import annotations
import ipaddress
import json
import os
import socket
import time
from urllib.parse import urlparse
from integrations.government_watchlist.providers.base import WatchlistProvider


def validate_remote_url(url: str) -> str:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError("Endpoint must be an HTTPS URL without embedded credentials, query strings, or fragments")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Private and local endpoints are not allowed")
    trusted_hosts = {item.strip().lower().rstrip(".") for item in os.getenv("EXTERNAL_WATCHLIST_ALLOWED_HOSTS", "").split(",") if item.strip()}
    explicitly_trusted = host in trusted_hosts
    if parsed.port not in (None, 443) and not explicitly_trusted:
        raise ValueError("Non-standard HTTPS ports must be explicitly trusted by server configuration")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)]
        except OSError as exc:
            raise ValueError("Endpoint hostname could not be resolved") from exc
    if not addresses or (not explicitly_trusted and any(not address.is_global for address in addresses)):
        raise ValueError("Endpoint must resolve only to public IP addresses or be explicitly trusted by server configuration")
    return url


class RestJsonProvider(WatchlistProvider):
    def fetch_records(self, source, credential: str | None) -> list[dict]:
        import httpx
        url = validate_remote_url(source.base_url)
        connect_timeout = max(1.0, min(float(os.getenv("EXTERNAL_WATCHLIST_CONNECT_TIMEOUT_SECONDS", "5")), 20.0))
        read_timeout = max(1.0, min(float(os.getenv("EXTERNAL_WATCHLIST_READ_TIMEOUT_SECONDS", "15")), 60.0))
        max_bytes = max(1024, min(int(os.getenv("EXTERNAL_WATCHLIST_MAX_RESPONSE_BYTES", str(5 * 1024 * 1024))), 25 * 1024 * 1024))
        retries = max(0, min(int(os.getenv("EXTERNAL_WATCHLIST_MAX_RETRIES", "2")), 4))
        headers = {"Accept": "application/json", "User-Agent": "INTEL-I-WatchlistGateway/1.0"}
        if source.auth_type == "BEARER" and credential:
            headers["Authorization"] = f"Bearer {credential}"
        elif source.auth_type == "API_KEY_HEADER" and credential:
            headers[str(source.api_key_header)] = credential
        config = dict(source.pagination or {})
        ca_bundle = os.getenv("EXTERNAL_WATCHLIST_CA_BUNDLE", "").strip()
        tls_setting = ca_bundle if source.tls_verify and ca_bundle else bool(source.tls_verify)
        records: list[dict] = []
        page = int(config.get("start_page", 1))
        page_param = str(config.get("page_param", "page"))[:64]
        max_pages = max(1, min(int(config.get("max_pages", 20)), 100))
        with httpx.Client(timeout=httpx.Timeout(read_timeout, connect=connect_timeout), verify=tls_setting, follow_redirects=False) as client:
            for page_count in range(max_pages):
                params = {page_param: page} if config.get("enabled") else None
                body = None
                for attempt in range(retries + 1):
                    try:
                        with client.stream("GET", url, headers=headers, params=params) as response:
                            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                                retry = True
                            else:
                                retry = False
                                response.raise_for_status()
                                chunks = []
                                size = 0
                                for chunk in response.iter_bytes():
                                    size += len(chunk)
                                    if size > max_bytes:
                                        raise ValueError("External response exceeded the configured size limit")
                                    chunks.append(chunk)
                                body = b"".join(chunks)
                        if body is not None:
                            break
                    except httpx.RequestError:
                        if attempt >= retries:
                            raise
                        retry = True
                    if retry:
                        time.sleep(min(0.5 * (2 ** attempt), 3.0))
                if body is None:
                    raise RuntimeError("External source did not return a response")
                try:
                    payload = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                    raise ValueError("External source returned invalid JSON") from exc
                if isinstance(payload, list):
                    batch = payload
                else:
                    batch = payload
                    for part in str(source.records_path or "records").split("."):
                        if isinstance(batch, dict):
                            batch = batch.get(part)
                        else:
                            batch = None
                            break
                if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
                    raise ValueError("Configured record path must resolve to a JSON array of objects")
                records.extend(batch)
                if not config.get("enabled") or len(batch) < int(config.get("page_size", 100)):
                    break
                page += 1
                if len(records) > 50000:
                    raise ValueError("External source exceeded the 50,000 record sync limit")
        return records
