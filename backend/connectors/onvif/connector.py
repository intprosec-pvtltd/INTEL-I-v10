from __future__ import annotations

import ipaddress
import re
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlparse

from connectors.base import CameraConnector, ConnectorResult


_HOST_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,255}$")


def _validate_host(value: Any) -> str:
    host = str(value or "").strip()
    if not host or len(host) > 255 or not _HOST_RE.fullmatch(host):
        raise ValueError("Invalid ONVIF host")

    # Reject URL syntax here; ONVIF host must be a hostname/IP.
    if "://" in host or "/" in host or "@" in host:
        raise ValueError("Invalid ONVIF host")

    try:
        ipaddress.ip_address(host)
    except ValueError:
        # Hostnames are accepted; URL parsing is not.
        pass

    return host


def _validate_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        port = 80
    if port < 1 or port > 65535:
        raise ValueError("Invalid ONVIF port")
    return port


class ONVIFConnector(CameraConnector):
    connector_type = "onvif"

    def _camera(self):
        try:
            from onvif import ONVIFCamera
        except ImportError as exc:
            raise RuntimeError(
                "ONVIF support is not installed. Install the backend ONVIF dependencies."
            ) from exc

        host = _validate_host(self.config.get("host"))
        port = _validate_port(self.config.get("port", 80))
        username = str(self.config.get("username") or "")
        password = str(self.config.get("password") or "")

        if not username or len(username) > 256:
            raise ValueError("ONVIF username is required")
        if len(password) > 4096:
            raise ValueError("ONVIF password is too long")

        return ONVIFCamera(host, port, username, password, no_cache=True)

    def discover_profiles(self) -> list[dict[str, Any]]:
        camera = self._camera()
        media = camera.create_media_service()
        profiles = media.GetProfiles()

        result = []
        for profile in profiles:
            token = str(getattr(profile, "token", "") or "").strip()
            name = str(getattr(profile, "Name", "") or "").strip()
            if not token:
                continue
            result.append({"token": token[:256], "name": name[:256]})
        return result

    def resolve(self) -> ConnectorResult:
        camera = self._camera()
        media = camera.create_media_service()

        profiles = media.GetProfiles()
        if not profiles:
            raise RuntimeError("ONVIF camera returned no media profiles")

        wanted = str(self.config.get("profile_token") or "").strip()
        selected = None

        for profile in profiles:
            token = str(getattr(profile, "token", "") or "").strip()
            if wanted and token == wanted:
                selected = profile
                break

        if selected is None:
            selected = profiles[0]

        token = str(getattr(selected, "token", "") or "").strip()
        if not token:
            raise RuntimeError("ONVIF media profile token is missing")

        request = {
            "StreamSetup": {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            },
            "ProfileToken": token,
        }

        uri_response = media.GetStreamUri(request)
        uri = str(getattr(uri_response, "Uri", "") or "").strip()

        parsed = urlparse(uri)
        if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
            raise RuntimeError("ONVIF returned an invalid RTSP stream URI")

        return ConnectorResult(
            source=uri,
            source_type="rtsp",
            metadata={
                "connector_type": "onvif",
                "profile_token": token[:256],
                "profile_name": str(getattr(selected, "Name", "") or "")[:256],
                "resolved_source_type": "rtsp",
            },
        )

    def health_check(self) -> dict[str, Any]:
        try:
            camera = self._camera()
            info = camera.devicemgmt.GetDeviceInformation()
            return {
                "ok": True,
                "manufacturer": str(getattr(info, "Manufacturer", "") or "")[:128],
                "model": str(getattr(info, "Model", "") or "")[:128],
                "firmware": str(getattr(info, "FirmwareVersion", "") or "")[:128],
            }
        except Exception:
            return {"ok": False}


def discover_onvif_devices(timeout_seconds: float = 3.0) -> list[dict[str, Any]]:
    """Perform a bounded WS-Discovery probe on the local network.

    This is intentionally discovery-only: credentials are not requested or
    returned. The result contains XAddrs advertised by devices.
    """
    timeout = min(max(float(timeout_seconds), 0.5), 5.0)
    message_id = f"urn:uuid:{uuid.uuid4()}"
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:a="http://www.w3.org/2005/08/addressing"
 xmlns:d="http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01">
 <s:Header>
  <a:Action>http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01/Probe</a:Action>
  <a:MessageID>{message_id}</a:MessageID>
  <a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>
 </s:Header>
 <s:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></s:Body>
</s:Envelope>""".replace(
        'xmlns:d="http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01"',
        'xmlns:d="http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01" xmlns:dn="http://www.onvif.org/ver10/network/wsdl"',
    )

    destination = ("239.255.255.250", 3702)
    results: dict[str, dict[str, Any]] = {}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(0.25)
        sock.sendto(envelope.encode("utf-8"), destination)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                payload, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue

            try:
                root = ET.fromstring(payload)
            except ET.ParseError:
                continue

            xaddrs = []
            epr = None
            for elem in root.iter():
                local = elem.tag.rsplit("}", 1)[-1]
                if local == "Address" and not epr:
                    epr = (elem.text or "").strip()
                elif local == "XAddrs":
                    xaddrs.extend((elem.text or "").split())

            if not xaddrs:
                continue

            key = epr or " ".join(xaddrs)
            results[key] = {
                "endpoint_reference": epr[:512] if epr else None,
                "xaddrs": [x[:2048] for x in xaddrs[:10]],
            }
    finally:
        sock.close()

    return list(results.values())
