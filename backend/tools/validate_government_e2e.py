#!/usr/bin/env python3
"""Run and capture the real Sentinel → INTEL-I acceptance path.

Credentials are read only from INTEL_I_DEMO_EMAIL / INTEL_I_DEMO_PASSWORD so
they do not appear in shell history or the operating-system process list.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import requests


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Validate a deployed INTEL-I Government-feed flow")
    command.add_argument("--base-url", required=True, help="Deployed backend origin, for example https://api.example.gov")
    command.add_argument("--integration-id", required=True, type=int)
    command.add_argument("--output-dir", default="government-e2e-evidence")
    command.add_argument("--timeout-seconds", type=int, default=300)
    command.add_argument("--poll-seconds", type=int, default=10)
    command.add_argument("--start-cameras", type=int, default=1)
    command.add_argument("--stop-after", action="store_true")
    command.add_argument("--insecure", action="store_true", help="Disable TLS validation only in an approved isolated test environment")
    return command


def request_json(session: requests.Session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parser().parse_args()
    email = os.getenv("INTEL_I_DEMO_EMAIL", "").strip()
    password = os.getenv("INTEL_I_DEMO_PASSWORD", "")
    if not email or not password:
        print("Set INTEL_I_DEMO_EMAIL and INTEL_I_DEMO_PASSWORD before running.", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    if not base_url.startswith("https://") and not args.insecure:
        print("HTTPS is required unless --insecure is explicitly supplied.", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.verify = not args.insecure
    started: list[str] = []
    manifest = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "integration_id": args.integration_id,
        "started_cameras": started,
        "steps": [],
        "complete": False,
    }

    try:
        login = request_json(session, "POST", f"{base_url}/auth/login", json={"email": email, "password": password})
        csrf = login.get("csrf_token") or session.cookies.get("csrf_token")
        if not csrf:
            raise RuntimeError("Login succeeded without a CSRF token")
        session.headers.update({"X-CSRF-Token": csrf})
        manifest["steps"].append({"name": "authenticated", "ok": True, "role": login.get("user", {}).get("role")})

        sync = request_json(session, "POST", f"{base_url}/api/integrations/{args.integration_id}/sync")
        manifest["steps"].append({"name": "sentinel_sync", "ok": sync.get("status") == "SUCCEEDED", "result": sync})

        inventory = request_json(session, "GET", f"{base_url}/cameras")
        sentinel_cameras = [camera for camera in inventory.get("cameras", []) if camera.get("external_provider") == "sentinel"]
        if not sentinel_cameras:
            raise RuntimeError("Synchronization returned no Sentinel cameras")
        manifest["steps"].append({
            "name": "camera_inventory",
            "ok": True,
            "count": len(sentinel_cameras),
            "cameras": [{key: camera.get(key) for key in (
                "cam_id", "camera_name", "external_camera_id", "external_live_status",
                "codec", "stream_width", "stream_height", "location_name",
            )} for camera in sentinel_cameras],
        })

        eligible = [camera for camera in sentinel_cameras if str(camera.get("external_live_status") or "").upper() not in {"OFFLINE", "MISSING"}]
        for camera in eligible[:max(1, min(args.start_cameras, 10))]:
            cam_id = camera["cam_id"]
            response = request_json(session, "POST", f"{base_url}/camera/{requests.utils.quote(cam_id, safe='')}/start")
            started.append(cam_id)
            manifest["steps"].append({"name": "camera_started", "ok": True, "cam_id": cam_id, "status": response.get("status")})
        if not started:
            raise RuntimeError("No online Sentinel camera was eligible to start")

        deadline = time.monotonic() + max(10, args.timeout_seconds)
        readiness = None
        while time.monotonic() < deadline:
            readiness = request_json(session, "GET", f"{base_url}/api/reports/government-e2e/readiness", params={"hours": 24})
            if readiness.get("complete"):
                break
            time.sleep(max(1, args.poll_seconds))
        manifest["readiness"] = readiness
        manifest["complete"] = bool(readiness and readiness.get("complete"))

        for report_format in ("pdf", "csv"):
            response = session.get(
                f"{base_url}/api/reports/analytics/export",
                params={"format": report_format, "camera_id": started[0], "limit": 10000},
                timeout=120,
            )
            response.raise_for_status()
            report_id = response.headers.get("X-Report-ID")
            path = output_dir / f"intel-i-government-e2e.{report_format}"
            path.write_bytes(response.content)
            manifest["steps"].append({"name": f"{report_format}_report_exported", "ok": True, "report_id": report_id, "file": path.name})
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if args.stop_after:
            for cam_id in started:
                try:
                    request_json(session, "POST", f"{base_url}/camera/{requests.utils.quote(cam_id, safe='')}/stop")
                except Exception as exc:
                    manifest["steps"].append({"name": "camera_stopped", "ok": False, "cam_id": cam_id, "error": type(exc).__name__})
        manifest["run_completed_at"] = datetime.now(timezone.utc).isoformat()
        (output_dir / "government-e2e-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"complete": manifest["complete"], "output_dir": str(output_dir), "error": manifest.get("error")}, indent=2))
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
